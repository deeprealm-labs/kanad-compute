"""IBM-style Sample-based Quantum Diagonalization (M4).

References:
- Robledo-Moreno et al., *Chemistry beyond exact solutions on a quantum-centric
  supercomputer*, Nature 638, 87 (2025) — 77-qubit [2Fe-2S] demonstration.
- Kanno et al., *Quantum-classical computation of Schwinger model dynamics*,
  arXiv:2308.04372 — earlier SQD-style.
- IBM's `qiskit-addon-sqd` package — reference implementation.

Why this exists (vs the existing `solvers/sqd_solver.py`):

The legacy `SQDSolver` builds a deterministic HF + singles + doubles basis as
explicit `2^n_qubits` statevectors, then projects the full Hamiltonian. That
approach (a) doesn't scale past ~10 qubits because the dense statevector
explodes, and (b) doesn't use a quantum circuit at all — the "Q" in SQD is
purely decorative.

IBM's actual SQD algorithm (Robledo-Moreno 2025):
  1. Prepare a state on a quantum device using a short, hardware-efficient
     ansatz (e.g. LUCJ — Local Unitary Cluster Jastrow).
  2. **Sample the circuit** in the computational basis. Each shot returns one
     bitstring corresponding to a Slater determinant.
  3. **Configuration recovery**: hardware noise causes bit flips → many
     samples land outside the (N, S_z) sector. Recover them by projecting
     to the nearest valid configuration (the simplest form), or drop them
     entirely (what we do here).
  4. The set of unique, valid bitstrings is the **selected-CI subspace**.
  5. Build the Hamiltonian matrix in this subspace via Slater-Condon rules.
     Cost is O(N_det^2 · n_orb^4) but with sparsity (≤ doubly-excited pairs
     have non-zero matrix elements) it's fast for N_det up to 10^5.
  6. Classically diagonalize for the lowest eigenvalue.
  7. Iterate: feed the dominant determinants of the eigenvector back into
     the next sampling round to expand the subspace.

This module implements the algorithm in `SamplingSQDSolver`. Two simulator
backends:
  - ``backend='statevector'``: sample from the exact probability distribution
    of the ansatz (no noise, deterministic).
  - ``backend='qasm'``: shot-based sampling on Qiskit Aer (zero-noise
    simulator with shot noise).

Real-hardware backends ('ibm', 'bluequbit') are M5 — they require the cloud
backend integration that already exists in `kanad-app` (separate product) but
isn't pulled into `kanad-framework` yet.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Callable, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from kanad.solvers.base_solver import BaseSolver
from kanad.solvers.capabilities import FiniteDifferenceForceMixin
from kanad.core.solver_result import SolverResult

logger = logging.getLogger(__name__)

# CODATA Bohr radius in Angstrom — same constant as physics_vqe.py / system_spec.py,
# so the per-geometry rebuild in energy_fn matches the builder/quantum_forces path.
_BOHR_TO_ANGSTROM = 0.52917721092

from kanad.core.ci.slater_condon import (
    _det_arr, _split_alpha_beta, _interleave_to_block_sign, _count_bits,
    _fermion_sign, _diff_spin_orbitals, _generate_singles_doubles, _count_bits_below,
    _double_excitation_sign, _slater_condon_offdiag, _build_sparse_h_subspace, _h_diag,
)
from kanad.core.ci.config_recovery import (
    _filter_by_n_sz, _recover_spin_sector, _filter_with_recovery,
    _compute_orbital_marginals, _filter_with_iterative_recovery,
)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _hf_circuit(n_qubits: int, n_electrons: int) -> QuantumCircuit:
    """Hartree-Fock reference circuit in JW interleaved convention.

    α at even qubits (0, 2, 4, ...), β at odd qubits (1, 3, 5, ...).
    The first ``n_electrons`` spin orbitals (sorted by energy) are occupied.
    For closed-shell singlets the occupied spin-orbitals interleave α/β.
    """
    qc = QuantumCircuit(n_qubits)
    for i in range(n_electrons):
        qc.x(i)
    return qc




def _sample_circuit_statevector(qc: QuantumCircuit, n_samples: int,
                                rng: np.random.Generator) -> np.ndarray:
    """Sample computational-basis bitstrings from the exact statevector of `qc`.

    Returns an int array of length `n_samples` with bitstring integers
    (Qiskit little-endian convention: bit 0 is qubit 0).

    Uses Aer's statevector method rather than ``Statevector.from_instruction``:
    the latter synthesizes any ``PauliEvolutionGate`` (LUCJ orbital rotations,
    Givens-SD doubles, …) via ``to_matrix()`` → scipy ``expm`` on the full 2ⁿ
    operator, which is single-threaded and hangs for n ≳ 16 qubits (observed:
    N₂ CAS(10,10)/20q ran >30 min with no result). Aer lowers the gate to
    native rotations and produces the identical statevector (overlap 1.0) in
    seconds.
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    sim = AerSimulator(method='statevector')
    qc_sv = qc.copy()
    qc_sv.save_statevector()
    sv = np.asarray(sim.run(transpile(qc_sv, sim)).result().get_statevector())
    probs = np.abs(sv) ** 2
    probs /= probs.sum()
    return rng.choice(len(probs), size=n_samples, p=probs)


def _sample_circuit_qasm(qc: QuantumCircuit, n_samples: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Sample via Qiskit Aer's qasm simulator (shot-based, no noise).

    Identical to statevector sampling in expectation, but uses the same code
    path that a real hardware run would. Used to test the algorithm under
    finite-shot statistics.
    """
    from qiskit_aer import AerSimulator
    qc_m = qc.copy()
    qc_m.measure_all()
    sim = AerSimulator()
    job = sim.run(qc_m, shots=n_samples, seed_simulator=int(rng.integers(1 << 31)))
    counts = job.result().get_counts()
    bitstrings = []
    for bitstr, n in counts.items():
        # Qiskit returns bitstring with qubit N-1 leftmost; convert to int.
        # The string may have spaces between classical registers.
        bstr = bitstr.replace(' ', '')
        val = int(bstr, 2)
        bitstrings.extend([val] * n)
    return _det_arr(bitstrings)


def _counts_to_bitstrings(counts: dict) -> np.ndarray:
    """Common helper: cloud-backend counts → int64 bitstring array."""
    out = []
    for bstr, n in counts.items():
        clean = str(bstr).replace(' ', '').replace('0x', '')
        out.extend([int(clean, 2)] * int(n))
    return _det_arr(out)


def _sample_circuit_bluequbit(qc: QuantumCircuit, n_samples: int,
                              api_key: Optional[str] = None,
                              device: str = 'cpu',
                              job_name: Optional[str] = None) -> np.ndarray:
    """Sample on BlueQubit cloud (CPU/GPU/MPS).

    Default `device='cpu'` is free on the BlueQubit free tier and handles
    ≤32 qubits at sub-second sampling. `device='gpu'` and `'mps'` are
    paid tiers (cost in $ per shot — DO NOT use without budget approval).

    Requires:
      - `bluequbit` package
      - API key via ``api_key=`` arg OR ``BLUEQUBIT_API_KEY`` env var
    """
    import os
    import bluequbit  # type: ignore
    api_key = api_key or os.environ.get('BLUEQUBIT_API_KEY')
    if not api_key:
        raise RuntimeError(
            "BlueQubit sampling requires `api_key=` or BLUEQUBIT_API_KEY env var."
        )
    bq = bluequbit.init(api_key)
    qc_m = qc.copy()
    if qc_m.num_clbits == 0:
        qc_m.measure_all()
    result = bq.run(
        circuits=qc_m, device=device, shots=n_samples,
        job_name=job_name or f'kanad-sqd-{device}',
    )
    counts = result.get_counts()
    logger.info(f"BlueQubit {device}: {n_samples} shots, job {result.job_id}")
    return _counts_to_bitstrings(counts)


def _sample_circuit_ibm(qc: QuantumCircuit, n_samples: int,
                         backend_name: Optional[str] = None,
                         token: Optional[str] = None,
                         instance: Optional[str] = None,
                         optimization_level: int = 1,
                         timeout_s: int = 3600,
                         poll_interval_s: int = 10,
                         on_submit=None) -> np.ndarray:
    """Sample on real IBM Quantum hardware (Heron r3 by default).

    Submits the circuit via the runtime SamplerV2 inside a Batch context,
    then polls until DONE (or timeout). For long-running jobs prefer
    submit-then-poll workflows in benchmark scripts — this synchronous
    path blocks until completion.

    Requires:
      - `qiskit_ibm_runtime` package
      - `IBM_QUANTUM_TOKEN` and `IBM_QUANTUM_CRN` env vars (or arg overrides)
      - `backend_name=None` → auto-least-busy operational non-simulator
        with ≥ ``qc.num_qubits`` qubits.
    """
    import os
    import time as _time
    from qiskit_ibm_runtime import QiskitRuntimeService, Batch, SamplerV2
    from qiskit import transpile

    token = token or os.environ.get('IBM_QUANTUM_TOKEN')
    instance = instance or os.environ.get('IBM_QUANTUM_CRN')
    if not token or not instance:
        raise RuntimeError(
            "IBM sampling requires `token=` + `instance=` or "
            "IBM_QUANTUM_TOKEN + IBM_QUANTUM_CRN env vars."
        )
    svc = QiskitRuntimeService(channel='ibm_cloud', token=token, instance=instance)
    if backend_name:
        backend = svc.backend(backend_name)
    else:
        backends = svc.backends(
            operational=True, simulator=False, min_num_qubits=qc.num_qubits,
        )
        if not backends:
            raise RuntimeError(
                f"No IBM hardware backend available with ≥ {qc.num_qubits} qubits."
            )
        backend = min(backends, key=lambda b: b.status().pending_jobs)
    logger.info(f"IBM backend: {backend.name} ({backend.num_qubits}q, "
                f"pending: {backend.status().pending_jobs})")

    qc_m = qc.copy()
    if qc_m.num_clbits == 0:
        qc_m.measure_all()
    ct = transpile(qc_m, backend=backend, optimization_level=optimization_level)
    n_2q = sum(1 for inst in ct.data if inst.operation.num_qubits == 2)
    logger.info(f"Transpiled depth={ct.depth()}, 2q gates={n_2q}")

    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch)
        job = sampler.run([ct], shots=n_samples)
    job_id = job.job_id()
    logger.info(f"IBM job submitted: {job_id}")
    if on_submit:
        try:
            on_submit({"job_id": job_id, "backend": backend.name,
                       "n_qubits": backend.num_qubits, "depth": ct.depth(), "n_2q": n_2q})
        except Exception:
            pass

    deadline = _time.time() + timeout_s
    while _time.time() < deadline:
        status = str(job.status())
        if status in ('DONE', 'CANCELLED', 'ERROR'):
            break
        _time.sleep(poll_interval_s)
    if str(job.status()) != 'DONE':
        raise RuntimeError(
            f"IBM job {job_id} did not complete in {timeout_s}s "
            f"(final status: {job.status()}). Use submit/poll workflow for "
            f"long-queue jobs."
        )
    result = job.result()
    pub = result[0]
    if hasattr(pub.data, 'meas'):
        counts = pub.data.meas.get_counts()
    else:
        field = list(pub.data.keys())[0]
        counts = getattr(pub.data, field).get_counts()
    return _counts_to_bitstrings(counts)


# ---------------------------------------------------------------------------
# Configuration recovery + filtering
# ---------------------------------------------------------------------------

















# ---------------------------------------------------------------------------
# Slater-Condon matrix elements
# ---------------------------------------------------------------------------

















# ---------------------------------------------------------------------------
# Sampling-based SQD solver
# ---------------------------------------------------------------------------

class SamplingSQDSolver(FiniteDifferenceForceMixin, BaseSolver):
    """IBM-style sample-based quantum diagonalization.

    Parameters
    ----------
    hamiltonian : ActiveHamiltonian or CovalentHamiltonian (the solver ``system``)
        Provides ``h_core`` and ``eri`` in MO basis (chemist's notation) and
        ``n_orbitals``, ``n_electrons``, ``nuclear_repulsion``. May also be a
        Bond / builder QuantumSystem exposing ``.hamiltonian`` — the unified
        solver protocol's ``BaseSolver._resolve_system`` normalizes it.
    n_samples : int
        Total samples drawn from the ansatz circuit.
    n_iterations : int
        Subspace-expansion iterations. v1 uses a single pass (n_iterations=1);
        the iteration-feedback path is M5.
    backend : str
        ``'statevector'`` (default, exact probability distribution) or
        ``'qasm'`` (shot-based simulator). ``'bluequbit'`` / ``'ibm'`` route to
        the cloud sampling helpers. NOTE: ``self.backend`` is now the
        :class:`~kanad.backends.base_backend.BaseBackend` object (unified solver
        protocol); the original string lives on ``self.backend_name`` and drives
        the sampling dispatch.
    target_sz : float
        Target ``S_z = (n_α − n_β) / 2``. Default 0 (closed-shell singlet).
    random_seed : int | None
        Seeds NumPy and Qiskit Aer for reproducibility.
    """

    def __init__(
        self,
        hamiltonian,
        *,
        n_samples: int = 10000,
        n_iterations: int = 1,
        backend: str = 'statevector',
        target_sz: float = 0.0,
        spin_s: Optional[float] = None,
        cisd_seed: bool = False,
        random_seed: Optional[int] = None,
        recover_configurations: bool = False,
        ci_backend: str = 'pyscf',
        bq_device: str = 'cpu',
        ibm_backend_name: Optional[str] = None,
        ibm_timeout_s: int = 3600,
        ibm_api_token: Optional[str] = None,
        ibm_crn: Optional[str] = None,
        gpu_device: str = 'auto',
        job_name: Optional[str] = None,
        recovery_rounds: int = 1,
        recovery_tol: float = 1e-5,
    ):
        if backend not in ('statevector', 'qasm', 'bluequbit', 'ibm'):
            raise ValueError(
                f"backend must be 'statevector', 'qasm', 'bluequbit', or 'ibm'; "
                f"got {backend!r}"
            )
        # The legacy sampling-backend string ('statevector'|'qasm'|'bluequbit'|
        # 'ibm') drives `solve()`'s sampling dispatch. SQD owns its own sampling
        # helpers (incl. 'qasm', which the make_backend factory does not know),
        # so build the BaseBackend object on the statevector path and preserve
        # the requested string in self.backend_name (set just below). The cloud
        # job-submission paths remain string-routed through the SQD helpers.
        from kanad.backends.statevector_backend import StatevectorBackend
        try:
            super().__init__(
                hamiltonian,
                backend='statevector',
                enable_analysis=False,
                enable_optimization=False,
            )
        except TypeError:
            # SQD historically accepts any duck-typed Hamiltonian exposing
            # `h_core`/`eri`/`n_orbitals`/`n_electrons`/`nuclear_repulsion`
            # (notably `ActiveHamiltonian`, an active-space Hamiltonian that is
            # NOT a MolecularHamiltonian subclass and so isn't recognized by
            # `BaseSolver._resolve_system`). Preserve that contract: set the
            # solver attributes directly and build the backend ourselves.
            if not hasattr(hamiltonian, 'h_core'):
                raise  # genuinely unsupported input — surface the original error
            self.enable_analysis = False
            self.enable_optimization = False
            self.bond = None
            self.hamiltonian = hamiltonian
            self.molecule = getattr(hamiltonian, 'molecule', None)
            self.atoms = getattr(hamiltonian, 'atoms', [])
            self._bond_type = 'molecular'
            from kanad.backends.factory import make_backend
            self.backend = make_backend('statevector')
            self.backend_name = self.backend.name
            self.results = {}
        # Override BaseSolver's backend_name with the SQD-requested sampling
        # backend; `self.backend` stays the StatevectorBackend object so the
        # unified-protocol invariant (`self.backend` is a BaseBackend) holds.
        self.backend_name = backend
        self._use_statevector = isinstance(self.backend, StatevectorBackend)

        # `self.hamiltonian` is set by BaseSolver._resolve_system (or the
        # duck-typed fallback) above.
        hamiltonian = self.hamiltonian
        self.n_samples = int(n_samples)
        self.n_iterations = int(n_iterations)
        self.target_sz = float(target_sz)
        # Optional total-spin target S (NOT 2S). When set, the subspace
        # diagonalization adds an S² penalty so the recovered state has the
        # requested multiplicity — fixing the spin-contamination failure where,
        # on near-degenerate singlet/triplet diradicals, the M_s-sector subspace
        # diagonalizer returns the LOWER (triplet) M_s=0 component, dropping the
        # energy BELOW the spin-pure singlet reference (e.g. twisted ethylene was
        # 180 mHa below CASCI(ss=0)). Default None = no penalty (fast path, exact
        # for spin-clean systems); ⟨S²⟩ is reported and a warning emitted on
        # contamination regardless, so the issue is never silent.
        self.spin_s = (float(spin_s) if spin_s is not None else None)
        # Deterministic CISD seed: union the sampled subspace with HF + all
        # (N,Sz)-preserving singles & doubles. Closes the strong-correlation
        # completeness gap where flat near-degenerate sampling misses dominant
        # configs (N₂/H-chain dissociation left tens-to-hundreds of mHa). This is
        # a CLASSICAL completeness guarantee — it makes results robust for strong
        # correlation, but means the energy floor is CISD-in-active-space, not a
        # pure quantum-sample result (report both when claiming QPU advantage).
        self.cisd_seed = bool(cisd_seed)
        self.rng = np.random.default_rng(random_seed)
        self.recover_configurations = bool(recover_configurations)
        # Cloud-backend configuration (used only when backend in ('bluequbit', 'ibm')):
        self.bq_device = bq_device              # 'cpu' (free) | 'gpu' | 'mps'
        self.ibm_backend_name = ibm_backend_name  # None = auto-least-busy Heron
        self.ibm_timeout_s = int(ibm_timeout_s)
        # User's IBM credentials for QPU sampling (passed through from kanad-app via
        # the compute node; falls back to env vars in _sample_circuit_ibm if None).
        self.ibm_api_token = ibm_api_token
        self.ibm_crn = ibm_crn
        # GPU device for the subspace diagonalization (rocm-planck det_ci):
        # 'auto'|'amd'|'nvidia'|'cpu'. Threaded into selected_ci.diagonalize_*.
        self.gpu_device = gpu_device
        self.job_name = job_name
        # M3 D2: multi-round confidence-weighted recovery
        # recovery_rounds=1 → current single-shot greedy (M4-D, backwards-compat).
        # recovery_rounds>1 → iterative recovery using eigenvector marginals as
        # the bit-flip preference signal.
        self.recovery_rounds = int(recovery_rounds)
        self.recovery_tol = float(recovery_tol)
        # CI matrix construction backend.
        # - 'pyscf' (default): use PySCF's direct_spin1.contract_2e — correct
        #   for any active-space size, slightly slower per matrix element.
        # - 'custom': homegrown bit-level Slater-Condon (sparse, K-scaled). The
        #   αβ-mixed-doubles sign was fixed; now validated == pyscf FCI to ~1e-14
        #   on full sectors WITH αβ doubles (N₂ CAS(6,6), H₂O CAS(4,4)) — see
        #   tests/validation/test_tier1_fixes.py::test_c1_*. Use it for the
        #   large/dilute selected-CI scale path where the full-tensor pyscf path
        #   would OOM.
        if ci_backend not in ('pyscf', 'custom'):
            raise ValueError(f"ci_backend must be 'pyscf' or 'custom', got {ci_backend!r}")
        self.ci_backend = ci_backend
        self.results: dict = {}

        # Per-geometry settings the energy_fn closure reuses when computing forces
        # (ForceProvider capability). These mirror the validated quantum-forces MD
        # pattern (tests/validation/test_quantum_forces_md.py).
        self._fd_iter_kwargs = dict(max_iterations=4, expansion_per_round=30, energy_tol=1e-9)
        self._fd_n_layers = 1

        # MO energies for configuration recovery (when enabled).
        # ActiveHamiltonian: mf.mo_energy[active_indices]
        # CovalentHamiltonian: _mo_energies
        # MolecularHamiltonian: mf.mo_energy
        self._mo_energies = None
        if self.recover_configurations:
            self._mo_energies = self._resolve_mo_energies()

        # Resolve MO-basis integrals once at construction.
        # - `ActiveHamiltonian.h_core` / `.eri` are already MO basis (active space).
        # - `CovalentHamiltonian.h_core` is AO basis — must transform via
        #   `_get_mo_integrals()`.
        # - Polyatomic `MolecularHamiltonian` (core/molecule.py) ditto.
        if hasattr(hamiltonian, '_get_mo_integrals'):
            self._h1, self._h2 = hamiltonian._get_mo_integrals()
        elif hasattr(hamiltonian, 'active_space'):
            # ActiveHamiltonian — `h_core` and `eri` already in (active) MO basis
            self._h1 = np.asarray(hamiltonian.h_core, dtype=float)
            self._h2 = np.asarray(hamiltonian.eri, dtype=float)
        else:
            # Best-effort fallback: try a one-shot transform via mf.mo_coeff
            if hasattr(hamiltonian, 'mf') and hamiltonian.mf is not None:
                from kanad.core.integrals.transforms import ao2mo_transform_from_mol, one_index_transform
                C = hamiltonian.mf.mo_coeff
                self._h1 = one_index_transform(hamiltonian.h_core, C)
                self._h2 = ao2mo_transform_from_mol(hamiltonian.mol, C)
            else:
                # Last resort — assume already MO basis
                self._h1 = np.asarray(hamiltonian.h_core, dtype=float)
                self._h2 = np.asarray(hamiltonian.eri, dtype=float)

    def _resolve_mo_energies(self) -> Optional[np.ndarray]:
        """Resolve MO energies for the active orbital subset."""
        ham = self.hamiltonian
        if hasattr(ham, 'active_space'):
            # ActiveHamiltonian — pluck active subset of mf.mo_energy
            mf = ham.mf
            active = list(ham.active_space.active_indices)
            return np.asarray(mf.mo_energy)[active]
        if hasattr(ham, '_mo_energies') and ham._mo_energies is not None:
            return np.asarray(ham._mo_energies)
        if hasattr(ham, 'mf') and ham.mf is not None:
            return np.asarray(ham.mf.mo_energy)
        logger.warning(
            "Could not resolve MO energies; configuration recovery will use "
            "lowest-index ordering instead."
        )
        return None

    # ------ public API -----------------------------------------------------

    def solve_iterative(
        self,
        ansatz_circuit: Optional[QuantumCircuit] = None,
        max_iterations: int = 5,
        expansion_per_round: int = 100,
        energy_tol: float = 1e-4,
        seed_determinants: Optional[list] = None,
    ) -> SolverResult:
        """Iterative subspace expansion (M4-C).

        After each round:
        1. Identify the top ``expansion_per_round`` determinants by |c_i|²
           in the current eigenvector.
        2. For each, generate all single + double excitations and add the
           new determinants to the subspace.
        3. Rebuild + rediagonalize the CI matrix.
        4. Repeat until energy converges (|ΔE| < energy_tol) or
           ``max_iterations`` reached.

        This is the classical-postprocessing variant of subspace expansion
        from Robledo-Moreno 2025 §III.B. The quantum re-sampling variant
        (re-run circuit biased toward top dets) is M5.

        Returns: same dict as ``solve()`` plus ``iterations_done`` and
        ``energy_history``.
        """
        # Initial subspace from sampling (raw dict — we subscript + expand it)
        result = self._solve_raw(ansatz_circuit=ansatz_circuit)
        history = [result['energy']]
        determinants = list(result['determinants'])

        n_orb = self.hamiltonian.n_orbitals
        n_qubits = 2 * n_orb
        n_elec = self.hamiltonian.n_electrons

        # Warm-start: seed the subspace with determinants from a previous solve
        # (e.g. the prior geometry in a scan / MD step). Strictly additive, so
        # the variational energy can only improve — and the expansion converges
        # in fewer rounds. Re-diagonalize once on the seeded subspace.
        if seed_determinants:
            seeded = sorted(set(determinants) | {int(d) for d in seed_determinants})
            if len(seeded) > len(determinants):
                determinants = seeded
                result = (self._diagonalize_in_subspace_pyscf(determinants)
                          if self.ci_backend == 'pyscf'
                          else self._diagonalize_in_subspace(determinants))
                history = [result['energy']]

        for it in range(max_iterations):
            # Top-K by |c|² in current eigenvector
            evec = result['eigenvector']
            top_indices = np.argsort(np.abs(evec) ** 2)[::-1][:expansion_per_round]
            seed_dets = [determinants[i] for i in top_indices]

            # Generate all single + double excitations
            new_dets = set()
            for d in seed_dets:
                new_dets.update(_generate_singles_doubles(d, n_qubits, n_elec))

            old_size = len(determinants)
            determinants = sorted(set(determinants) | new_dets)
            new_size = len(determinants)
            if new_size == old_size:
                logger.info(f"Iteration {it+1}: no new determinants — converged")
                break

            # Rediagonalize on the expanded subspace (use chosen backend)
            if self.ci_backend == 'pyscf':
                result = self._diagonalize_in_subspace_pyscf(determinants)
            else:
                result = self._diagonalize_in_subspace(determinants)
            history.append(result['energy'])
            dE = abs(history[-1] - history[-2])
            logger.info(
                f"Iteration {it+1}: |dets| = {old_size} → {new_size}, "
                f"E = {history[-1]:.8f} (ΔE = {dE*1000:.3f} mHa)"
            )
            if dE < energy_tol:
                logger.info(f"Iteration {it+1}: converged within {energy_tol:.1e} Ha")
                break

        result['iterations_done'] = len(history) - 1
        result['energy_history'] = history
        self.results = result
        return SolverResult.from_mapping(
            result, solver="sampling_sqd", backend=self.backend_name,
        )

    def _diagonalize_in_subspace_pyscf(self, determinants: list) -> dict:
        """Delegate to the indigenous core.ci.selected_ci.diagonalize_pyscf."""
        from kanad.core.ci.selected_ci import diagonalize_pyscf
        r = diagonalize_pyscf(
            determinants, self._h1, self._h2,
            float(self.hamiltonian.nuclear_repulsion),
            self.hamiltonian.n_orbitals, self.hamiltonian.n_electrons,
            self.target_sz, self.spin_s,
            device=getattr(self, 'gpu_device', 'auto'),
        )
        # Record which device the subspace diagonalization actually ran on
        # (rocm-planck det_ci 'amd'/'nvidia' vs native scipy 'cpu') for honest reporting.
        self._diag_device_used = r.get('device_used') or 'cpu'
        return r

    def _diagonalize_in_subspace(self, determinants: list) -> dict:
        """Delegate to the indigenous core.ci.selected_ci.diagonalize_custom."""
        from kanad.core.ci.selected_ci import diagonalize_custom
        r = diagonalize_custom(
            determinants, self._h1, self._h2,
            float(self.hamiltonian.nuclear_repulsion),
            self.hamiltonian.n_orbitals, self.hamiltonian.n_electrons,
            device=getattr(self, 'gpu_device', 'auto'),
        )
        self._diag_device_used = r.get('device_used') or 'cpu'
        return r

    def energy_fn(self) -> Callable[[np.ndarray, Optional[Any]], tuple]:
        """Geometry-parametric energy closure for the md/reaction domains (ForceProvider).

        Returns ``energy_fn(atoms_bohr (n,3), warm_state) -> (energy_Ha, warm_state)``:
        rebuild the molecule at the displaced geometry, re-run SCF + active-space, prepare
        a fresh correlated LUCJ seed, and re-solve via iterative SQD. The
        ``FiniteDifferenceForceMixin`` central-differences this (delta=0.01 Bohr) to forces.
        The warm_state (prior geometry's recovered determinant subspace) seeds the next
        solve. Always runs on the statevector backend (FD must never dispatch cloud jobs).

        Honesty/limits: the active-space partition is reproduced with the *canonical* RHF
        MOs (manual frozen/active indices); AVAS/frontier rotated spaces are NOT exactly
        reproduced at displaced geometries, so forces are reliable for full / contiguous
        frozen-core spaces (matching the builder freeze policy).
        """
        import pyscf
        from pyscf import scf

        mol0 = getattr(self.hamiltonian, 'mol', None) or getattr(
            getattr(self.hamiltonian, 'mf', None), 'mol', None
        )
        if mol0 is None:
            raise RuntimeError(
                "SamplingSQDSolver.energy_fn: no PySCF molecule available to rebuild "
                "geometry (Hamiltonian exposes neither .mol nor .mf.mol)."
            )
        symbols = [mol0.atom_symbol(i) for i in range(mol0.natm)]
        basis, charge, spin = mol0.basis, mol0.charge, mol0.spin
        aspace = getattr(self.hamiltonian, 'active_space', None)
        frozen = list(aspace.frozen_indices) if aspace is not None else None
        active = list(aspace.active_indices) if aspace is not None else None
        cfg = dict(
            n_samples=self.n_samples, ci_backend=self.ci_backend,
            target_sz=self.target_sz, random_seed=0,
        )
        iter_kwargs = dict(self._fd_iter_kwargs)
        n_layers = self._fd_n_layers

        def _energy(atoms_bohr, warm_state=None):
            from kanad.core.active_space import (
                ActiveSpaceSelector, build_active_space_hamiltonian,
            )
            from kanad.core.ansatze import LUCJAnsatz
            c = np.asarray(atoms_bohr, dtype=float) * _BOHR_TO_ANGSTROM
            atomstr = '; '.join(
                f'{s} {c[i, 0]:.12f} {c[i, 1]:.12f} {c[i, 2]:.12f}'
                for i, s in enumerate(symbols)
            )
            m = pyscf.gto.M(atom=atomstr, basis=basis, charge=charge, spin=spin, verbose=0)
            mf = (scf.ROHF(m) if spin else scf.RHF(m)).run(verbose=0)
            sel = ActiveSpaceSelector(mf).manual(
                frozen=frozen if frozen is not None else [],
                active=active if active is not None else list(range(m.nao_nr())),
            )
            ham = build_active_space_hamiltonian(mf, sel)
            ansatz = LUCJAnsatz(
                n_qubits=2 * ham.n_orbitals, n_electrons=ham.n_electrons,
                n_layers=n_layers,
            )
            qc = ansatz.build_circuit()
            rng = np.random.default_rng(cfg['random_seed'])
            params = rng.uniform(-0.8, 0.8, size=qc.num_parameters)
            bound = qc.assign_parameters(
                {qc.parameters[i]: float(params[i]) for i in range(qc.num_parameters)}
            )
            # Fresh statevector SQD solver at this geometry (never a cloud backend).
            solver = SamplingSQDSolver(
                ham, n_samples=cfg['n_samples'], backend='statevector',
                recover_configurations=True, ci_backend=cfg['ci_backend'],
                target_sz=cfg['target_sz'], random_seed=cfg['random_seed'],
            )
            res = solver.solve_iterative(
                ansatz_circuit=bound, seed_determinants=warm_state, **iter_kwargs,
            ).to_dict()
            return float(res['energy']), res.get('determinants')

        return _energy

    def solve(self, ansatz_circuit: Optional[QuantumCircuit] = None) -> SolverResult:
        """Run sample-based SQD and return a unified :class:`SolverResult`.

        Args:
            ansatz_circuit: Qiskit circuit preparing the seed state. If None,
                uses the Hartree-Fock reference. For real-hardware runs,
                pass a LUCJ or hardware-efficient ansatz to explore beyond HF.

        Returns:
            :class:`~kanad.core.solver_result.SolverResult` whose ``.energy`` is
            the ground-state energy (Ha). Solver-specific fields
            (``determinants``, ``eigenvector``, ``valid_fraction``,
            ``n_determinants``, ...) live in ``.extra`` and flatten to the top
            level under ``.to_dict()``. The full legacy dict is preserved on
            ``self.results``.
        """
        raw = self._solve_raw(ansatz_circuit=ansatz_circuit)
        return SolverResult.from_mapping(
            raw, solver="sampling_sqd", backend=self.backend_name,
        )

    def _solve_raw(self, ansatz_circuit: Optional[QuantumCircuit] = None) -> dict:
        """Internal solve returning the legacy result dict (see ``solve``).

        Kept as the raw-dict entry point so the iterative / excited-state
        methods that expand the determinant subspace can subscript the result
        in place without round-tripping through SolverResult.
        """
        n_orb = self.hamiltonian.n_orbitals
        n_elec = self.hamiltonian.n_electrons
        n_qubits = 2 * n_orb

        if ansatz_circuit is None:
            ansatz_circuit = _hf_circuit(n_qubits, n_elec)

        # Guard against a VACUOUS ansatz (U2). A circuit with no entangling
        # (2-qubit) gates prepares a product state — sampling it yields a single
        # determinant (the Hartree-Fock reference), i.e. a fake "SQD". This nearly
        # got submitted to a real QPU. Refuse on hardware backends; warn otherwise.
        n_entangling = sum(1 for inst in ansatz_circuit.data
                           if inst.operation.num_qubits >= 2)
        if n_entangling == 0:
            _msg = (
                "SQD ansatz circuit has no entangling (2-qubit) gates — sampling it "
                "yields a single determinant (the Hartree-Fock reference), i.e. a vacuous "
                "'SQD'. Pass a correlated ansatz (e.g. LUCJ / Givens-SD) via ansatz_circuit=."
            )
            if self.backend_name in ('ibm', 'bluequbit'):
                raise ValueError(_msg + " Refusing to submit a trivial circuit to a QPU.")
            logger.warning(_msg)

        # MO-basis integrals (resolved at construction time)
        h1 = self._h1
        h2 = self._h2
        nuc = float(self.hamiltonian.nuclear_repulsion)

        # 1. Sample. `self.backend_name` is the SQD sampling-backend string
        # (`self.backend` is now the BaseBackend object under the unified
        # solver protocol). The statevector path can also route through the
        # backend object's `sample()`, but the existing helper preserves the
        # exact RNG stream / Aer fast path, so it stays as the canonical path.
        if self.backend_name == 'statevector':
            samples = _sample_circuit_statevector(ansatz_circuit, self.n_samples, self.rng)
        elif self.backend_name == 'qasm':
            samples = _sample_circuit_qasm(ansatz_circuit, self.n_samples, self.rng)
        elif self.backend_name == 'bluequbit':
            samples = _sample_circuit_bluequbit(
                ansatz_circuit, self.n_samples,
                device=getattr(self, 'bq_device', 'cpu'),
                job_name=getattr(self, 'job_name', None),
            )
        elif self.backend_name == 'ibm':
            samples = _sample_circuit_ibm(
                ansatz_circuit, self.n_samples,
                backend_name=getattr(self, 'ibm_backend_name', None),
                token=getattr(self, 'ibm_api_token', None),
                instance=getattr(self, 'ibm_crn', None),
                timeout_s=getattr(self, 'ibm_timeout_s', 3600),
                on_submit=getattr(self, '_on_ibm_submit', None),
            )
        else:
            raise NotImplementedError(
                f"backend={self.backend_name!r} not supported. "
                "Use 'statevector', 'qasm', 'bluequbit', or 'ibm'."
            )

        # 2. Filter by (N, Sz), optionally with configuration recovery
        recovery_history = []
        if self.recover_configurations and self.recovery_rounds > 1:
            # M3 D2: multi-round confidence-weighted recovery
            valid, recovery_history = _filter_with_iterative_recovery(
                samples, n_orb, n_elec,
                diagonalize_callback=self._diagonalize_in_subspace_pyscf,
                target_sz=self.target_sz, mo_energies=self._mo_energies,
                max_rounds=self.recovery_rounds,
                energy_tol=self.recovery_tol,
            )
            n_kept = recovery_history[0]['n_valid_total'] if recovery_history else 0
            n_recovered = len(valid) - n_kept
            n_dropped = len(samples) - len(valid)
            logger.info(
                f"SamplingSQD multi-round recovery: {self.recovery_rounds} rounds, "
                f"final valid={len(valid)}/{len(samples)} "
                f"({100*len(valid)/max(len(samples),1):.1f}%)"
            )
        elif self.recover_configurations:
            # M4-D: single-shot greedy recovery (default)
            valid, n_kept, n_recovered, n_dropped = _filter_with_recovery(
                samples, n_orb, n_elec, self.target_sz, self._mo_energies,
            )
            logger.info(
                f"SamplingSQD recovery: {n_kept} valid, {n_recovered} recovered, "
                f"{n_dropped} dropped (of {len(samples)} samples)"
            )
        else:
            valid = _filter_by_n_sz(samples, n_orb, n_elec, self.target_sz)
            n_kept, n_recovered, n_dropped = len(valid), 0, len(samples) - len(valid)
        valid_fraction = float(len(valid) / max(len(samples), 1))
        if len(valid) == 0:
            raise RuntimeError(
                f"No samples survived (N, S_z) filtering. "
                "Likely the ansatz circuit doesn't conserve particle number — "
                "use HF or a Givens-only ansatz."
            )

        # 3. Unique determinants = selected-CI subspace
        determinants = set(int(d) for d in valid)
        if self.cisd_seed:
            n_orb = self.hamiltonian.n_orbitals
            nq = 2 * n_orb
            na = (n_elec + int(round(2 * self.target_sz))) // 2
            nb = n_elec - na
            hf = 0
            for p in range(na):
                hf |= (1 << (2 * p))        # α occupied at even qubits
            for p in range(nb):
                hf |= (1 << (2 * p + 1))    # β occupied at odd qubits
            determinants |= {hf} | _generate_singles_doubles(hf, nq, n_elec)
        determinants = sorted(determinants)
        n_det = len(determinants)
        logger.info(
            f"SamplingSQD: {self.n_samples} samples → "
            f"{len(valid)} valid ({100*valid_fraction:.1f}%) → "
            f"{n_det} unique determinants"
        )

        # 4-5. CI matrix construction + diagonalization.
        # Use PySCF backend by default (correct for any active space size);
        # 'custom' available for small systems where it's verified.
        if self.ci_backend == 'pyscf':
            sub_result = self._diagonalize_in_subspace_pyscf(determinants)
        else:
            sub_result = self._diagonalize_in_subspace(determinants)

        self.results = {
            'energy': sub_result['energy'],
            'n_determinants': n_det,
            'n_samples': self.n_samples,
            'n_valid_samples': int(len(valid)),
            'valid_fraction': valid_fraction,
            'n_kept_directly': int(n_kept),
            'n_recovered': int(n_recovered),
            'n_dropped': int(n_dropped),
            'determinants': determinants,
            'eigenvector': sub_result['eigenvector'],
            'iterations': 1,
            'backend': self.backend_name,
        }
        logger.info(
            f"SamplingSQD: ground-state energy = {sub_result['energy']:.8f} Ha "
            f"(selected-CI dim = {n_det})"
        )

        # M4 D1 (2026-05-28): auto-extract wavefunction-derived 1-RDM and
        # store on the hamiltonian so PropertyCalculator can compute dipole,
        # polarizability, NMR, IR, charges, NO occupations from the REAL
        # SQD density (not silently from HF). Gate on N_FCI to avoid OOM
        # on huge active spaces (CAS(20,20) → 1.3 GB embed vector).
        try:
            n_a = (n_elec + int(round(2 * self.target_sz))) // 2
            n_b = n_elec - n_a
            from math import comb
            n_fci = comb(n_orb, n_a) * comb(n_orb, n_b)
            RDM_MAX_FCI = 100_000_000   # ~800 MB embed vector ceiling
            if n_fci <= RDM_MAX_FCI:
                self.populate_hamiltonian_density()
                logger.info(
                    f"SamplingSQD: 1-RDM extracted, trace = "
                    f"{float(np.trace(self.results['quantum_1rdm'])):.6f} "
                    f"(n_e_active = {n_elec})"
                )
            else:
                logger.info(
                    f"SamplingSQD: skipping 1-RDM extraction "
                    f"(N_FCI = {n_fci:,} > {RDM_MAX_FCI:,}; "
                    f"call get_1rdm_active_mo() manually on a big-RAM machine)"
                )
        except Exception as e:
            logger.warning(
                f"SamplingSQD: 1-RDM extraction failed ({type(e).__name__}: {e}); "
                "observables will fall back to HF density"
            )

        return self.results

    # ------------------------------------------------------------------
    # M4 D1 — Wavefunction-derived observables (2026-05-28)
    # ------------------------------------------------------------------

    def get_1rdm_active_mo(self) -> np.ndarray:
        """Spin-summed 1-RDM in the active-space MO basis.

        Built from the converged selected-CI eigenvector. The eigenvector is
        embedded into PySCF's full FCI layout (zeros at non-sampled positions),
        then ``direct_spin1.make_rdm1`` does the canonical Slater–Condon sum.

        Memory note: the embedded full FCI vector is ``O(N_FCI)`` floats.
        For ``CAS(16, 16)``: 165 MB. For ``CAS(20, 20)``: 1.3 GB.
        Use a cluster-class machine above ``CAS(16, 16)``.

        Returns:
            ``rdm1`` — numpy array shape ``(n_orb, n_orb)``, trace =
            ``n_active_electrons``.
        """
        from kanad.core.density.sampled_rdm import rdm1_from_ci_vector

        if not self.results:
            raise RuntimeError(
                "SamplingSQDSolver.get_1rdm_active_mo: call solve() first."
            )
        dets = self.results['determinants']
        evec = self.results['eigenvector']
        if evec is None or len(evec) == 0:
            raise RuntimeError("get_1rdm_active_mo: empty eigenvector in results.")

        n_e = self.hamiltonian.n_electrons
        n_a = (n_e + int(round(2 * self.target_sz))) // 2
        n_b = n_e - n_a
        return rdm1_from_ci_vector(
            dets, evec, self.hamiltonian.n_orbitals, n_a, n_b, n_e_expected=n_e,
        )

    def get_2rdm_active_mo(self) -> np.ndarray:
        """Spin-summed 2-RDM in the active-space MO basis (chemist's notation).

        ``rdm2[p, q, r, s] = ⟨ψ| a†_p a†_r a_s a_q |ψ⟩``

        Computed via the same selected-CI embedding as ``get_1rdm_active_mo``
        but with ``direct_spin1.make_rdm12``. Memory cost dominated by the
        full FCI vector (same as 1-RDM); the 2-RDM itself is
        ``O(n_orb⁴)`` which is small for typical active spaces.
        """
        from kanad.core.density.sampled_rdm import rdm12_from_ci_vector

        if not self.results:
            raise RuntimeError("Call solve() before extracting 2-RDM.")
        dets = self.results['determinants']
        evec = self.results['eigenvector']
        n_e = self.hamiltonian.n_electrons
        n_a = (n_e + int(round(2 * self.target_sz))) // 2
        n_b = n_e - n_a
        _, rdm2 = rdm12_from_ci_vector(
            dets, evec, self.hamiltonian.n_orbitals, n_a, n_b,
        )
        return rdm2

    def solve_excited_states(self, n_states: int = 3) -> dict:
        """Low-lying spectrum from the selected-CI subspace (M5 D4).

        Re-diagonalizes the CONVERGED determinant subspace for the lowest
        ``n_states`` eigenvalues. This is the photodynamics foundation:
        vertical excitation energies, S₁/T₁ gaps, excited-state PES — all
        derive from the spectrum of the same subspace SQD already built.

        Requires a prior ``solve()`` / ``solve_iterative()`` call to set up
        ``self.results['determinants']``. For physical excited states the
        subspace must be rich enough to span the target states — run
        ``solve_iterative`` with enough expansion first.

        Returns:
            dict with ``energies`` (list of n_states, ascending),
            ``excitation_energies_ha`` (E_i − E_0), ``excitation_energies_ev``,
            ``eigenvectors`` (columns), ``n_determinants``.
        """
        from scipy.sparse.linalg import eigsh as _sp_eigsh

        if not self.results or 'determinants' not in self.results:
            raise RuntimeError("Call solve()/solve_iterative() before solve_excited_states().")
        dets = self.results['determinants']
        n_det = len(dets)
        if n_det < n_states:
            raise RuntimeError(
                f"Subspace has {n_det} dets but {n_states} states requested. "
                "Expand the subspace (solve_iterative) first."
            )

        h1, h2 = self._h1, self._h2
        nuc = float(self.hamiltonian.nuclear_repulsion)
        n_orb = self.hamiltonian.n_orbitals

        # Build the sparse SC matrix once, get k lowest roots
        H_sparse, n_nz = _build_sparse_h_subspace(dets, h1, h2, nuc, n_orb)
        k = min(n_states, n_det - 1)
        ncv = max(2 * k + 1, min(n_det - 1, 50))
        evals, evecs = _sp_eigsh(H_sparse, k=k, which='SA', tol=1e-8, ncv=ncv)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

        # `_build_sparse_h_subspace` builds H in interleaved-JW convention; PySCF's
        # make_rdm1 (used downstream by get_1rdm_active_mo) assumes block ordering.
        # The ground-state path corrects every eigenvector by these per-determinant
        # signs; the excited-states path skipped it, corrupting any RDM/property taken
        # after this call. Correct ALL columns. (CORE_BUGS B17.)
        signs = np.array([_interleave_to_block_sign(int(d), n_orb) for d in dets])
        evecs = evecs * signs[:, None]

        HA_TO_EV = 27.211386245988
        e0 = float(evals[0])
        return {
            'energies': [float(e) for e in evals],
            'excitation_energies_ha': [float(e - e0) for e in evals],
            'excitation_energies_ev': [float((e - e0) * HA_TO_EV) for e in evals],
            'eigenvectors': evecs,
            'n_determinants': n_det,
            'n_states': k,
        }

    def solve_excited_states_iterative(
        self,
        ansatz_circuit: Optional[QuantumCircuit] = None,
        n_states: int = 3,
        max_iterations: int = 6,
        expansion_per_round: int = 50,
        energy_tol: float = 1e-4,
        seed_determinants: Optional[list] = None,
        spin_s: Optional[float] = None,
    ) -> dict:
        """State-averaged selected-CI for excited states.

        `solve_iterative` expands the subspace from the **ground-state**
        eigenvector only, so the subspace converges E₀ but stays blind to the
        excited manifold — low-lying states (notably the lowest triplet) are
        missed. This method expands from the top determinants of **all**
        ``n_states`` target states each round, so the subspace grows to span the
        whole low-lying manifold (state-averaged selected CI).

        Returns the same dict shape as `solve_excited_states` (energies ascending,
        excitation energies in Ha/eV, eigenvectors, n_determinants).
        """
        from scipy.sparse.linalg import eigsh as _sp_eigsh

        result = self._solve_raw(ansatz_circuit=ansatz_circuit)
        determinants = sorted(set(int(d) for d in result['determinants']))
        if seed_determinants:
            determinants = sorted(set(determinants) | {int(d) for d in seed_determinants})

        n_orb = self.hamiltonian.n_orbitals
        n_qubits = 2 * n_orb
        n_elec = self.hamiltonian.n_electrons
        h1, h2 = self._h1, self._h2
        nuc = float(self.hamiltonian.nuclear_repulsion)
        HA_TO_EV = 27.211386245988

        evals = evecs = None
        last_energies = None
        for it in range(max_iterations):
            n_det = len(determinants)
            k = min(n_states, n_det - 1)
            if k < 1:
                # Subspace too small to resolve excited states; diagonalize what we have.
                H_sparse, _ = _build_sparse_h_subspace(determinants, h1, h2, nuc, n_orb)
                evals, evecs = np.linalg.eigh(H_sparse.toarray())
                break
            H_sparse, _ = _build_sparse_h_subspace(determinants, h1, h2, nuc, n_orb)
            ncv = max(2 * k + 1, min(n_det - 1, 50))
            evals, evecs = _sp_eigsh(H_sparse, k=k, which='SA', tol=1e-8, ncv=ncv)
            order = np.argsort(evals)
            evals, evecs = evals[order], evecs[:, order]

            converged = (last_energies is not None and len(last_energies) == len(evals)
                         and max(abs(e - le) for e, le in zip(evals, last_energies)) < energy_tol)
            if converged or it == max_iterations - 1:
                break
            last_energies = [float(e) for e in evals]

            # State-averaged expansion: union of top dets across ALL states.
            new = set()
            for col in range(evecs.shape[1]):
                evec = evecs[:, col]
                top = np.argsort(np.abs(evec) ** 2)[::-1][:expansion_per_round]
                for i in top:
                    new.update(_generate_singles_doubles(determinants[i], n_qubits, n_elec))
            grown = sorted(set(determinants) | new)
            if len(grown) == len(determinants):
                break          # subspace saturated; evals already match determinants
            determinants = grown

        # Optional multiplicity filter: the bare subspace diagonalization returns the
        # lowest M_s=0 roots (mixed singlet/triplet). When spin_s is requested (the
        # singlet-default for closed-shell spectra, consistent with the CI route),
        # re-diagonalize for extra roots, compute ⟨S²⟩ per root (block-convention via
        # contract_ss, applying the interleaved→block sign), and keep the n_states
        # lowest whose multiplicity matches s(s+1). (reorg Phase D)
        if spin_s is not None and len(determinants) > 1:
            from kanad.core.ci.selected_ci import s_squared_of_subspace
            n_a = (n_elec + int(round(2 * self.target_sz))) // 2
            n_b = n_elec - n_a
            sgn = np.array([_interleave_to_block_sign(int(d), n_orb) for d in determinants])

            def _ss(evec_int):
                # interleaved-JW -> PySCF block convention, then ⟨S²⟩ via core. (reorg B-audit #17)
                return s_squared_of_subspace(determinants, evec_int * sgn, n_orb, n_a, n_b)

            n_det_f = len(determinants)
            want = min(3 * n_states + 3, n_det_f)   # enough roots to find n_states singlets
            Hs, _ = _build_sparse_h_subspace(determinants, h1, h2, nuc, n_orb)
            if n_det_f <= 400 or want >= n_det_f - 1:
                # Small subspace: dense eigh gives ALL roots, so every singlet is
                # available (eigsh capped at n_det-1 could miss a singlet behind a triplet).
                eb, vb = np.linalg.eigh(Hs.toarray())
            else:
                ncv = max(2 * want + 1, min(n_det_f - 1, 60))
                eb, vb = _sp_eigsh(Hs, k=want, which='SA', tol=1e-8, ncv=ncv)
            o = np.argsort(eb); eb, vb = eb[o], vb[:, o]
            ss_t = spin_s * (spin_s + 1.0)
            keep = [i for i in range(len(eb)) if abs(_ss(vb[:, i]) - ss_t) < 0.3][:n_states]
            if keep:
                evals, evecs = eb[keep], vb[:, keep]

        # Correct interleaved-JW → block convention on ALL state columns before they
        # are stored/returned (PySCF make_rdm1 assumes block ordering). The ground-state
        # path does this; the excited-states path skipped it, corrupting any subsequent
        # RDM/dipole. evecs corresponds 1:1 to `determinants` at loop exit. (CORE_BUGS B17.)
        signs = np.array([_interleave_to_block_sign(int(d), n_orb) for d in determinants])
        evecs = evecs * signs[:, None]

        e0 = float(evals[0])
        self.results = {
            'determinants': determinants,
            'eigenvector': evecs[:, 0],
            'energy': e0,
        }
        return {
            'energies': [float(e) for e in evals],
            'excitation_energies_ha': [float(e - e0) for e in evals],
            'excitation_energies_ev': [float((e - e0) * HA_TO_EV) for e in evals],
            'eigenvectors': evecs,
            'n_determinants': len(determinants),
            'n_states': len(evals),
        }

    def populate_hamiltonian_density(self) -> None:
        """Extract SQD 1-RDM and store on the hamiltonian.

        The hamiltonian's ``set_quantum_density_matrix`` handles
        active-MO → full-MO embedding (frozen orbitals get +2 on the
        diagonal) and the MO → AO transform. After this call,
        ``PropertyCalculator.compute_quantum_dipole_moment()`` and friends
        will use the SQD wavefunction density rather than a silent HF
        fallback.
        """
        rdm1 = self.get_1rdm_active_mo()
        if not hasattr(self.hamiltonian, 'set_quantum_density_matrix'):
            raise RuntimeError(
                f"Hamiltonian {type(self.hamiltonian).__name__} does not "
                "implement set_quantum_density_matrix(). Add it so "
                "wavefunction-derived observables (dipole, NMR, polarizability) "
                "work after SQD."
            )
        self.hamiltonian.set_quantum_density_matrix(rdm1)
        self.results['quantum_1rdm'] = rdm1
