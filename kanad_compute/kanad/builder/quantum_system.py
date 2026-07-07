"""`QuantumSystem` — a materialized molecular system every workflow consumes.

Produced by `MolecularBuilder.build()`. Wraps ``(spec, mf, ActiveHamiltonian)``
and exposes a uniform surface:

- ``solve()``        — ground-state energy via auto/explicit solver dispatch
- ``energy_fn()``    — the ``(atoms_bohr, warm_state) -> (E, warm_state)``
                       closure that the dynamics force engine and reaction
                       scans consume (foundation for the next-pass adapters)

Property / dynamics / reaction / excited-state *methods* are wired in the next
pass; the data and closure they need are already here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class QuantumSystem:
    def __init__(self, spec, mf, hamiltonian):
        self.spec = spec
        self.mf = mf
        self.hamiltonian = hamiltonian
        self.results: Optional[Dict[str, Any]] = None
        self._sqd_solver = None   # stashed by _solve_sqd for excited_states reuse

    @property
    def n_orbitals(self) -> int:
        return int(self.hamiltonian.n_orbitals)

    @property
    def n_electrons(self) -> int:
        return int(self.hamiltonian.n_electrons)

    @property
    def n_qubits(self) -> int:
        return 2 * int(self.hamiltonian.n_orbitals)

    # ----- solve --------------------------------------------------------

    def solve(self, warm_state=None, apply_conditions: bool = True) -> Dict[str, Any]:
        """Solve for the ground state, dispatching by ``spec.solver``.

        ``warm_state`` (the opaque payload a previous solve returned — VQE
        parameters or an SQD determinant subspace) warm-starts this solve when
        given. Used by `energy_fn` to thread the previous geometry's solution
        through a scan / MD trajectory.

        ``apply_conditions`` adds the environmental-condition corrections
        (solvation / thermal-RRHO / pH) to the reported energy. It is forced
        OFF inside ``energy_fn`` (forces/MD/scans) because those corrections are
        not part of the Born-Oppenheimer electronic PES — the thermal term in
        particular runs a full Hessian per geometry and turns the returned value
        into a free energy, corrupting the numerical force. (CORE_BUGS B13.)
        """
        route = self.spec.solver
        if route == 'auto':
            from kanad.solvers.solver_router import SolverRouter
            route = SolverRouter.select(self.n_qubits, self.spec.backend)

        if route == 'ci':
            res = self._solve_ci()
        elif route == 'vqe':
            res = self._solve_vqe(warm_state)
        elif route == 'sqd':
            res = self._solve_sqd(warm_state)
        else:
            raise ValueError(
                f"Unknown solver route {route!r}; expected 'auto', 'ci', 'vqe', or 'sqd'."
            )
        res['solver'] = route
        res['n_qubits'] = self.n_qubits
        if apply_conditions:
            res = self._apply_conditions(res)
        self.results = res
        return res

    # ----- conditions (M9: solvent / pH / T,P) --------------------------

    def _apply_conditions(self, res: Dict[str, Any]) -> Dict[str, Any]:
        """Add environmental-condition corrections to the solved energy.

        Composite: G(T,P,solvent,pH) = E_elec + ΔG_solv + ΔG_thermal + ΔG_pH.
        Each term is real physics (PCM, RRHO, Henderson-Hasselbalch).
        """
        cond = self.spec.conditions
        solvent = cond.get('solvent', 'vacuum')
        T = cond.get('temperature', 298.15)
        corrections = {}
        if solvent and solvent != 'vacuum':
            corrections['solvation_ha'] = self._pcm_solvation_correction(solvent)
        if cond.get('thermal'):
            corrections['thermal_ha'] = self._thermal_correction(T, cond.get('pressure', 1.0))
        if cond.get('pH') is not None:
            corrections['pH_ha'] = self._ph_correction(
                cond['pH'], T, solvent, cond.get('ph_sites'))
        if not corrections:
            return res
        gas = res['energy']
        res['gas_energy'] = gas
        res['energy'] = gas + sum(corrections.values())
        res['conditions'] = {
            'solvent': solvent, 'pH': cond.get('pH'), 'temperature': T,
            'pressure': cond.get('pressure'), 'thermal': cond.get('thermal'),
            'corrections_ha': corrections,
        }
        return res

    def _spec_to_molecule(self):
        from kanad.core.atom import Atom
        from kanad.core.molecule import Molecule
        atoms = [Atom(sym, list(xyz)) for (sym, xyz) in self.spec.atoms]
        return Molecule(atoms, charge=self.spec.charge, spin=self.spec.spin,
                        basis=self.spec.basis)

    def _thermal_correction(self, temperature: float, pressure_atm: float) -> float:
        """RRHO free-energy correction ΔG_thermal = G(T,P) − E_elec.

        Real rigid-rotor-harmonic-oscillator: computes vibrational frequencies
        (Hessian), then ZPE + thermal enthalpy − T·S at (T, P) via
        `ThermochemistryCalculator`. Pressure enters through the translational
        entropy (Sackur-Tetrode). Imaginary modes (non-minimum geometry) are
        recorded in the result as a warning, not silently dropped.
        """
        from kanad.analysis.vibrational_analysis import FrequencyCalculator
        from kanad.analysis.thermochemistry import ThermochemistryCalculator
        mol = self._spec_to_molecule()
        fr = FrequencyCalculator(mol).compute_frequencies(method='HF', verbose=False)
        n_imag = int(fr.get('n_imaginary', 0))
        if n_imag > 0:
            self._thermal_warning = (
                f"{n_imag} imaginary frequencies — geometry is not a minimum; "
                "RRHO thermal correction is approximate.")
        freqs = [float(f) for f in fr['frequencies'] if f > 0]
        tc = ThermochemistryCalculator(mol, frequencies=freqs)
        out = tc.compute_thermochemistry(temperature=temperature,
                                         pressure=pressure_atm * 101325.0)
        return float(out['g'] - out['e_elec'])

    def _ph_correction(self, pH: float, temperature: float, solvent: str,
                       sites=None) -> float:
        """Henderson-Hasselbalch protonation free energy ΔG(pH) via `pHModulator`.

        ΔG(pH) is isolated as free_energy − E_base, so it is independent of which
        electronic energy the modulator used. ``sites`` lets the caller declare
        protonation sites explicitly — ``[{'atom_index', 'group_type', 'pKa'?}]``
        — so pH works on geometry-only molecules. Without sites (and no
        connectivity to auto-detect them), pH is a reported no-op.
        """
        from kanad.core.environment.ph_effects import pHModulator
        mol = self._spec_to_molecule()
        solv = solvent if solvent and solvent != 'vacuum' else 'water'
        mod = pHModulator()
        for s in (sites or []):
            mod.add_site(s['atom_index'], s['group_type'], custom_pKa=s.get('pKa'))
        out = mod.apply_pH(mol, pH=pH, temperature=temperature, solvent=solv)
        # The Henderson-Hasselbalch protonation free energy (ΔG° + RT ln10·(pH−pKa));
        # 0 when no protonation site is registered.
        return float(out.get('protonation_free_energy', 0.0))

    def _pcm_solvation_correction(self, solvent: str) -> float:
        """ΔG_solv = E_PCM(HF) − E_gas(HF) via real PCM (`pyscf.solvent.pcm`).

        A mean-field solvation correction added to the correlated gas-phase
        energy — uses the apparent-surface-charge PCM, replacing the prior
        dielectric/Born model (inspection D7).
        """
        from pyscf import scf
        from kanad.core.environment.solvent import SOLVENT_DATABASE
        entry = SOLVENT_DATABASE.get(solvent.lower())
        if entry is None:
            raise ValueError(
                f"Unknown solvent {solvent!r}; known: {sorted(SOLVENT_DATABASE)}")
        mol = self.mf.mol
        mf_pcm = (scf.ROHF(mol) if self.spec.spin != 0 else scf.RHF(mol)).PCM()
        mf_pcm.with_solvent.eps = entry['epsilon']
        mf_pcm.verbose = 0
        e_pcm = float(mf_pcm.run().e_tot)
        return e_pcm - float(self.mf.e_tot)

    def _resolve_ansatz_type(self, route: str) -> str:
        at = self.spec.ansatz_type
        if at != 'auto':
            return at
        # SQD wants a shallow sampling ansatz; VQE wants a chemistry ansatz.
        return 'lucj' if route == 'sqd' else 'givens_sd'

    def _solve_ci(self) -> Dict[str, Any]:
        """Exact CASCI on the active space via PySCF FCI, spin-targeted to spec.spin.

        PySCF's bare Davidson can converge to the wrong multiplicity when states
        are near-degenerate (e.g. C2's singlet/triplet) — the root cause of the
        apparent "SQD < CASCI" anomaly. `fix_spin_` pins S(S+1) so the CI route
        returns the requested spin state, which matters for reaction profiles.
        """
        from pyscf import fci

        # Optional multireference-PT2 upgrade (the real accuracy fix vs CASPT2/exp):
        #   orbital_optimization=True -> CASSCF (relaxes the canonical HF orbitals;
        #       halves vertical-excitation errors empirically)
        #   pt2_correction='nevpt2'  -> adds out-of-active-space dynamical correlation
        #       (SC-NEVPT2 on the CAS reference), which bare CASCI structurally misses.
        kw = self.spec.solver_kwargs
        pt2 = kw.get('pt2_correction')               # None | 'nevpt2'
        orb_opt = kw.get('orbital_optimization', pt2 is not None)
        if pt2 or orb_opt:
            return self._solve_ci_mrpt(pt2, orb_opt)

        h1 = np.asarray(self.hamiltonian.h_core, dtype=float)
        eri = np.asarray(self.hamiltonian.eri, dtype=float)
        norb = self.n_orbitals
        nelec = self.n_electrons
        na = (nelec + self.spec.spin) // 2
        nb = nelec - na
        s = self.spec.spin / 2.0
        cisolver = fci.direct_spin1.FCI()
        cisolver.conv_tol = 1e-11
        fci.addons.fix_spin_(cisolver, ss=s * (s + 1))
        e, civec = cisolver.kernel(
            h1, eri, norb, (na, nb),
            ecore=float(self.hamiltonian.nuclear_repulsion),
        )
        return {'energy': float(e), 'method': 'ci', 'fci_vector': civec}

    def _solve_ci_mrpt(self, pt2, orb_opt) -> Dict[str, Any]:
        """CASSCF (orbital-optimized) and/or NEVPT2-corrected CAS energy on the
        SAME active orbitals, via PySCF mcscf + mrpt. This attacks the true
        accuracy ceiling of bare CASCI (no orbital relaxation, no PT2)."""
        from pyscf import mcscf, mrpt
        ncas = self.n_orbitals
        nelecas = self.n_electrons
        s = self.spec.spin / 2.0
        active_space = self.hamiltonian.active_space
        active = list(active_space.active_indices)
        mc = (mcscf.CASSCF(self.mf, ncas, nelecas) if orb_opt
              else mcscf.CASCI(self.mf, ncas, nelecas))
        if self.spec.spin == 0:
            try:
                mc.fix_spin_(ss=0)
            except Exception:
                pass
        # Pin the SAME active orbitals Kanad selected (1-indexed for sort_mo).
        # AUDIT H3: active_indices index active_space.mo_coeff, which for rotated
        # strategies (mp2no, avas) is NOT the canonical mf.mo_coeff that sort_mo
        # defaults to. Passing mo_coeff= keeps CASSCF/CASCI on the SAME basis
        # Kanad selected (a no-op for canonical frozen_core/frontier/manual,
        # whose active_space.mo_coeff IS the canonical C); without it, CASCI on
        # mp2no diagonalized a different active space (N2/sto-3g: ~14 mHa wrong).
        mo = mc.sort_mo([i + 1 for i in active],
                        mo_coeff=np.asarray(active_space.mo_coeff))
        mc.run(mo, verbose=0)
        e = float(mc.e_tot)
        cas_e = float(mc.e_tot)
        method = 'casscf' if orb_opt else 'casci'
        # Snapshot the CI vector (and, for CASSCF, the optimized orbitals) BEFORE
        # any NEVPT2 call below — mrpt.NEVPT(mc).kernel() canonicalizes mc in place,
        # which would otherwise leave observables() reading an inconsistent
        # CI-vector/orbital pair. Surfacing mc.ci for CASSCF too is what makes
        # observables() work after orbital_optimization=True (previously
        # fci_vector=None → fci.make_rdm1(None, …) asserted).
        ci_snapshot = None if mc.ci is None else np.asarray(mc.ci).copy()
        out: Dict[str, Any] = {'fci_vector': ci_snapshot, 'cas_energy': cas_e}
        if orb_opt:
            # CASSCF rotates the active orbitals; mc.ci lives in mc.mo_coeff's
            # [core | active | virtual] layout. Surface the optimized MOs so
            # observables() embeds the active 1-RDM in the CONSISTENT basis rather
            # than the stale, pre-optimization active_space.mo_coeff.
            out['active_mo_coeff'] = np.asarray(mc.mo_coeff).copy()
        if pt2 == 'nevpt2':
            # DIRADICAL / STRONG-OPEN-SHELL GUARD. Single-state NEVPT2 is unreliable
            # when the active-space reference has strong open-shell/diradical character
            # (several orbitals with fractional occupation, e.g. two degenerate e_g
            # orbitals each ~singly occupied) — single-state PT over-stabilizes the
            # diradical. Measured failure: the D4h square cyclobutadiene TS barrier goes
            # +8.86 (bare CASCI, ≈ TBE 8.9) → −5.39 kcal/mol with NEVPT2. Diagnose via the
            # natural-orbital "diradical index" Σ min(n_i, 2−n_i) (≈ number of effectively
            # unpaired electrons), WARN, and surface the robust bare-CASCI energy.
            # (RETEST_RESULTS.) State-gap is the WRONG diagnostic here (the D4h
            # singlet-singlet gap is 2.25 eV — the degeneracy is in the ORBITALS).
            diradical = None
            try:
                dm1 = mc.fcisolver.make_rdm1(mc.ci, ncas, nelecas)
                noons = np.clip(np.linalg.eigvalsh(np.asarray(dm1, dtype=float)), 0.0, 2.0)
                diradical = float(np.sum(np.minimum(noons, 2.0 - noons)))
            except Exception:
                diradical = None
            near_deg = diradical is not None and diradical > 1.0
            out['diradical_index'] = diradical
            out['near_degenerate'] = bool(near_deg)
            e = e + float(mrpt.NEVPT(mc).kernel())
            method = method + '+nevpt2'
            if near_deg:
                msg = (f"NEVPT2 on a strongly OPEN-SHELL/diradical reference (diradical "
                       f"index {diradical:.2f} > 1.0): single-state PT may be unbalanced "
                       f"(over-stabilizes the diradical). The bare-CASCI energy "
                       f"({cas_e:.6f} Ha) is the robust fallback; consider state-averaged / "
                       f"quasi-degenerate PT. (cas_energy in result.)")
                logger.warning(msg)
                out['nevpt2_warning'] = msg
        out['energy'] = float(e)
        out['method'] = method
        return out

    def _solve_vqe(self, warm_state=None) -> Dict[str, Any]:
        from kanad.solvers import VQESolver

        passthrough = ('optimizer', 'max_iterations', 'conv_threshold', 'ansatz_n_layers')
        kw = {k: v for k, v in self.spec.solver_kwargs.items() if k in passthrough}
        solver = VQESolver(
            hamiltonian=self.hamiltonian,
            ansatz_type=self._resolve_ansatz_type('vqe'),
            mapper_type=self.spec.mapper,
            backend=self.spec.backend,
            **kw,
        )
        # Warm-start from the previous geometry's optimal parameters when their
        # length matches this ansatz (true under the default freeze policy, where
        # the active space — hence the parameter count — is constant along a scan).
        solve_kwargs = {}
        if warm_state is not None and np.ndim(warm_state) == 1:
            solve_kwargs['initial_parameters'] = np.asarray(warm_state, dtype=float)
        # VQESolver now returns a SolverResult (unified solver protocol); flatten
        # to the legacy dict shape the builder consumes.
        res = solver.solve(**solve_kwargs).to_dict()
        return {
            'energy': float(res['energy']),
            'parameters': res.get('parameters'),
            'method': 'vqe',
            'raw': res,
        }

    def _make_sqd_solver(self):
        """Construct the SamplingSQDSolver + bound sampling circuit from the spec."""
        from kanad.solvers.sampling_sqd import SamplingSQDSolver
        kw = self.spec.solver_kwargs
        ctor_keys = ('bq_device', 'ibm_backend_name', 'ibm_timeout_s', 'job_name',
                     'recovery_rounds', 'recovery_tol', 'spin_s', 'cisd_seed')
        solver = SamplingSQDSolver(
            self.hamiltonian,
            n_samples=kw.get('n_samples', 10000),
            backend=self.spec.backend,
            target_sz=self.spec.spin / 2.0,
            random_seed=kw.get('random_seed', 0),
            recover_configurations=kw.get('recover_configurations', True),
            ci_backend=kw.get('ci_backend', 'pyscf'),
            **{k: kw[k] for k in ctor_keys if k in kw},
        )
        circuit = self._build_sampling_circuit(self._resolve_ansatz_type('sqd'))
        return solver, circuit

    def _solve_sqd(self, warm_state=None) -> Dict[str, Any]:
        solver, circuit = self._make_sqd_solver()
        kw = self.spec.solver_kwargs
        # quantum_only=True: report the PURE-quantum SQD energy — single sample→
        # filter/recover→diagonalize, with NO classical singles+doubles expansion.
        # The default (solve_iterative) always layers a classical CISD-style expansion
        # on the sample, which on saturable spaces masks the quantum-sample quality and
        # makes "quantum-driven accuracy" claims indefensible. Use this mode to measure
        # what the QPU sample alone delivers vs an exact reference.
        if kw.get('quantum_only', False):
            # SamplingSQDSolver now returns a SolverResult (unified solver
            # protocol); flatten to the legacy dict the builder mutates below.
            res = solver.solve(ansatz_circuit=circuit).to_dict()
            res['method'] = 'sqd-quantum-only'
            self._sqd_solver = solver
            return res
        iter_keys = ('max_iterations', 'expansion_per_round', 'energy_tol')
        # Warm-start: seed with the previous geometry's determinant subspace.
        res = solver.solve_iterative(
            ansatz_circuit=circuit,
            seed_determinants=warm_state if isinstance(warm_state, (list, tuple, np.ndarray)) else None,
            **{k: kw[k] for k in iter_keys if k in kw},
        ).to_dict()
        res['method'] = 'sqd'
        self._sqd_solver = solver   # reused by excited_states()
        return res

    # ----- excited states (M8) ------------------------------------------

    def excited_states(self, n_states: int = 3, spin: Optional[float] = None,
                       pt2_correction: Optional[str] = None,
                       orbital_optimization: bool = False,
                       select: Optional[str] = None,
                       bright_threshold: float = 0.01,
                       rydberg_r2: float = 15.0) -> Dict[str, Any]:
        """Low-lying vertical spectrum via the builder.

        NOTE (I3 — large conjugated systems): a small frontier active space
        quantitatively MISORDERS states for extended π systems (e.g. indigo,
        acenes, carotenoids — the dark 2Ag vs bright 1Bu ordering). For those use
        the FULL π active space, e.g. ``.active_space('avas', ao_labels=['C 2pz',
        'N 2pz'])``, not a small ``frontier`` window. State energies/character on a
        too-small CAS are unreliable even though the call succeeds.

        CI / VQE routes diagonalize the active-space CASCI for the lowest
        ``n_states`` roots; the SQD route re-diagonalizes the converged
        selected-CI subspace (`SamplingSQDSolver.solve_excited_states`).

        Args:
            n_states: number of states (ground + excited), ascending in energy.
            spin: optional S to target one manifold via `fix_spin_` (0 → singlet
                ladder S0/S1/S2…, 1 → triplets). Default None = the true lowest
                spectrum across manifolds (so T1 can appear below S1).
            pt2_correction: 'nevpt2' adds per-root SC-NEVPT2 dynamical correlation
                on top of the multireference states (the CASPT2/MRCI-grade layer
                bare CASCI lacks). Routes through the SA-CASSCF/CASCI path.
            orbital_optimization: True runs STATE-AVERAGED CASSCF over the
                ``n_states`` roots (balanced orbitals for the whole manifold)
                instead of canonical HF orbitals — the correct orbital basis for
                excitation energies and transition intensities.
            select: pick states by CHARACTER instead of returning all by energy
                (essential in diffuse bases where dark Rydberg states intrude as
                low roots). None = all; 'bright' = ground + states with
                f≥``bright_threshold``; 'brightest' = ground + the single most
                intense excited state; 'valence' = ground + non-Rydberg states
                (Δ⟨r²⟩ < ``rydberg_r2``). Ground state is always kept at index 0.
            bright_threshold: oscillator-strength cutoff for 'bright'/character tag.
            rydberg_r2: Δ⟨r²⟩ (bohr²) above which a state is tagged 'rydberg'.

        Returns:
            dict with ``energies`` (Ha, ascending), ``excitation_energies_ha`` /
            ``excitation_energies_ev`` (relative to the lowest), ``n_states``,
            ``method``, ``solver``.
        """
        route = self.spec.solver
        if route == 'auto':
            from kanad.solvers.solver_router import SolverRouter
            route = SolverRouter.select(self.n_qubits, self.spec.backend)
        if pt2_correction is not None or orbital_optimization:
            # Multireference excited-state ladder: SA-CASSCF (balanced orbitals
            # across the whole manifold) + per-root NEVPT2 (dynamical correlation).
            res = self._excited_states_sa_casscf(
                n_states, spin, pt2_correction, orbital_optimization)
        else:
            res = (self._excited_states_sqd(n_states) if route == 'sqd'
                   else self._excited_states_ci(n_states, spin))
        res['solver'] = route
        res = self._annotate_and_select(res, select, bright_threshold, rydberg_r2)
        return res

    def _excited_states_ci(self, n_states, spin=None) -> Dict[str, Any]:
        from pyscf import fci
        HA_TO_EV = 27.211386245988
        h1 = np.asarray(self.hamiltonian.h_core, dtype=float)
        eri = np.asarray(self.hamiltonian.eri, dtype=float)
        norb = self.n_orbitals
        nelec = self.n_electrons
        na = (nelec + self.spec.spin) // 2
        nb = nelec - na
        # Default to the GROUND-STATE multiplicity so the excited roots are
        # spin-pure. Mixed-spin direct_spin1 returns spin-forbidden triplets among
        # the low roots (transition dipole ≈ 0), burying the bright singlet
        # transitions → oscillator strengths come out ~0 (e.g. ethylene π→π* showed
        # f≈1e-29; with the singlet manifold it is f≈0.66). For a closed-shell
        # (n_α = n_β) ground state with no explicit override, target S=0.
        #
        # The manifold is selected by ⟨S²⟩ FILTERING (core.ci.fci_excited_states),
        # NOT fci.addons.fix_spin_. The fix_spin_ level shift (λ=0.2) is too weak
        # when an off-multiplicity state sits within λ·S² of the next target state:
        # on H₂/STO-3G the lowest triplet (−0.5308) shifts to −0.3308 and out-ranks
        # the true second singlet (1σ_u², +0.4831), so fix_spin_ returns a
        # penalty-contaminated triplet as "S1" and disagrees with the SQD route.
        # ⟨S²⟩ filtering is shift-free and matches the SQD lane exactly. (Phase D)
        from kanad.core.ci import fci_excited_states
        if spin is None and na == nb:
            spin = 0.0
        target_sz = (na - nb) / 2.0
        energies, civecs = fci_excited_states(
            h1, eri, float(self.hamiltonian.nuclear_repulsion),
            norb, nelec, n_states, spin_s=spin, target_sz=target_sz)
        energies = [float(x) for x in energies]
        e0 = energies[0]
        osc, tdips = self._oscillator_strengths(civecs, energies, norb, (na, nb))
        civ_list = (list(civecs) if isinstance(civecs, (list, tuple))
                    else list(np.atleast_1d(civecs)))
        active = list(self.hamiltonian.active_space.active_indices)
        C_act = np.asarray(self.hamiltonian.active_space.mo_coeff)[:, active]
        ext = self._state_extents(civ_list, norb, (na, nb), C_act)
        return {
            'energies': energies,
            'excitation_energies_ha': [x - e0 for x in energies],
            'excitation_energies_ev': [(x - e0) * HA_TO_EV for x in energies],
            'oscillator_strengths': osc,
            'transition_dipoles': tdips,
            'state_extent_r2': ext,
            'n_states': len(energies),
            'method': 'casci-nroots',
        }

    def _oscillator_strengths(self, civecs, energies, norb, nelec):
        """f_0i = (2/3)·(E_i−E_0)·|⟨0|μ|i⟩|² from CASCI transition 1-RDMs.

        The transition dipole uses only the active-orbital dipole integrals —
        the frozen core is identical in both states so it cancels. Singlet→
        triplet transitions come out ~0 (spin-forbidden) because `trans_rdm1`
        between different multiplicities vanishes — the physics falls out for free.
        """
        from pyscf import fci

        civecs = list(np.atleast_1d(civecs)) if not isinstance(civecs, (list, tuple)) else list(civecs)
        n = len(energies)
        if n < 2:
            return [0.0], [[0.0, 0.0, 0.0]]

        # Active-MO electric-dipole integrals: ⟨p|r|q⟩ over active orbitals.
        mol = self.hamiltonian.mol
        active = list(self.hamiltonian.active_space.active_indices)
        C_act = np.asarray(self.hamiltonian.active_space.mo_coeff)[:, active]
        from kanad.core.integrals.property_integrals import compute_dipole
        from kanad.core.integrals.transforms import property_integral_transform
        dip_ao = compute_dipole(mol)                          # (3, n_ao, n_ao) ⟨p|r|q⟩
        dip_mo = property_integral_transform(dip_ao, C_act)   # (3, n_act, n_act)

        osc = [0.0]
        tdips = [[0.0, 0.0, 0.0]]
        c0 = civecs[0]
        for i in range(1, n):
            tdm = fci.direct_spin1.trans_rdm1(c0, civecs[i], norb, nelec)  # ⟨0|a†_p a_q|i⟩
            d = -np.einsum('xpq,pq->x', dip_mo, tdm)     # electronic transition dipole (e = −1)
            f = (2.0 / 3.0) * (energies[i] - energies[0]) * float(np.dot(d, d))
            osc.append(float(f))
            tdips.append([float(x) for x in d])
        return osc, tdips

    def _osc_from_orbs(self, civecs, energies, norb, nelec, C_act):
        """f_0i = (2/3)·(E_i−E_0)·|⟨0|μ|i⟩|² with an EXPLICIT active-orbital
        coefficient matrix ``C_act`` (n_ao, n_act).

        Identical physics to ``_oscillator_strengths`` but takes the orbitals as
        an argument so the SA-CASSCF *optimized* active orbitals are used for the
        transition dipoles, not the canonical HF orbitals — the two differ once
        orbital relaxation moves the π/π* (or n/π*) densities.
        """
        from pyscf import fci

        civecs = list(civecs) if isinstance(civecs, (list, tuple)) else [civecs]
        n = len(energies)
        if n < 2 or len(civecs) < 2:
            return [0.0] * n, [[0.0, 0.0, 0.0] for _ in range(n)]
        mol = self.hamiltonian.mol
        from kanad.core.integrals.property_integrals import compute_dipole
        from kanad.core.integrals.transforms import property_integral_transform
        dip_ao = compute_dipole(mol)                          # (3, n_ao, n_ao)
        dip_mo = property_integral_transform(dip_ao, C_act)
        osc = [0.0]
        tdips = [[0.0, 0.0, 0.0]]
        c0 = civecs[0]
        for i in range(1, min(n, len(civecs))):
            tdm = fci.direct_spin1.trans_rdm1(c0, civecs[i], norb, nelec)
            d = -np.einsum('xpq,pq->x', dip_mo, tdm)
            f = (2.0 / 3.0) * (energies[i] - energies[0]) * float(np.dot(d, d))
            osc.append(float(f))
            tdips.append([float(x) for x in d])
        return osc, tdips

    def _state_extents(self, civecs, norb, nelec, C_act):
        """Δ⟨r²⟩ (bohr²) of each state's electron density vs the ground state, in
        the active-orbital basis — a Rydberg diagnostic. Diffuse Rydberg states
        (e.g. π→3s) have a large positive Δ⟨r²⟩; compact valence states ≈0. Returns
        a list aligned with ``civecs`` (ground = 0.0), or None if unavailable.
        """
        from pyscf import fci
        civecs = list(civecs) if isinstance(civecs, (list, tuple)) else [civecs]
        if not civecs:
            return None
        try:
            mol = self.hamiltonian.mol
            from kanad.core.integrals.property_integrals import compute_r2
            from kanad.core.integrals.transforms import property_integral_transform
            r2_ao = compute_r2(mol)                       # ⟨p|r²|q⟩ (n_ao, n_ao)
            r2_mo = property_integral_transform(r2_ao, np.asarray(C_act))
            ext, r2_0 = [], None
            for i, c in enumerate(civecs):
                dm = fci.direct_spin1.make_rdm1(c, norb, nelec)
                r2_i = float(np.einsum('pq,pq->', r2_mo, dm))
                if i == 0:
                    r2_0 = r2_i
                    ext.append(0.0)
                else:
                    ext.append(float(r2_i - r2_0))
            return ext
        except Exception:
            return None

    def _annotate_and_select(self, res, select, bright_threshold, rydberg_r2):
        """Tag each state with character (ground / bright|dark / valence|rydberg)
        and optionally filter the returned manifold by that character."""
        energies = res.get('energies') or []
        n = res.get('n_states', len(energies))
        osc = res.get('oscillator_strengths') or []
        ext = res.get('state_extent_r2')          # may be None (e.g. SQD route)

        def _f(i):
            return float(osc[i]) if i < len(osc) else 0.0

        def _is_ryd(i):
            return (ext is not None and i < len(ext) and ext[i] is not None
                    and ext[i] >= rydberg_r2)

        char = []
        for i in range(n):
            if i == 0:
                char.append('ground')
                continue
            tag = 'bright' if _f(i) >= bright_threshold else 'dark'
            if ext is not None and i < len(ext) and ext[i] is not None:
                tag += '/rydberg' if _is_ryd(i) else '/valence'
            char.append(tag)
        res['character'] = char

        if select in (None, 'all'):
            return res
        keep = [0]
        if select == 'bright':
            keep += [i for i in range(1, n) if _f(i) >= bright_threshold]
        elif select == 'brightest':
            if n > 1:
                keep.append(max(range(1, n), key=_f))
        elif select == 'valence':
            keep += [i for i in range(1, n) if not _is_ryd(i)]
        else:
            raise ValueError(f"unknown select={select!r} (use 'bright'|'brightest'|'valence'|None)")
        return self._subselect_states(res, keep)

    def _subselect_states(self, res, keep):
        """Reindex all per-state arrays to ``keep`` (ground stays at new index 0,
        so excitation energies — already relative to the ground state — stay valid)."""
        out = dict(res)
        for key in ('energies', 'excitation_energies_ha', 'excitation_energies_ev',
                    'oscillator_strengths', 'transition_dipoles', 'character',
                    'state_extent_r2'):
            v = res.get(key)
            if isinstance(v, list):
                out[key] = [v[i] for i in keep if i < len(v)]
        out['n_states'] = len(keep)
        out['selected_indices'] = list(keep)
        return out

    def _excited_states_sa_casscf(self, n_states, spin, pt2, orb_opt) -> Dict[str, Any]:
        """State-averaged CASSCF (+ per-root NEVPT2) excited-state ladder.

        ``orbital_optimization=True`` runs SA-CASSCF: the active orbitals are
        optimized for an EQUAL-WEIGHT average of the lowest ``n_states`` roots, so
        no single state is favoured — the standard recipe for balanced excitation
        energies. ``pt2_correction='nevpt2'`` then adds strongly-contracted NEVPT2
        independently to each root on the converged (SA-)reference. Together this
        is the CASPT2/MRCI-grade multireference excited-state treatment that bare
        CASCI-on-canonical-orbitals (``_excited_states_ci``) cannot reach.

        Oscillator strengths are recomputed in the SA-CASSCF active-orbital basis.
        """
        from pyscf import mcscf, mrpt, fci
        HA_TO_EV = 27.211386245988
        ncas = self.n_orbitals
        nelecas = self.n_electrons
        active_space = self.hamiltonian.active_space
        active = list(active_space.active_indices)
        # AUDIT H4: active_indices index active_space.mo_coeff, which for rotated
        # strategies (mp2no, avas) is NOT the canonical mf.mo_coeff that sort_mo
        # defaults to. Pass mo_coeff= to sort_mo so SA-CASSCF/CASCI-nroots build
        # on the SAME basis Kanad selected (a no-op for canonical
        # frozen_core/frontier/manual). Without it, N2/sto-3g mp2no(6) returned
        # ground E=-107.6263 vs the bare-CI -107.6361 on the mp2no Hamiltonian.
        start_mo = np.asarray(active_space.mo_coeff)
        na = (nelecas + self.spec.spin) // 2
        nb = nelecas - na
        # Default to the ground-state multiplicity so the manifold is spin-pure
        # (closed-shell n_α=n_β → singlet ladder). None = mixed-spin (open shell).
        s_target = spin if spin is not None else (0.0 if na == nb else None)

        def _new_fcisolver():
            return fci.direct_spin0.FCI() if s_target == 0.0 else fci.direct_spin1.FCI()

        if orb_opt:
            # SA-CASSCF: optimize orbitals for an equal-weight average of the
            # lowest n_states roots → one balanced orbital set for the manifold.
            base = mcscf.CASSCF(self.mf, ncas, nelecas)
            base.fcisolver = _new_fcisolver()
            base.fcisolver.nroots = n_states
            sa = mcscf.addons.state_average_(base, [1.0 / n_states] * n_states)
            mo = sa.sort_mo([i + 1 for i in active], mo_coeff=start_mo)  # 1-based
            sa.run(mo, verbose=0)
            opt_mo = np.asarray(sa.mo_coeff)
            method = 'sa-casscf'
            # The state-average FCI wrapper CANNOT be used for NEVPT2 (pyscf raises).
            # The states (energies, CI vectors) AND the per-root NEVPT2 reference must
            # come from a SEPARATE multi-root CASCI built IN the SA-optimized orbitals.
            states_mc = mcscf.CASCI(self.mf, ncas, nelecas)
            states_mc.fcisolver = _new_fcisolver()
            states_mc.fcisolver.nroots = n_states
            states_mc.kernel(opt_mo)
        else:
            states_mc = mcscf.CASCI(self.mf, ncas, nelecas)
            states_mc.fcisolver = _new_fcisolver()
            states_mc.fcisolver.nroots = n_states
            mo = states_mc.sort_mo([i + 1 for i in active], mo_coeff=start_mo)
            states_mc.kernel(mo)
            opt_mo = np.asarray(states_mc.mo_coeff)
            method = 'casci-nroots'

        e_cas = [float(x) for x in np.atleast_1d(states_mc.e_tot)]

        # SNAPSHOT the CI vectors + active orbitals NOW, as a consistent pair,
        # BEFORE NEVPT2 runs: mrpt.NEVPT(...).kernel() canonicalizes states_mc in
        # place (rotates .mo_coeff and re-expresses .ci), which would otherwise
        # desync the transition densities from the orbital basis and produce
        # nonsensical oscillator strengths (f≫1).
        ncore = states_mc.ncore
        ci_snap = [np.array(c, copy=True) for c in
                   (states_mc.ci if isinstance(states_mc.ci, (list, tuple)) else [states_mc.ci])]
        C_act = np.array(opt_mo[:, ncore:ncore + ncas], copy=True)

        # Per-root strongly-contracted NEVPT2 on the multi-root CASCI reference
        # (works for both routes: SA orbitals for sa-casscf, canonical for casci).
        pt2_applied = False
        pt2_warning = None
        if pt2 == 'nevpt2':
            e_states, failed = [], []
            for i in range(len(e_cas)):
                try:
                    corr = float(mrpt.NEVPT(states_mc, root=i).kernel())
                    e_states.append(float(e_cas[i] + corr))
                except Exception as ex:        # surface, never silently drop PT2
                    e_states.append(float(e_cas[i]))
                    failed.append((i, f"{type(ex).__name__}: {str(ex)[:60]}"))
            pt2_applied = len(failed) < len(e_cas)
            if failed:
                pt2_warning = f"NEVPT2 failed for roots {[f[0] for f in failed]}: {failed[0][1]}"
            method += '+nevpt2' if pt2_applied else '+nevpt2(FAILED)'
        else:
            e_states = [float(x) for x in e_cas]

        # Oscillator strengths from the (SA-)CASSCF transition densities (the
        # pre-NEVPT2 snapshot), in the optimized active-orbital basis. NEVPT2
        # corrects energies; intensities use the zeroth-order multireference
        # states scaled by the NEVPT2 excitation energy — the standard treatment.
        osc, tdips = self._osc_from_orbs(ci_snap, e_states, ncas, (na, nb), C_act)
        ext = self._state_extents(ci_snap, ncas, (na, nb), C_act)

        e0 = e_states[0]
        out = {
            'energies': [float(x) for x in e_states],
            'excitation_energies_ha': [float(x - e0) for x in e_states],
            'excitation_energies_ev': [float((x - e0) * HA_TO_EV) for x in e_states],
            'oscillator_strengths': osc,
            'transition_dipoles': tdips,
            'state_extent_r2': ext,
            'n_states': len(e_states),
            'method': method,
            'pt2_applied': pt2_applied if pt2 == 'nevpt2' else None,
        }
        if pt2_warning:
            out['pt2_warning'] = pt2_warning
        return out

    def _excited_states_sqd(self, n_states) -> Dict[str, Any]:
        # STATE-AVERAGED subspace that spans the excited manifold (a ground-state
        # converged subspace misses low-lying states such as the lowest triplet).
        # Returns excited-state ENERGIES; oscillator strengths on the SQD route
        # (subspace transition densities) are a follow-up — use the CI route for
        # exact intensities up to ~30 qubits.
        solver, circuit = self._make_sqd_solver()
        kw = self.spec.solver_kwargs
        ekeys = ('max_iterations', 'expansion_per_round', 'energy_tol')
        # Singlet-default for closed-shell, consistent with the CI route, so the SQD
        # and CI excited spectra agree (the bare subspace returns the mixed M_s=0
        # ladder incl. triplets). (reorg Phase D)
        spin_s = 0.0 if self.spec.spin == 0 else None
        ex = solver.solve_excited_states_iterative(
            ansatz_circuit=circuit, n_states=n_states, spin_s=spin_s,
            **{k: kw[k] for k in ekeys if k in kw},
        )
        self._sqd_solver = solver
        ex['method'] = 'sqd-state-averaged'
        return ex

    def absorption_spectrum(self, n_states: int = 5, broadening_ev: float = 0.3,
                            lineshape: str = 'gaussian', grid_points: int = 600,
                            energy_range_ev=None) -> Dict[str, Any]:
        """UV-Vis absorption spectrum from the excited-state oscillator strengths.

        Computes the vertical spectrum (`excited_states`), then both the stick
        spectrum (excitation energies + f) and an area-conserving broadened curve
        ``A(E) = Σ_i f_i · L(E − E_i)`` (so ∫A dE = Σ f_i, the TRK-style sum).

        Args:
            n_states: states to include (ground + excited).
            broadening_ev: lineshape width σ (Gaussian) or γ (Lorentzian), eV.
            lineshape: 'gaussian' | 'lorentzian'.
            grid_points: energy-grid resolution.
            energy_range_ev: optional (lo, hi) eV; default 0 → max + 5·width.

        Returns:
            dict: ``excitation_energies_ev``, ``oscillator_strengths`` (sticks),
            ``grid_ev``, ``intensity`` (broadened curve), plus lineshape metadata.
        """
        ex = self.excited_states(n_states=n_states)
        if 'oscillator_strengths' not in ex:
            raise NotImplementedError(
                "absorption_spectrum needs oscillator strengths, which the SQD "
                "route does not yet provide (subspace transition densities are a "
                "follow-up). Use the CI/VQE route, or excited_states() for bare "
                "energies."
            )
        exc = list(ex['excitation_energies_ev'][1:])   # drop the ground state (0 eV)
        osc = list(ex['oscillator_strengths'][1:])

        if energy_range_ev is None:
            hi = (max(exc) if exc else 1.0) + 5.0 * broadening_ev
            lo = 0.0
        else:
            lo, hi = energy_range_ev
        grid = np.linspace(lo, hi, grid_points)
        intensity = np.zeros_like(grid)
        for e, f in zip(exc, osc):
            if lineshape == 'gaussian':
                norm = 1.0 / (broadening_ev * np.sqrt(2.0 * np.pi))
                intensity += f * norm * np.exp(-0.5 * ((grid - e) / broadening_ev) ** 2)
            elif lineshape == 'lorentzian':
                norm = 1.0 / (np.pi * broadening_ev)
                intensity += f * norm * broadening_ev ** 2 / ((grid - e) ** 2 + broadening_ev ** 2)
            else:
                raise ValueError(f"lineshape must be 'gaussian' or 'lorentzian'; got {lineshape!r}")
        return {
            'excitation_energies_ev': exc,
            'oscillator_strengths': osc,
            'grid_ev': grid.tolist(),
            'intensity': intensity.tolist(),
            'lineshape': lineshape,
            'broadening_ev': broadening_ev,
        }

    def _build_sampling_circuit(self, ansatz_type: str):
        """Build + parameter-bind a sampling circuit for SQD.

        Random parameters spread amplitude across the dominant determinants —
        the proven sampling-SQD pattern; SQD's selected-CI does the variational
        work, so the ansatz need not be VQE-optimal.
        """
        n_qubits = self.n_qubits
        n_elec = self.n_electrons
        kw = self.spec.ansatz_kwargs
        n_layers = kw.get('n_layers', 1)
        target_sz = kw.get('target_sz', self.spec.spin / 2.0)
        at = ansatz_type.lower()

        if at == 'lucj':
            from kanad.core.ansatze import LUCJAnsatz
            ansatz = LUCJAnsatz(
                n_qubits=n_qubits, n_electrons=n_elec, n_layers=n_layers,
                mapper=self.spec.mapper, target_sz=target_sz,
            )
        elif at in ('hardware_efficient', 'hea'):
            from kanad.core.ansatze import HardwareEfficientAnsatz
            ansatz = HardwareEfficientAnsatz(
                n_qubits=n_qubits, n_electrons=n_elec, n_layers=n_layers,
                mapper=self.spec.mapper,
            )
        else:
            raise ValueError(
                f"SQD sampling circuit supports 'lucj' or 'hardware_efficient'; "
                f"got {ansatz_type!r}."
            )

        qc = ansatz.build_circuit()
        if qc.num_parameters > 0:
            # The legacy random-U(-0.3,0.3) binding collapses the LUCJ to ≈HF: the
            # sampled subspace is essentially the HF determinant alone, so the QUANTUM
            # sample contributes ~nothing and accuracy comes entirely from the classical
            # singles+doubles expansion (validated: random-0.3 captures ~2% of N₂ CAS(6,6)
            # correlation). For the LUCJ we instead use a PHYSICALLY-STRUCTURED init that
            # spreads amplitude onto the important configurations: the orbital rotations
            # are weighted toward the HOMO–LUMO boundary (where the dominant excitations
            # live) with a moderate Jastrow. The PURE quantum sample then captures the
            # majority of correlation (~69% on N₂ CAS(6,6) vs ~2% random). Override with
            # solver_kwargs['sampling_init']='random' to restore the legacy behavior.
            init = self.spec.solver_kwargs.get('sampling_init', 'physical')
            if ansatz_type == 'lucj' and init == 'physical':
                n_occ = getattr(ansatz, 'n_occ_spatial', n_elec // 2)
                vals = []
                for prm in qc.parameters:
                    nm = str(prm)
                    if nm.startswith('θ_orb'):
                        p_sp = int(nm.split('_')[-1])
                        vals.append(1.2 if p_sp == n_occ - 1 else 0.4)  # larger at HOMO–LUMO
                    else:                                                # K_jas
                        vals.append(0.8)
                qc = qc.assign_parameters({qc.parameters[i]: float(vals[i]) for i in range(len(vals))})
            else:
                rng = np.random.default_rng(self.spec.solver_kwargs.get('random_seed', 0))
                params = rng.uniform(-0.3, 0.3, size=qc.num_parameters)
                qc = qc.assign_parameters(
                    {qc.parameters[i]: float(params[i]) for i in range(qc.num_parameters)}
                )
        return qc

    # ----- observables suite (M10) --------------------------------------

    def _active_rdm1(self) -> np.ndarray:
        """Spin-summed 1-RDM in the active-MO basis, from whichever solver ran."""
        if self.results is None:
            self.solve()
        route = self.results.get('solver')
        if route == 'sqd':
            return self._sqd_solver.get_1rdm_active_mo()
        if route == 'ci':
            from pyscf import fci
            na = (self.n_electrons + self.spec.spin) // 2
            nb = self.n_electrons - na
            return np.asarray(fci.direct_spin1.make_rdm1(
                self.results['fci_vector'], self.n_orbitals, (na, nb)))
        raise NotImplementedError(
            f"observables() supports the 'ci' and 'sqd' routes; got {route!r}. "
            "(VQE-route 1-RDM extraction is a follow-up.)")

    def observables(self, which: str = 'core') -> Dict[str, Any]:
        """Property suite from the solved wavefunction.

        ``which='core'`` (default): cheap 1-RDM properties — dipole, natural-
        orbital occupations, M-diagnostic (multireference character), unpaired-
        electron count, Mulliken charges — plus the HOMO-LUMO gap.
        ``which='all'``: also finite-field polarizability and diamagnetic NMR
        shielding (expensive — extra solves / integrals).

        The 1-RDM properties (dipole, NOONs, M-diagnostic, unpaired count, charges)
        consume the genuine quantum density (`set_quantum_density_matrix`), not an
        HF fallback. **`homo_lumo_gap_ev` is the exception**: it is taken from the
        HF canonical orbital energies (Koopmans), not the correlated 1-RDM. The
        ``which='all'`` polarizability/NMR additions are not yet value-validated
        (see the trust map) — treat them as provisional.
        """
        from kanad.analysis.property_calculator import PropertyCalculator
        from kanad.core.density.quantum_rdm import (
            compute_natural_orbital_occupations, compute_m_diagnostic,
            compute_n_unpaired_electrons,
        )
        rdm1 = self._active_rdm1()
        # Density used for AO-basis properties (dipole, charges) must be the FULL
        # density — frozen core + active. The active-space solvers return only the
        # active 1-RDM; embedding it against the full nuclear charge without adding
        # the frozen-core electrons leaves a large net +charge and a physically
        # absurd, origin-dependent dipole (glycine CAS(6,6) gave 126 D). Build
        #   D_AO = 2·C_f C_fᵀ + C_a · ρ_active · C_aᵀ
        # (frozen orbitals are doubly occupied for the closed-shell active spaces
        # this module produces). NOON / M-diagnostic / unpaired-count below stay on
        # the ACTIVE rdm1 — those are active-space quantities and are correct as-is.
        acs = getattr(self.hamiltonian, 'active_space', None)
        cas_mo = self.results.get('active_mo_coeff')   # CASSCF-optimized MOs, if any
        if cas_mo is not None and acs is not None:
            # CASSCF route: the active 1-RDM is in the OPTIMIZED orbital basis,
            # whose columns are ordered [core | active | virtual]. The originally
            # selected acs.mo_coeff is stale after orbital optimization, so embed
            # against the optimized orbitals (ncore=|frozen|, ncas=n_orbitals).
            ncore = len(getattr(acs, 'frozen_indices', ()))
            ncas = self.n_orbitals
            C = np.asarray(cas_mo)
            Cf = C[:, :ncore]
            Ca = C[:, ncore:ncore + ncas]
            dm_ao_full = 2.0 * (Cf @ Cf.T) + Ca @ np.asarray(rdm1) @ Ca.T
            self.hamiltonian._quantum_density_matrix_ao = dm_ao_full
            self.hamiltonian._quantum_density_matrix = dm_ao_full
        elif acs is not None and len(getattr(acs, 'frozen_indices', ())) > 0:
            C = np.asarray(acs.mo_coeff)
            Cf = C[:, list(acs.frozen_indices)]
            Ca = C[:, list(acs.active_indices)]
            dm_ao_full = 2.0 * (Cf @ Cf.T) + Ca @ np.asarray(rdm1) @ Ca.T
            self.hamiltonian._quantum_density_matrix_ao = dm_ao_full
            self.hamiltonian._quantum_density_matrix = dm_ao_full
        else:
            self.hamiltonian.set_quantum_density_matrix(rdm1)
        pc = PropertyCalculator(self.hamiltonian)

        dip = pc.compute_dipole_moment(method='auto')
        noons = compute_natural_orbital_occupations(rdm1)
        # density_source from PropertyCalculator labels any quantum 1-RDM 'vqe';
        # report the ACTUAL solver route here so CI/SQD aren't mislabelled 'vqe'.
        route = self.results.get('solver')
        density_source = route if route in ('ci', 'sqd', 'vqe') else dip.get('density_source')
        obs: Dict[str, Any] = {
            'energy': self.results['energy'],
            'solver': route,
            'n_qubits': self.n_qubits,
            'dipole_debye': [float(x) for x in dip['dipole_vector']],
            'dipole_magnitude_debye': float(dip['dipole_magnitude']),
            'density_source': density_source,
            'natural_orbital_occupations': [float(x) for x in noons],
            'm_diagnostic': float(compute_m_diagnostic(rdm1)),
            'n_unpaired_electrons': float(compute_n_unpaired_electrons(rdm1)),
            'homo_lumo_gap_ev': self._homo_lumo_gap_ev(),
        }
        if 'conditions' in self.results:
            obs['conditions'] = self.results['conditions']
        if which == 'all':
            # Response properties (polarizability / diamagnetic NMR) are computed by
            # finite field, which re-runs the mean field. That works for a FULL active
            # space (frozen=[] canonical orbitals) but NOT for a frozen-core or rotated
            # (mp2no/avas) space — there the finite-field cannot re-derive the active
            # orbitals, so compute_polarizability spins on "Could not compute HF
            # reference" / eventually raises. Skip those EXTRAS upfront with a clear note
            # rather than crashing or stalling the whole observables('all') call; the
            # full-space case still computes (guarded by try/except). (CORE_BUGS B24.)
            acs = getattr(self.hamiltonian, 'active_space', None)
            _frozen_or_rotated = acs is not None and (
                len(getattr(acs, 'frozen_indices', ())) > 0
                or getattr(acs, 'method', '') in ('mp2no', 'avas'))
            if _frozen_or_rotated:
                obs['polarizability_mean_au'] = float('nan')
                obs['nmr_diamagnetic_shielding'] = None
                obs['response_properties_unavailable'] = (
                    "finite-field polarizability/NMR are not available for a frozen-core "
                    "or rotated (mp2no/avas) active space (the field cannot re-derive the "
                    "active orbitals). Use which='core', a full active space, or a "
                    "full-space Hamiltonian.")
            else:
                try:
                    if obs['solver'] == 'sqd':
                        acs = self.hamiltonian.active_space
                        pol = pc.compute_polarizability(
                            wavefunction='sqd',
                            sqd_active_frozen=list(acs.frozen_indices),
                            sqd_active_orbs=list(acs.active_indices),
                        )
                    else:
                        pol = pc.compute_polarizability(wavefunction='vqe')
                    # 'alpha_mean' is the isotropic mean; surface NaN (not a fake 0.0) if
                    # absent, so a missing value is visible, not "zero polarizability".
                    obs['polarizability_mean_au'] = float(pol.get('alpha_mean', float('nan')))
                except Exception as e:
                    obs['polarizability_mean_au'] = float('nan')
                    obs['polarizability_unavailable'] = f"{type(e).__name__}: {str(e)[:80]}"
                # Diamagnetic (Lamb) shielding ONLY — paramagnetic response not computed
                # (the result carries paramagnetic_part_missing=True); NOT a full shift.
                try:
                    obs['nmr_diamagnetic_shielding'] = pc.compute_diamagnetic_nmr_shielding(method='auto')
                except Exception as e:
                    obs['nmr_diamagnetic_shielding'] = None
                    obs['nmr_unavailable'] = f"{type(e).__name__}: {str(e)[:80]}"
        return obs

    def export_cube(self, filename: str, kind: str = 'density',
                    nx: int = 80, ny: int = 80, nz: int = 80) -> str:
        """Write a volumetric field to a Gaussian ``.cube`` file (VMD / kanad-app).

        Uses the **wavefunction-derived** (SQD/CI) density, not an HF fallback.

        Args:
            filename: output ``.cube`` path.
            kind: ``'density'`` (electron density ρ from the quantum 1-RDM) or
                ``'esp'`` (molecular electrostatic potential from that density).
            nx, ny, nz: grid resolution.

        Returns:
            the output filename.
        """
        from pyscf.tools import cubegen
        # Ensure the quantum density is on the Hamiltonian.
        if getattr(self.hamiltonian, '_quantum_density_matrix_ao', None) is None:
            self.hamiltonian.set_quantum_density_matrix(self._active_rdm1())
        dm_ao = self.hamiltonian.get_density_matrix('ao')
        mol = self.hamiltonian.mol
        if kind == 'density':
            cubegen.density(mol, filename, dm_ao, nx=nx, ny=ny, nz=nz)
        elif kind == 'esp':
            cubegen.mep(mol, filename, dm_ao, nx=nx, ny=ny, nz=nz)
        else:
            raise ValueError(
                f"kind must be 'density' or 'esp'; got {kind!r}. "
                "(spin-density / Fukui / ELF / Bader are follow-ups.)")
        return filename

    def _homo_lumo_gap_ev(self) -> float:
        # Frontier orbitals at the *total* molecular filling — mf.mo_energy is the
        # full canonical set (core + active + virtual), so index by all electrons,
        # not the active-space count.
        mo_e = np.asarray(self.mf.mo_energy)
        # Derive frontier indices from the actual occupations, so open-shell
        # (odd-electron / ROHF) systems use the true singly-occupied frontier
        # rather than nelectron//2 (which is wrong when occ is not 2/0).
        # Builder only produces RHF/ROHF (system_spec.py:89), so mo_occ is 1D.
        occ = np.asarray(self.mf.mo_occ)
        occ_idx = np.where(occ > 0)[0]
        vir_idx = np.where(occ == 0)[0]
        if occ_idx.size == 0 or vir_idx.size == 0:
            return float('nan')
        homo_i, lumo_i = int(occ_idx.max()), int(vir_idx.min())
        return float((mo_e[lumo_i] - mo_e[homo_i]) * 27.211386245988)

    def reactivity_descriptors(self, smiles: Optional[str] = None) -> Dict[str, Any]:
        """Conceptual-DFT global reactivity indices (Parr/Pearson).

        Returns electronegativity χ, chemical hardness η, global softness S, and
        electrophilicity ω (all eV). **These are HF-level**: they come from the
        HF canonical frontier-orbital energies in Koopmans' approximation
        (``source='koopmans_hf'``), NOT from the correlated SQD/CI wavefunction.
        The dipole magnitude carried alongside *is* wavefunction-derived (from the
        solved 1-RDM) — don't conflate the two provenances.

        If ``smiles`` is given (and RDKit is installed), also returns the validated
        physicochemical descriptors (Crippen logP, Ertl TPSA, H-bond counts, ...)
        and the Lipinski/Veber/Ghose rule-filter verdicts. The two families are
        kept separate — physicochemical descriptors are standard cheminformatics,
        not quantum.
        """
        from kanad.analysis.molecular_descriptors import (
            quantum_reactivity, physicochemical_from_smiles, druglikeness_rules,
        )
        ev = 27.211386245988
        # Minimal-basis guard (I1): on sto-Ng / minao the virtual-orbital energies are
        # unphysical (LUMO can sit at +6 eV), so Koopmans χ/η/ω and the HOMO-LUMO gap are
        # unreliable. Warn rather than silently return garbage descriptors.
        _basis = str(getattr(getattr(self.mf, 'mol', None), 'basis', '')
                     or getattr(self.spec, 'basis', '')).lower()
        if any(b in _basis for b in ('sto-', 'minao')):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "reactivity_descriptors on a minimal basis (%s): virtual-orbital energies are "
                "unphysical, so Koopmans χ/η/ω and the HOMO-LUMO gap are unreliable — use cc-pVDZ "
                "or larger for trustworthy reactivity descriptors.", _basis)
        mo_e = np.asarray(self.mf.mo_energy)
        # Frontier indices from actual occupations (handles open-shell ROHF),
        # not nelectron//2. Builder only produces RHF/ROHF so mo_occ is 1D.
        occ = np.asarray(self.mf.mo_occ)
        occ_idx = np.where(occ > 0)[0]
        vir_idx = np.where(occ == 0)[0]
        if occ_idx.size == 0 or vir_idx.size == 0:
            raise ValueError(
                "reactivity_descriptors needs both an occupied HOMO and a virtual "
                f"LUMO; this molecule has n_occ={occ_idx.size}, "
                f"n_vir={vir_idx.size}, n_mo={len(mo_e)}.")
        homo_i, lumo_i = int(occ_idx.max()), int(vir_idx.min())
        homo_ev, lumo_ev = float(mo_e[homo_i] * ev), float(mo_e[lumo_i] * ev)

        # The descriptors themselves are HF-level (Koopmans, above). The dipole is
        # an OPTIONAL wavefunction-derived decoration — only attach it if a solution
        # already exists. Never trigger a fresh solve here: on a large VQE-route
        # system that means running an intractable full VQE (and the VQE 1-RDM path
        # raises NotImplementedError anyway), so the solve would be both expensive
        # and futile. Cheap HF descriptors must stay cheap.
        dipole = None
        if self.results is not None:
            try:
                dipole = self.observables('core')['dipole_magnitude_debye']
            except Exception:
                pass

        qr = quantum_reactivity(homo_ev, lumo_ev, dipole_debye=dipole,
                                source='koopmans_hf')
        out: Dict[str, Any] = {'quantum_reactivity': qr}
        if smiles is not None:
            phys = physicochemical_from_smiles(smiles)
            out['physicochemical'] = phys
            out['druglikeness'] = druglikeness_rules(phys)
        return out

    # ----- geometry-parametric closure (dynamics / reactions) -----------

    def energy_fn(self):
        """Return ``energy_fn(atoms_bohr, warm_state) -> (energy_Ha, warm_state)``.

        Exactly the contract `dynamics.quantum_forces.compute_numerical_forces`
        and `run_quantum_md` expect. Re-materializes the spec at each geometry
        and solves, **warm-started** from the previous geometry's solution: the
        incoming ``warm_state`` (VQE parameters or an SQD determinant subspace)
        is threaded into `solve()`, and the new solution is returned as the next
        ``warm_state``. Under the default freeze policy the active space — and
        thus the parameter count / determinant labels — is constant along the
        scan, so warm-starting is valid and cuts per-geometry cost.
        """
        spec = self.spec

        def _energy_fn(atoms_bohr, warm_state=None):
            mf, ham = spec.materialize_at(atoms_bohr)
            qs = QuantumSystem(spec, mf, ham)
            # Bare electronic energy only — no per-geometry condition corrections
            # (a Hessian-per-step thermal term / solvation would corrupt the force). B13.
            res = qs.solve(warm_state=warm_state, apply_conditions=False)
            payload = res.get('determinants') if res.get('solver') == 'sqd' \
                else res.get('parameters')
            return res['energy'], payload

        return _energy_fn
