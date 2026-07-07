"""Selected-CI subspace diagonalizers (core.ci.selected_ci).

Indigenous home for diagonalizing H in a selected-determinant subspace, extracted
from solvers/sampling_sqd.py methods (reorg Phase B1, 2026-05-31). Two engines:

  diagonalize_pyscf  — PySCF direct_spin1 (dense <=500 / sparse-SC <=200k /
                       matrix-free >200k), full FCI-tensor guard for high-q/large-CAS,
                       ⟨S²⟩ reporting + spin_s multiplicity selection (direct path).
  diagonalize_custom — homegrown bit-level Slater-Condon (core.ci.slater_condon),
                       interleaved-JW build + block-convention sign correction.

Both preserve the validated sign conventions (CORE_BUGS B14/B15/B16/B17). Returned
dict matches the solver's expectations: energy, eigenvector, determinants,
n_determinants (+ s_squared/spin_enforced for the pyscf path).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from kanad.core.ci.slater_condon import (
    _split_alpha_beta, _interleave_to_block_sign, _build_sparse_h_subspace,
    _h_diag, _slater_condon_offdiag,
)

logger = logging.getLogger(__name__)


def s_squared_of_subspace(determinants, coeffs_block, n_orb, n_a, n_b):
    """⟨S²⟩ of a selected-determinant subspace vector via ``direct_spin1.contract_ss``.

    Args:
        determinants: iterable of integer determinant bitstrings.
        coeffs_block: per-determinant coefficients ALREADY in PySCF block convention
            (the caller applies any interleaved-JW -> block sign first).
        n_orb: active spatial orbitals.
        n_a, n_b: alpha / beta electron counts (the (n_a, n_b) CI sector).

    Returns:
        ``⟨ψ|Ŝ²|ψ⟩ / ⟨ψ|ψ⟩``. Single home for the cistring-embed + contract_ss
        ⟨S²⟩ that solvers (sampling_sqd) previously reimplemented inline.
    """
    from pyscf.fci import cistring, direct_spin1
    strs_a = cistring.make_strings(range(n_orb), n_a)
    strs_b = cistring.make_strings(range(n_orb), n_b)
    a_idx = {int(s): i for i, s in enumerate(strs_a)}
    b_idx = {int(s): i for i, s in enumerate(strs_b)}
    vf = np.zeros((len(strs_a), len(strs_b)))
    for d, c in zip(determinants, coeffs_block):
        a, b = _split_alpha_beta(int(d), n_orb)
        if a in a_idx and b in b_idx:
            vf[a_idx[a], b_idx[b]] = c
    ssv = direct_spin1.contract_ss(vf, n_orb, (n_a, n_b))
    nrm = float(np.einsum('ij,ij->', vf, vf)) + 1e-300
    return float(np.einsum('ij,ij->', vf, ssv) / nrm)


def fci_excited_states(h1, eri, nuc, n_orb, n_e, n_states,
                       spin_s=None, target_sz=0.0, conv_tol=1e-11):
    """Lowest ``n_states`` FULL-FCI roots, optionally restricted to one spin
    multiplicity by ⟨S²⟩ filtering. Returns ``(energies, civecs)``.

    Why not ``fci.addons.fix_spin_``: that pins the multiplicity with a level
    shift ``λ·(Ŝ² − s(s+1))``. When the default λ (0.2) is smaller than the gap
    between an off-multiplicity state and the next target-multiplicity state, the
    *shifted* off-multiplicity state stays ranked among the requested roots and is
    returned with a penalty-contaminated energy. Concretely on H₂/STO-3G the
    lowest triplet (−0.5308) shifts to −0.3308 and out-ranks the true second
    singlet (the 1σ_u² state at +0.4831), so ``nroots=3`` returns the shifted
    triplet as "S1" — wrong physics, and it disagrees with the SQD route's
    ⟨S²⟩-filtered spectrum.

    This is shift-free: diagonalize the bare H for extra roots, compute exact
    ⟨S²⟩ per root, keep the lowest ``n_states`` whose ⟨S²⟩ matches ``s(s+1)``,
    growing ``nroots`` until enough are found (capped at the FCI dimension). With
    ``spin_s=None`` it returns the raw lowest ``n_states`` (mixed multiplicity).
    """
    from pyscf import fci
    from pyscf.fci import cistring

    n_a = (n_e + int(round(2 * target_sz))) // 2
    n_b = n_e - n_a
    cis = fci.direct_spin1.FCI()
    cis.conv_tol = conv_tol
    # pyscf's default FCI max_memory (~4 GB) spuriously aborts with "Not enough
    # memory for FCI solver" well below what the host can handle (e.g. CAS(16,16)
    # needs ~10 GB; a 240 GB box has it). Size the cap from available RAM. (U1)
    try:
        import psutil
        cis.max_memory = max(4000, int(psutil.virtual_memory().available / 1e6 * 0.7))
    except Exception:
        cis.max_memory = 32000

    if spin_s is None:
        e, c = cis.kernel(h1, eri, n_orb, (n_a, n_b), ecore=nuc, nroots=n_states)
        e = list(np.atleast_1d(e))
        c = list(c) if isinstance(c, (list, tuple)) else [c]
        return [float(x) for x in e], c

    ss_t = spin_s * (spin_s + 1.0)
    n_dim = int(cistring.num_strings(n_orb, n_a) * cistring.num_strings(n_orb, n_b))
    want = n_states
    while True:
        n_roots = min(n_dim, max(want, n_states))
        e, c = cis.kernel(h1, eri, n_orb, (n_a, n_b), ecore=nuc, nroots=n_roots)
        e = list(np.atleast_1d(e))
        c = list(c) if isinstance(c, (list, tuple)) else [c]
        sel_e, sel_c = [], []
        for ei, ci in zip(e, c):
            ss = fci.spin_op.spin_square(ci, n_orb, (n_a, n_b))[0]
            if abs(ss - ss_t) < 0.3:
                sel_e.append(float(ei))
                sel_c.append(ci)
                if len(sel_e) >= n_states:
                    break
        if len(sel_e) >= n_states or n_roots >= n_dim:
            if len(sel_e) < n_states:
                logger.warning(
                    "fci_excited_states: only %d of %d requested roots have "
                    "S=%.1f (⟨S²⟩=%.2f) in the full FCI — returning what exists.",
                    len(sel_e), n_states, spin_s, ss_t)
            return sel_e[:n_states], sel_c[:n_states]
        want = n_roots + n_states


def diagonalize_pyscf(determinants, h1, h2, nuc, n_orb, n_e, target_sz=0.0, spin_s=None, device='auto'):
    """Diagonalize H in the selected-determinant subspace using PySCF's
    `direct_spin1` Slater-Condon engine, **matrix-free** via a scipy
    LinearOperator. Returns the same dict shape as the homegrown path.

    Why matrix-free (M11a fix, 2026-05-27):
    - Old: built dense ``H_sub[N_det, N_det]`` by applying H to each
      unit vector via ``contract_2e`` and extracting matrix elements.
      Cost: O(N_det · N_FCI) ops + dense N_det² storage. OOMs around
      N_det ~ 10000 (M11a iterative expansion hit this at N_det=21358).
    - New: a single ``H @ v`` step expands the N_det-length subspace
      vector to the full FCI tensor, calls ``contract_2e`` once
      (O(N_FCI)), and projects back. ``scipy.eigsh`` with a
      LinearOperator finds the ground state in ~10-50 mat-vec products
      → O(N_FCI) total. Works to N_det = N_FCI (i.e., full CASCI).

    Scaling envelope:
    - N_FCI = C(n_orb, n_a) · C(n_orb, n_b).
    - For (10,10) active: N_FCI = 252² = 63 504 (M11a).
    - For (12,12) active: N_FCI = 924² ≈ 854 000 (M11b — still tractable).
    - For (30,30) active: N_FCI ≈ 2.4×10¹⁶ (M11e — needs DMRG/AFQMC instead).
    """
    from pyscf.fci import cistring, direct_spin1
    from scipy.sparse.linalg import eigsh, LinearOperator

    # Use target_sz so open-shell (triplet, doublet etc.) work correctly.
    # Sz = (n_α - n_β)/2  →  n_α = (n_e + 2·Sz)/2
    n_a = (n_e + int(round(2 * target_sz))) // 2
    n_b = n_e - n_a

    # Convert bitstrings to (alpha_str, beta_str) PySCF format
    strs_a = cistring.make_strings(range(n_orb), n_a)
    strs_b = cistring.make_strings(range(n_orb), n_b)
    a_idx = {int(s): i for i, s in enumerate(strs_a)}
    b_idx = {int(s): i for i, s in enumerate(strs_b)}

    # Map each bitstring → (alpha_index, beta_index) in PySCF CI layout.
    # Determinants OUTSIDE the (n_a, n_b) sector (e.g. injected via
    # seed_determinants / cross-geometry MD warm-starts) have no CI position.
    # DROP them and rebind `determinants` to the kept subset, so the returned
    # determinant list stays 1:1 with the eigenvector (which is built over
    # ci_positions). Keeping the full list desynced the two: get_1rdm_active_mo's
    # zip(dets, evec) misaligned coefficients to the WRONG determinants (silent
    # RDM corruption on the dense/matrix-free paths) and the ⟨S²⟩ scatter
    # vf[ci_a, ci_b] = evec crashed on the sparse path. (CORE_BUGS B16.)
    ci_positions = []
    kept_determinants = []
    for d in determinants:
        a, b = _split_alpha_beta(int(d), n_orb)
        if a in a_idx and b in b_idx:
            ci_positions.append((a_idx[a], b_idx[b]))
            kept_determinants.append(int(d))
    if not ci_positions:
        return {'energy': float('inf'), 'eigenvector': np.array([]),
                'determinants': [], 'n_determinants': 0}

    # GPU fast path: offload the kept-subspace build+diagonalize to rocm-planck's
    # det_ci. Only when no spin multiplicity is enforced — det_ci returns the
    # lowest root regardless of ⟨S²⟩, so spin_s selection stays on the pyscf path.
    if spin_s is None:
        from kanad.core.ci.gpu_ci import try_planck_det_ci
        _gpu = try_planck_det_ci(h1, h2, nuc, kept_determinants, n_orb, n_e,
                                 sz=target_sz, device=device)
        if _gpu is not None:
            _ev = _gpu['eigenvector']
            try:
                _ss = s_squared_of_subspace(kept_determinants, _ev, n_orb, n_a, n_b)
            except Exception:
                _ss = None
            return {'energy': _gpu['energy'], 'eigenvector': _ev,
                    'determinants': kept_determinants, 'n_determinants': len(kept_determinants),
                    's_squared': _ss, 'spin_enforced': False,
                    'device_used': _gpu.get('device_used')}
    determinants = kept_determinants

    n_det = len(ci_positions)
    n_fci_a, n_fci_b = len(strs_a), len(strs_b)

    # Pre-absorb h1 into h2 for direct_spin1.contract_2e (one-time)
    h2e_eff = direct_spin1.absorb_h1e(h1, h2, n_orb, (n_a, n_b), 0.5)

    # Cache ci_positions as numpy index arrays for fast scatter/gather
    ci_a = np.array([p[0] for p in ci_positions], dtype=np.int64)
    ci_b = np.array([p[1] for p in ci_positions], dtype=np.int64)

    # ⟨S²⟩ via contract_ss needs the FULL FCI tensor (n_fci_a × n_fci_b). That is
    # cheap at <=~40 qubits but explodes for large dilute active spaces (e.g.
    # CAS(10,38)=76q → C(38,5)²≈2.5e11 floats ≈ 2 TB). Guard it: skip ⟨S²⟩ above a
    # memory budget so the SPARSE Slater-Condon ENERGY path (which never forms the
    # full tensor) can scale to 60-100+ qubit subspaces on real hardware.
    _SS_MAX_FCI = 2_000_000_000  # ~16 GB float64 ceiling for the full-tensor S² embed

    def _ss_of(evec):
        """⟨S²⟩ of a subspace eigenvector; None if the full FCI tensor is too large."""
        if n_fci_a * n_fci_b > _SS_MAX_FCI:
            return None
        vf = np.zeros((n_fci_a, n_fci_b)); vf[ci_a, ci_b] = evec
        ssv = direct_spin1.contract_ss(vf, n_orb, (n_a, n_b))
        nrm = float(np.einsum('ij,ij->', vf, vf)) + 1e-300
        return float(np.einsum('ij,ij->', vf, ssv) / nrm)

    def _finalize(energy, evec, spin_enforced=False):
        """Attach ⟨S²⟩ (None if too large to embed), warn on contamination."""
        ss = _ss_of(evec)
        if ss is not None:
            ss_min = abs(target_sz) * (abs(target_sz) + 1.0)
            if not spin_enforced and ss - ss_min > 0.5:
                logger.warning(
                    "SQD: solution ⟨S²⟩=%.3f exceeds the minimal %.3f for S_z=%.1f — "
                    "likely SPIN-CONTAMINATED (a higher-spin component sits at/below the "
                    "target, so the energy may fall below the spin-pure reference). Pass "
                    "spin_s=<target S> to enforce the multiplicity." % (ss, ss_min, target_sz))
        return {'energy': float(energy), 'eigenvector': evec,
                'determinants': determinants, 'n_determinants': n_det,
                's_squared': (None if ss is None else float(ss)),
                'spin_enforced': bool(spin_enforced)}

    # High-qubit / large-CAS guard threshold (full FCI tensor ceiling, ~16 GB
    # float64). Defined HERE — before the spin_s branch — because that branch
    # also materializes the full n_fci_a × n_fci_b tensor (np.zeros below), so
    # it must respect the same budget. (AUDIT H14: spin_s branch previously
    # preceded this guard and OOMed / TypeError'd on large dilute CAS, since
    # _ss_of() returns None above the budget and abs(None - ss_t) crashes.)
    _FCI_TENSOR_MAX = 2_000_000_000

    # ---- S²-targeted path (opt-in via spin_s) ----------------------------
    # Robust multiplicity selection WITHOUT a tuned penalty: diagonalize the
    # bare subspace H for the lowest K roots, then return the LOWEST root whose
    # ⟨S²⟩ matches s(s+1). This is exact (no shift to tune), handles both the
    # ground multiplicity (e.g. spin_s=0 → singlet) and excited multiplicities
    # (spin_s=1 → the M_s-sector triplet), and — unlike an S² penalty — does not
    # suffer eigsh convergence trouble when target and off-target states are
    # near-degenerate. If the sampled subspace contains NO root of the requested
    # multiplicity, it warns and returns the lowest (so the gap isn't faked).
    # AUDIT H14: only take this path when the full FCI tensor fits the budget;
    # above it, fall through to the sparse Slater-Condon guard below (S²-targeting
    # is unavailable there because ⟨S²⟩ needs the full tensor).
    if spin_s is not None and n_fci_a * n_fci_b <= _FCI_TENSOR_MAX:
        ss_t = spin_s * (spin_s + 1.0)

        if n_det <= 64:
            # Small subspace: dense build + full eigh (eigsh needs k<ncv<=n,
            # which is impossible for tiny n_det — this also avoids the
            # 'ncv must be k<ncv<=n' crash when a sample collapses to a few dets).
            H_sub = np.zeros((n_det, n_det))
            for j in range(n_det):
                ej = np.zeros((n_fci_a, n_fci_b)); ej[ci_a[j], ci_b[j]] = 1.0
                Hv = direct_spin1.contract_2e(h2e_eff, ej, n_orb, (n_a, n_b))
                H_sub[:, j] = Hv[ci_a, ci_b]
            H_sub += nuc * np.eye(n_det)
            evals_k, evecs_k = np.linalg.eigh(H_sub)
        else:
            def _matvec_bare(v):
                vf = np.zeros((n_fci_a, n_fci_b)); vf[ci_a, ci_b] = v
                Hv = direct_spin1.contract_2e(h2e_eff, vf, n_orb, (n_a, n_b))
                return Hv[ci_a, ci_b] + nuc * v

            Hop = LinearOperator(shape=(n_det, n_det), matvec=_matvec_bare, dtype=np.float64)
            kk = int(min(12, n_det - 2))
            ncv = int(min(n_det - 1, max(2 * kk + 1, 40)))
            try:
                evals_k, evecs_k = eigsh(Hop, k=kk, which='SA', tol=1e-7, maxiter=800, ncv=ncv)
            except Exception:
                kk = max(2, kk // 2)
                evals_k, evecs_k = eigsh(Hop, k=kk, which='SA', tol=1e-5,
                                         maxiter=1500, ncv=min(n_det - 1, max(2 * kk + 1, 25)))
        order = np.argsort(evals_k)
        chosen = None
        for idx in order:
            # AUDIT H14: _ss_of() returns None when the full FCI tensor exceeds
            # the budget; guard it so abs(None - ss_t) can never raise TypeError.
            ss_idx = _ss_of(evecs_k[:, idx])
            if ss_idx is not None and abs(ss_idx - ss_t) < 0.3:
                chosen = int(idx); break
        if chosen is None:
            chosen = int(order[0])
            ss0 = _ss_of(evecs_k[:, order[0]])
            logger.warning(
                "spin_s=%.1f (S²target=%.2f): no root with that multiplicity in the "
                "sampled %d-determinant subspace (the closed-shell seed may not cover "
                "this multiplicity — increase n_samples or seed from the open-shell "
                "reference). Returning the lowest root (⟨S²⟩=%s)."
                % (spin_s, ss_t, n_det, "n/a" if ss0 is None else "%.3f" % ss0))
        return _finalize(float(evals_k[chosen]), evecs_k[:, chosen], spin_enforced=True)

    # High-qubit / large-CAS guard: the dense (<=500) and matrix-free (>200k)
    # paths both MATERIALIZE the full FCI tensor (n_fci_a × n_fci_b) — at
    # CAS(10,38)=76q that is 501942² ≈ 1.8 TiB. Force the SPARSE Slater-Condon
    # path (no full tensor; now arbitrary-precision for >62q) and CAP the
    # subspace: noisy hardware at high q yields a huge, mostly-spurious
    # determinant set, so we keep a tractable slice. Energy is then sparse-exact
    # within the capped subspace (approximate vs the full sampled set).
    # _FCI_TENSOR_MAX is defined above (before the spin_s branch) so both gates
    # share one budget (AUDIT H14).
    if n_fci_a * n_fci_b > _FCI_TENSOR_MAX:
        from scipy.sparse.linalg import eigsh as sparse_eigsh
        # The arbitrary-precision sparse build is O(n_det · occ²·virt²) (no numpy
        # vectorization at >62q), and high-q virt is large (66 at 76q), so the cap
        # must be modest to complete. The high-q subspace is noise-dominated
        # anyway → a tight cap is honest, not lossy.
        DET_CAP = 4_000
        dets_hi = list(determinants)
        if len(dets_hi) > DET_CAP:
            logger.warning("High-q subspace capped %d→%d dets (full FCI tensor %.2e too "
                           "large for dense/matrix-free; sparse path)." % (len(dets_hi), DET_CAP, n_fci_a * n_fci_b))
            dets_hi = dets_hi[:DET_CAP]
        nd = len(dets_hi)
        H_sparse, _ = _build_sparse_h_subspace(dets_hi, h1, h2, nuc, n_orb)
        kk = 1
        ncv = int(min(nd - 1, max(2 * kk + 1, 40)))
        ev, evec = sparse_eigsh(H_sparse, k=1, which='SA', tol=1e-6, maxiter=600, ncv=ncv)
        signs = np.array([_interleave_to_block_sign(int(d), n_orb) for d in dets_hi])
        res = {'energy': float(ev[0]), 'eigenvector': evec[:, 0] * signs,
               'determinants': dets_hi, 'n_determinants': nd,
               's_squared': None, 'spin_enforced': False}
        return res

    # Small-subspace fast path: dense build is fine, exact eigh
    SMALL = 500
    if n_det <= SMALL:
        H_sub = np.zeros((n_det, n_det))
        for j in range(n_det):
            ci_e_j = np.zeros((n_fci_a, n_fci_b))
            ci_e_j[ci_a[j], ci_b[j]] = 1.0
            Hv = direct_spin1.contract_2e(h2e_eff, ci_e_j, n_orb, (n_a, n_b))
            H_sub[:, j] = Hv[ci_a, ci_b]
        H_sub += nuc * np.eye(n_det)
        evals, evecs = np.linalg.eigh(H_sub)
        return _finalize(evals[0], evecs[:, 0])

    # Medium subspace (500 < N_det ≤ 200000): sparse Slater-Condon path.
    # M11b fix (2026-05-27): builds sparse H via bit-level SC, scipy
    # eigsh. M11c follow-up (2026-05-28): vectorized with numpy
    # bulk XOR + searchsorted → ~3× speedup. 18k dets now in ~17s;
    # raises the practical ceiling from 30k → ~200k dets in tractable
    # wall (~1-3 min). Above 200k we still fall through to matrix-free.
    SPARSE_OK = 200_000
    if n_det <= SPARSE_OK:
        from scipy.sparse.linalg import eigsh as sparse_eigsh
        H_sparse, n_nz = _build_sparse_h_subspace(
            determinants, h1, h2, nuc, n_orb,
        )
        logger.info(
            f"Sparse SC path: N_det={n_det}, N_nonzero={n_nz}, "
            f"sparsity={100*(1 - n_nz/n_det**2):.4f}%"
        )
        ncv = max(20, min(n_det - 1, 50))
        try:
            evals, evecs = sparse_eigsh(H_sparse, k=1, which='SA',
                                          tol=1e-7, maxiter=500, ncv=ncv)
        except Exception:
            evals, evecs = sparse_eigsh(H_sparse, k=1, which='SA',
                                          tol=1e-5, maxiter=1000,
                                          ncv=min(20, n_det - 1))
        order = np.argsort(evals)
        # The sparse H was built in interleaved JW convention; correct the
        # eigenvector to PySCF block convention so the downstream RDM embed
        # has the right fermionic phases (see _interleave_to_block_sign).
        signs = np.array([_interleave_to_block_sign(int(d), n_orb)
                          for d in determinants])
        return _finalize(evals[order][0], evecs[:, order[0]] * signs)

    # Matrix-free path for very large subspaces (>200k dets)
    def matvec(v_sub):
        """v_sub (N_det,) → H v_sub (N_det,) via scatter/contract/gather."""
        v_full = np.zeros((n_fci_a, n_fci_b))
        v_full[ci_a, ci_b] = v_sub
        Hv = direct_spin1.contract_2e(h2e_eff, v_full, n_orb, (n_a, n_b))
        out = Hv[ci_a, ci_b].copy()
        out += nuc * v_sub  # nuclear repulsion (constant on the diagonal)
        return out

    H_op = LinearOperator(shape=(n_det, n_det), matvec=matvec, dtype=np.float64)

    # Davidson-class via eigsh; SA = smallest algebraic
    # M11b empirics: matrix-free contract_2e on N_FCI~10⁶ is ~1-3s per matvec,
    # so we need to cap matvec count. tol=1e-6 (~µHa) is plenty for SQD
    # (we're targeting mHa accuracy), and ncv=max(20, min(N_det, 100)) gives
    # a larger Krylov subspace to converge in fewer restarts.
    ncv = max(20, min(n_det - 1, 100))
    try:
        evals, evecs = eigsh(H_op, k=1, which='SA', tol=1e-6,
                              maxiter=200, ncv=ncv)
    except Exception:
        # Fallback: smaller ncv, looser tol
        evals, evecs = eigsh(H_op, k=1, which='SA', tol=1e-4,
                              maxiter=500, ncv=min(20, n_det - 1))
    order = np.argsort(evals)
    return _finalize(evals[order][0], evecs[:, order[0]])


def diagonalize_custom(determinants, h1, h2, nuc, n_orb, n_e, device='auto'):
    """Rebuild + diagonalize CI matrix for a given determinant list.

    **Efficient connectivity** (M4 enhancement):
    - Old: O(N² · n) — iterate every pair.
    - New: O(N · n²) — for each det, enumerate its singles+doubles via
      bit operations and look them up in a hash table. Connected pairs
      only (most pairs differ by > 2 spin orbitals and have zero matrix
      element).
    - Sparse COO accumulator (faster construction than LIL).

    ``device`` ('auto'|'amd'|'nvidia'|'cpu'): when a GPU is available, the build +
    diagonalize is offloaded to rocm-planck's det_ci (drop-in result); else the
    native scipy path below runs.
    """
    # GPU fast path: planck.det_ci builds + diagonalizes the subspace on-device.
    # Its to_dict() is byte-compatible with the dict returned below.
    from kanad.core.ci.gpu_ci import try_planck_det_ci
    _gpu = try_planck_det_ci(h1, h2, nuc, list(determinants), n_orb, n_e, device=device)
    if _gpu is not None:
        return _gpu

    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh

    n_det = len(determinants)
    n_qubits = 2 * n_orb
    n_elec = n_e

    # Hash-based lookup: bitstring → row index
    det_to_idx = {d: i for i, d in enumerate(determinants)}

    # COO accumulators
    rows, cols, vals = [], [], []

    # Diagonal
    for i, di in enumerate(determinants):
        rows.append(i); cols.append(i)
        vals.append(_h_diag(di, h1, h2, n_qubits) + nuc)

    # Off-diagonal: for each det, enumerate connected partners and check
    # if they're in the subspace via O(1) hash lookup. Every element is
    # computed by the VALIDATED `_slater_condon_offdiag`, which recomputes the
    # JW double-excitation sign correctly via `_double_excitation_sign` and
    # keeps αβ-mixed doubles (multiset spin check + direct/exchange gates).
    # The old hand-rolled math here used `_diff_spin_orbitals`' 2-orbital sign
    # (structurally always +1 → wrong off-diagonal SIGN, CORE_BUGS B14) and a
    # bit-ordered `(p%2)!=(q%2)` test that DROPPED αβ-mixed doubles (CORE_BUGS
    # B15). Delegating fixes both and shares the default backend's code path.
    n_orb = n_orb
    for i, di in enumerate(determinants):
        occ_qubits = [q for q in range(n_qubits) if (di >> q) & 1]
        virt_qubits = [q for q in range(n_qubits) if not (di >> q) & 1]
        # Singles: same-spin i → a
        for q_o in occ_qubits:
            for q_v in virt_qubits:
                if (q_o % 2) != (q_v % 2):
                    continue
                dj = di ^ (1 << q_o) ^ (1 << q_v)
                j = det_to_idx.get(dj)
                if j is None or j <= i:  # symmetric; only upper-triangle
                    continue
                val = _slater_condon_offdiag(di, dj, h1, h2, n_orb)
                if val != 0.0:
                    rows.extend([i, j]); cols.extend([j, i])
                    vals.extend([val, val])
        # Doubles: i,j → a,b (Sz-conserving, INCLUDING αβ-mixed)
        for io_idx in range(len(occ_qubits)):
            for jo_idx in range(io_idx + 1, len(occ_qubits)):
                q_i, q_j = occ_qubits[io_idx], occ_qubits[jo_idx]
                for va_idx in range(len(virt_qubits)):
                    for vb_idx in range(va_idx + 1, len(virt_qubits)):
                        q_a, q_b = virt_qubits[va_idx], virt_qubits[vb_idx]
                        if (q_i % 2 + q_j % 2) != (q_a % 2 + q_b % 2):
                            continue
                        dj = di ^ (1 << q_i) ^ (1 << q_j) ^ (1 << q_a) ^ (1 << q_b)
                        j = det_to_idx.get(dj)
                        if j is None or j <= i:
                            continue
                        val = _slater_condon_offdiag(di, dj, h1, h2, n_orb)
                        if val != 0.0:
                            rows.extend([i, j]); cols.extend([j, i])
                            vals.extend([val, val])

    H = coo_matrix((vals, (rows, cols)), shape=(n_det, n_det)).tocsr()
    if n_det <= 200:
        evals, evecs = np.linalg.eigh(H.toarray())
    else:
        evals, evecs = eigsh(H, k=1, which='SA')
        order = np.argsort(evals)
        evals = evals[order]; evecs = evecs[:, order]
    # H built here in interleaved JW convention → correct the eigenvector to
    # PySCF block convention before the RDM embed (see _interleave_to_block_sign).
    n_orb = n_orb
    signs = np.array([_interleave_to_block_sign(int(d), n_orb) for d in determinants])
    return {
        'energy': float(evals[0]),
        'eigenvector': evecs[:, 0] * signs,
        'determinants': determinants,
        'n_determinants': n_det,
    }
