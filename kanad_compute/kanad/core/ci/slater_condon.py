"""Slater-Condon CI engine (core.ci.slater_condon).

Indigenous home for the bit-level Slater-Condon matrix-element engine that was
previously inlined in solvers/sampling_sqd.py. Extracted VERBATIM (2026-05-31,
reorg Phase B1) so the validated sign conventions are preserved bit-for-bit:
  - _slater_condon_offdiag recomputes the JW double-excitation sign correctly
    via _double_excitation_sign (CORE_BUGS B14) and keeps αβ-mixed doubles (B15);
  - _interleave_to_block_sign bridges interleaved-JW ↔ PySCF block ordering (B16/B17);
  - _det_arr switches to object dtype above 62 qubits (no int64 overflow).

High-level solvers (SamplingSQDSolver, and DeterministicCI's custom CI backend)
consume these. NOTE: the covalent FCI build in core/hamiltonians and the dynamics
NAC build (dynamics/nonadiabatic.py) use PySCF FCI primitives directly and do NOT
route through core.ci — they are distinct full-space / full-spectrum solvers, not
duplicates of this selected-subspace engine. JW spin convention: α at even qubit
2p, β at odd 2p+1; h2 in chemist notation g(pq|rs).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _det_arr(vals):
    """Bitstring/determinant array. Uses object dtype (arbitrary-precision Python
    ints) when any value exceeds the int64 range — i.e. above ~62 qubits — so a
    76/100-qubit determinant is not truncated at ingestion. Keeps the fast int64
    dtype for <=62 qubits (no regression to existing results)."""
    vals = list(vals)
    if vals and max(vals) >= (1 << 62):
        return np.array(vals, dtype=object)
    return np.array(vals, dtype=np.int64)

def _split_alpha_beta(occ: int, n_orbitals: int) -> tuple[int, int]:
    """Split an interleaved JW bitstring into (alpha_mask, beta_mask) spatial bits."""
    alpha = 0
    beta = 0
    for p in range(n_orbitals):
        if (occ >> (2 * p)) & 1:
            alpha |= (1 << p)
        if (occ >> (2 * p + 1)) & 1:
            beta |= (1 << p)
    return alpha, beta

def _interleave_to_block_sign(occ: int, n_orbitals: int) -> int:
    """Fermionic sign for reordering an interleaved-JW occupation (α at even
    qubit 2p, β at odd 2p+1) into PySCF's block layout (all α then all β).

    The subspace H in the sparse and custom Slater-Condon paths is built in
    interleaved convention, but PySCF's ``make_rdm1`` / ``contract_2e`` (used to
    turn the eigenvector into the 1-/2-RDM) assume block ordering. Dropping this
    permutation parity leaves the energy correct but corrupts the RDMs — and
    every observable derived from them (dipole, NMR, charges, NO occupations) —
    for subspaces that hit those paths. The dense (≤500) and matrix-free
    (>200k) paths build H via ``direct_spin1.contract_2e`` and are already in
    block convention, so they must NOT be sign-corrected.
    """
    occ_q = [q for q in range(2 * n_orbitals) if (occ >> q) & 1]
    target = sorted(occ_q, key=lambda q: (q & 1, q >> 1))   # block: spin, then orbital
    pos = {v: i for i, v in enumerate(occ_q)}
    perm = [pos[v] for v in target]
    inv = sum(1 for i in range(len(perm)) for j in range(i + 1, len(perm))
              if perm[i] > perm[j])
    return -1 if (inv & 1) else 1

def _count_bits(x: int) -> int:
    return bin(x).count('1')

def _fermion_sign(occ: int, p: int, q: int) -> int:
    """Sign of `a†_p a_q` acting on the slater determinant `occ`.

    Counts occupied orbitals between p and q (exclusive) in JW order.
    """
    if p == q:
        return 1
    lo, hi = min(p, q), max(p, q)
    count = 0
    for k in range(lo + 1, hi):
        if (occ >> k) & 1:
            count += 1
    return -1 if count % 2 == 1 else 1

def _diff_spin_orbitals(occ_I: int, occ_J: int) -> Optional[tuple]:
    """Identify spin orbitals that differ between two SDs.

    Returns:
        - ``(0, occ_I)``       if identical
        - ``(1, (p, q), sign)`` if differ by one orbital: I = a†_p a_q J
        - ``(2, (p, q, r, s), sign)`` if differ by two orbitals: I = a†_p a†_r a_s a_q J
        - ``None``              if differ by more than 2 (matrix element = 0)
    """
    diff_in_I = occ_I & ~occ_J  # orbitals in I but not J = creation indices for I
    diff_in_J = occ_J & ~occ_I  # orbitals in J but not I = annihilation indices for I
    n_diff = _count_bits(diff_in_I)
    if n_diff != _count_bits(diff_in_J):
        return None  # different N — Hamiltonian matrix element zero
    if n_diff == 0:
        return (0, occ_I)
    if n_diff == 1:
        p = (diff_in_I.bit_length() - 1)
        q = (diff_in_J.bit_length() - 1)
        # Compute sign: starting from occ_J, do a_q then a†_p
        sign = _fermion_sign(occ_J, p, q)
        return (1, (p, q), sign)
    if n_diff == 2:
        # NOTE: returns bit-ordered (p, q, r, s) tuple. This is consistent with
        # `_diagonalize_in_subspace`'s direct path which only consumed this
        # for the H₂-like small-subspace path (verified correct on H₂ FCI).
        # For larger active spaces we now delegate CI matrix construction to
        # PySCF's `selected_ci` (see `_diagonalize_in_subspace_pyscf`), so
        # the bit-ordered tuple here is only kept for diagnostic purposes
        # and the existing H₂/HeH⁺ tests.
        creations = []
        bits = diff_in_I
        while bits:
            b = (bits & -bits).bit_length() - 1
            creations.append(b)
            bits &= bits - 1
        annihilations = []
        bits = diff_in_J
        while bits:
            b = (bits & -bits).bit_length() - 1
            annihilations.append(b)
            bits &= bits - 1
        p, r = creations[0], creations[1]
        q, s = annihilations[0], annihilations[1]
        tmp = occ_J
        sign = 1
        sign *= _fermion_sign(tmp, q, q); tmp ^= (1 << q)
        sign *= _fermion_sign(tmp, s, s); tmp ^= (1 << s)
        sign *= _fermion_sign(tmp, r, r); tmp ^= (1 << r)
        sign *= _fermion_sign(tmp, p, p); tmp ^= (1 << p)
        return (2, (p, q, r, s), sign)
    return None

def _generate_singles_doubles(det: int, n_qubits: int, n_elec: int) -> set:
    """Generate all (N, Sz)-preserving single + double excitations of ``det``.

    Used by ``solve_iterative`` to expand the selected-CI subspace.
    Limits to spin-conserving moves (α→α, β→β) so the (N, Sz) sector
    stays correct.
    """
    out = set()
    n_orb = n_qubits // 2
    # Occupied spin-orbital indices
    occ_qubits = [q for q in range(n_qubits) if (det >> q) & 1]
    virt_qubits = [q for q in range(n_qubits) if not (det >> q) & 1]

    # Single excitations: i → a, same spin
    for i in occ_qubits:
        for a in virt_qubits:
            if (i % 2) != (a % 2):
                continue  # different spin → wrong Sz
            new = det ^ (1 << i) ^ (1 << a)
            out.add(new)

    # Double excitations: i,j → a,b
    for i_idx in range(len(occ_qubits)):
        for j_idx in range(i_idx + 1, len(occ_qubits)):
            i, j = occ_qubits[i_idx], occ_qubits[j_idx]
            for a_idx in range(len(virt_qubits)):
                for b_idx in range(a_idx + 1, len(virt_qubits)):
                    a, b = virt_qubits[a_idx], virt_qubits[b_idx]
                    # Spin-conservation: total Sz unchanged
                    # Move (i,j) → (a,b); spins must permute correctly.
                    if (i % 2 + j % 2) != (a % 2 + b % 2):
                        continue
                    new = det ^ (1 << i) ^ (1 << j) ^ (1 << a) ^ (1 << b)
                    out.add(new)
    return out

def _count_bits_below(occ: int, k: int) -> int:
    """Count occupied bits at indices < k in `occ`. JW string-parity helper."""
    if k <= 0:
        return 0
    mask = (1 << k) - 1
    return bin(occ & mask).count('1')

def _double_excitation_sign(occ_J: int, p: int, q: int, r: int, s: int) -> int:
    """JW sign of ``a†_p a†_r a_s a_q`` acting on ``|D_J⟩``.

    Applied operator order (right-to-left): a_q first, then a_s, then a†_r,
    then a†_p. Each operator contributes (-1)^{bits_below_k in current state}
    where the state is updated between applications.

    Returns:
        +1 or -1; or 0 if any operator acts on a state where its bit
        isn't in the correct state (e.g., a_q where bit q is already 0).
    """
    if q == s or p == r:
        return 0   # double-occupation of a fermion index: not allowed
    tmp = occ_J
    # a_q (bit q must be 1 in tmp)
    if not (tmp >> q) & 1:
        return 0
    sign = -1 if (_count_bits_below(tmp, q) & 1) else 1
    tmp ^= (1 << q)
    # a_s
    if not (tmp >> s) & 1:
        return 0
    sign *= -1 if (_count_bits_below(tmp, s) & 1) else 1
    tmp ^= (1 << s)
    # a†_r (bit r must be 0 in tmp)
    if (tmp >> r) & 1:
        return 0
    sign *= -1 if (_count_bits_below(tmp, r) & 1) else 1
    tmp ^= (1 << r)
    # a†_p
    if (tmp >> p) & 1:
        return 0
    sign *= -1 if (_count_bits_below(tmp, p) & 1) else 1
    tmp ^= (1 << p)
    return sign

def _slater_condon_offdiag(di: int, dj: int, h1: np.ndarray,
                            h2: np.ndarray, n_orb: int) -> float:
    """``⟨D_i|H|D_j⟩`` for di != dj via Slater-Condon rules.

    Returns 0 if the determinants differ by more than 2 spin-orbitals.
    Correctly handles αα, ββ, AND αβ-mixed doubles (the existing
    `_diagonalize_in_subspace` path skipped αβ doubles, which is the
    M11b bug we're fixing here).

    Convention:
        - JW interleaved: α at even qubit, β at odd qubit.
        - h2 is in chemist's notation: h2[p,q,r,s] = (pq|rs).
        - Spin conservation: the multiset {σ_p, σ_r} of bits set in I
          but not J must equal {σ_q, σ_s} of bits set in J but not I.
    """
    diff = _diff_spin_orbitals(di, dj)
    if diff is None or diff[0] == 0:
        return 0.0

    if diff[0] == 1:
        (p, q), sign = diff[1], diff[2]
        sigma_p = p & 1
        sigma_q = q & 1
        if sigma_p != sigma_q:
            return 0.0
        p_sp = p >> 1
        q_sp = q >> 1
        val = h1[p_sp, q_sp]
        # Common occupied (in BOTH I and J), exclude p and q
        common = di & dj
        for r_q in range(2 * n_orb):
            if (common >> r_q) & 1:
                r_sp = r_q >> 1
                sigma_r = r_q & 1
                # Coulomb: (pq|rr) for all spins r
                val += h2[p_sp, q_sp, r_sp, r_sp]
                # Exchange: -(pr|rq) only if spins match
                if sigma_r == sigma_p:
                    val -= h2[p_sp, r_sp, r_sp, q_sp]
        return sign * val

    if diff[0] == 2:
        (p, q, r, s), _bogus_sign = diff[1], diff[2]
        # NOTE: _diff_spin_orbitals' 2-orbital sign is broken (uses
        # _fermion_sign(tmp, X, X) which always returns 1, per the M5-A
        # audit comment). Compute the JW sign properly via the actual
        # operator-by-operator pass.
        sign = _double_excitation_sign(int(dj), p, q, r, s)
        if sign == 0:
            return 0.0
        sigma_p = p & 1
        sigma_q = q & 1
        sigma_r = r & 1
        sigma_s = s & 1
        # Spin conservation check (multiset equality)
        if (sigma_p + sigma_r) != (sigma_q + sigma_s):
            return 0.0
        p_sp = p >> 1
        q_sp = q >> 1
        r_sp = r >> 1
        s_sp = s >> 1
        val = 0.0
        # Direct (pq|rs) if σ_p=σ_q and σ_r=σ_s
        if sigma_p == sigma_q and sigma_r == sigma_s:
            val += h2[p_sp, q_sp, r_sp, s_sp]
        # Exchange -(ps|rq) if σ_p=σ_s and σ_r=σ_q
        if sigma_p == sigma_s and sigma_r == sigma_q:
            val -= h2[p_sp, s_sp, r_sp, q_sp]
        return sign * val

    return 0.0

def _build_sparse_h_subspace(determinants: list, h1: np.ndarray,
                              h2: np.ndarray, nuc: float, n_orb: int):
    """Sparse Slater-Condon CI Hamiltonian on the selected-det subspace.

    **Vectorized M11c fix (2026-05-28):** for each det, enumerates its
    singles + doubles as numpy arrays (bulk bit-XOR), bulk-checks
    membership via numpy ``searchsorted`` against the sorted det array,
    then computes SC matrix elements scalarly for valid connections.

    Speedup over the pure-Python version: ~10-30× at N_det ≈ 20k (sparse
    ceiling pushed from 30k → ~300k dets in tractable wall-time).

    Cost:  O(N_det · n_conn) bit operations, vectorized in numpy.
    Memory: O(N_nz × 16 bytes); N_nz is typically 0.05% × N_det².

    Returns: ``(scipy.sparse.csr_matrix, n_nonzero)``
    """
    from scipy.sparse import coo_matrix

    n_det = len(determinants)
    n_qubits = 2 * n_orb

    # ---- Arbitrary-precision path for > 62 qubits ----------------------------
    # The fast vectorized path below stores determinants as int64 (qubit_bits =
    # 1<<arange, di>>arange), which OVERFLOWS above ~62 qubits — a 76-qubit
    # determinant needs 76 bits. _h_diag and _slater_condon_offdiag already use
    # Python-int bit-ops, so here we drive them with pure-Python ints + a dict
    # lookup (no int64 anywhere). Slower (no numpy bulk ops) but unbounded in
    # qubit count → unlocks 76/90/100+ qubit SQD post-processing. The int64 path
    # is untouched for <=62q so existing results are unaffected. The threshold is
    # env-overridable (SQD_ARB_PREC_Q) so the arbitrary-precision path can be
    # validated against the int64 path on a small system.
    import os as _os
    _arb_thresh = int(_os.environ.get('SQD_ARB_PREC_Q', '62'))
    if n_qubits > _arb_thresh:
        from scipy.sparse import coo_matrix as _coo
        dets_py = [int(d) for d in determinants]
        det_index = {d: i for i, d in enumerate(dets_py)}
        rows, cols, vals = [], [], []
        for i, di in enumerate(dets_py):
            rows.append(i); cols.append(i); vals.append(_h_diag(di, h1, h2, n_qubits) + nuc)
            occ = [k for k in range(n_qubits) if (di >> k) & 1]
            virt = [k for k in range(n_qubits) if not ((di >> k) & 1)]
            # singles (spin-conserving: o,v same JW parity)
            for o in occ:
                for v in virt:
                    if (o & 1) != (v & 1):
                        continue
                    dj = di ^ (1 << o) ^ (1 << v)
                    j = det_index.get(dj)
                    if j is not None and j > i:
                        val = _slater_condon_offdiag(di, dj, h1, h2, n_orb)
                        if abs(val) > 1e-14:
                            rows += [i, j]; cols += [j, i]; vals += [val, val]
            # doubles (i<j occ, a<b virt, Sz-conserving)
            no = len(occ)
            for x in range(no):
                for y in range(x + 1, no):
                    o1, o2 = occ[x], occ[y]
                    for a in range(len(virt)):
                        for b in range(a + 1, len(virt)):
                            v1, v2 = virt[a], virt[b]
                            if ((o1 & 1) + (o2 & 1)) != ((v1 & 1) + (v2 & 1)):
                                continue
                            dj = di ^ (1 << o1) ^ (1 << o2) ^ (1 << v1) ^ (1 << v2)
                            j = det_index.get(dj)
                            if j is not None and j > i:
                                val = _slater_condon_offdiag(di, dj, h1, h2, n_orb)
                                if abs(val) > 1e-14:
                                    rows += [i, j]; cols += [j, i]; vals += [val, val]
        H = _coo((vals, (rows, cols)), shape=(n_det, n_det)).tocsr()
        return H, len(vals)

    dets_arr = np.asarray(determinants, dtype=np.int64)
    # Sort + inverse-permutation map for binary-search lookup
    sort_perm = np.argsort(dets_arr, kind='stable')
    sorted_dets = dets_arr[sort_perm]

    # Precompute per-qubit bitmasks once
    qubit_bits = (np.int64(1) << np.arange(n_qubits, dtype=np.int64))

    rows_list = []
    cols_list = []
    vals_list = []

    # Diagonal: vectorize across determinants
    diag_vals = np.empty(n_det)
    for i, di in enumerate(determinants):
        diag_vals[i] = _h_diag(int(di), h1, h2, n_qubits) + nuc
    rows_list.append(np.arange(n_det, dtype=np.int64))
    cols_list.append(np.arange(n_det, dtype=np.int64))
    vals_list.append(diag_vals)

    # Off-diagonal: per-det numpy enumeration + bulk lookup
    for i in range(n_det):
        di = int(dets_arr[i])

        # Occupied / virtual qubit indices for this det
        occ_q = np.where(((di >> np.arange(n_qubits)) & 1).astype(bool))[0].astype(np.int64)
        virt_q = np.setdiff1d(np.arange(n_qubits, dtype=np.int64), occ_q, assume_unique=True)

        # ---- Singles (same-spin: α→α and β→β) -----------------------
        # Outer-product all (occ, virt) pairs, filter by spin parity
        oo, vv = np.meshgrid(occ_q, virt_q, indexing='ij')
        spin_match = (oo & 1) == (vv & 1)
        oo_v = oo[spin_match]; vv_v = vv[spin_match]
        if oo_v.size > 0:
            singles_dj = di ^ qubit_bits[oo_v] ^ qubit_bits[vv_v]
            # Bulk binary-search lookup
            pos = np.searchsorted(sorted_dets, singles_dj)
            in_ss = (pos < n_det) & (sorted_dets[np.minimum(pos, n_det - 1)] == singles_dj)
            valid = in_ss
            for idx in np.where(valid)[0]:
                j_sorted = pos[idx]
                # searchsorted gives a position in `sorted_dets`; the ORIGINAL
                # index is sort_perm[pos] (sorted→original). Using inv_perm
                # (original→sorted) here scatters the element to the wrong
                # column unless the input list is already numerically sorted.
                j = int(sort_perm[j_sorted])
                if j <= i:
                    continue
                val = _slater_condon_offdiag(di, int(singles_dj[idx]), h1, h2, n_orb)
                if abs(val) > 1e-14:
                    rows_list.append(np.array([i, j], dtype=np.int64))
                    cols_list.append(np.array([j, i], dtype=np.int64))
                    vals_list.append(np.array([val, val]))

        # ---- Doubles (same-spin αα/ββ + mixed αβ, Sz-conserving) -----
        n_o = len(occ_q)
        n_v = len(virt_q)
        if n_o >= 2 and n_v >= 2:
            # All (i_idx, j_idx, a_idx, b_idx) with i<j, a<b
            i_idx, j_idx = np.triu_indices(n_o, k=1)
            a_idx, b_idx = np.triu_indices(n_v, k=1)
            # 4D mesh: combine occ-pair × virt-pair
            n_op = i_idx.size
            n_vp = a_idx.size
            ii = np.broadcast_to(i_idx[:, None], (n_op, n_vp)).ravel()
            jj = np.broadcast_to(j_idx[:, None], (n_op, n_vp)).ravel()
            aa = np.broadcast_to(a_idx[None, :], (n_op, n_vp)).ravel()
            bb = np.broadcast_to(b_idx[None, :], (n_op, n_vp)).ravel()
            qi = occ_q[ii]; qj = occ_q[jj]
            qa = virt_q[aa]; qb = virt_q[bb]
            # Sz conservation: (i.spin + j.spin) == (a.spin + b.spin)
            sz_ok = ((qi & 1) + (qj & 1)) == ((qa & 1) + (qb & 1))
            qi = qi[sz_ok]; qj = qj[sz_ok]; qa = qa[sz_ok]; qb = qb[sz_ok]
            if qi.size > 0:
                doubles_dj = (di
                              ^ qubit_bits[qi] ^ qubit_bits[qj]
                              ^ qubit_bits[qa] ^ qubit_bits[qb])
                pos = np.searchsorted(sorted_dets, doubles_dj)
                in_ss = (pos < n_det) & (sorted_dets[np.minimum(pos, n_det - 1)] == doubles_dj)
                for idx in np.where(in_ss)[0]:
                    j_sorted = pos[idx]
                    # sorted→original index (see singles path above).
                    j = int(sort_perm[j_sorted])
                    if j <= i:
                        continue
                    val = _slater_condon_offdiag(di, int(doubles_dj[idx]),
                                                   h1, h2, n_orb)
                    if abs(val) > 1e-14:
                        rows_list.append(np.array([i, j], dtype=np.int64))
                        cols_list.append(np.array([j, i], dtype=np.int64))
                        vals_list.append(np.array([val, val]))

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    H = coo_matrix((vals, (rows, cols)), shape=(n_det, n_det)).tocsr()
    return H, len(vals)

def _h_diag(occ: int, h1: np.ndarray, h2: np.ndarray, n_qubits: int) -> float:
    """``⟨D|H|D⟩`` for a Slater determinant.

    Uses JW spin convention (α at even, β at odd). ``h1`` and ``h2`` are spatial
    integrals in MO basis (chemist's notation for h2: g(pq|rs)).
    """
    # 1e: Σ_i h_ii (over occupied spin orbitals)
    n_orb = n_qubits // 2
    occ_alpha, occ_beta = _split_alpha_beta(occ, n_orb)

    e1 = 0.0
    for p in range(n_orb):
        if (occ_alpha >> p) & 1:
            e1 += h1[p, p]
        if (occ_beta >> p) & 1:
            e1 += h1[p, p]

    # 2e: 0.5 Σ_{ij} [(ii|jj) − (ij|ji)] with i,j over occupied spin-orbs
    # Spin-summed: J - K type terms.
    e2 = 0.0
    occ_a_list = [p for p in range(n_orb) if (occ_alpha >> p) & 1]
    occ_b_list = [p for p in range(n_orb) if (occ_beta >> p) & 1]
    # αα Coulomb-exchange
    for i in occ_a_list:
        for j in occ_a_list:
            if i != j:
                e2 += 0.5 * (h2[i, i, j, j] - h2[i, j, j, i])
    # ββ
    for i in occ_b_list:
        for j in occ_b_list:
            if i != j:
                e2 += 0.5 * (h2[i, i, j, j] - h2[i, j, j, i])
    # αβ Coulomb only (no exchange between opposite spins)
    for i in occ_a_list:
        for j in occ_b_list:
            e2 += h2[i, i, j, j]
    return e1 + e2
