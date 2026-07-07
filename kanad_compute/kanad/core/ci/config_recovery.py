"""Configuration recovery for sample-based SQD (core.ci.config_recovery).

Indigenous home for the (N, S_z)-sector filtering + Robledo-Moreno §III.A
configuration recovery, extracted VERBATIM from solvers/sampling_sqd.py
(reorg Phase B1, 2026-05-31). Decoupled from the solver: the iterative path
takes a `diagonalize_callback`, so any selected-CI diagonalizer can drive it.

JW spin convention: α at even qubit 2p, β at odd 2p+1.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from kanad.core.ci.slater_condon import _det_arr, _split_alpha_beta, _count_bits

logger = logging.getLogger(__name__)


def _filter_by_n_sz(bitstrings: np.ndarray, n_orbitals: int,
                    n_electrons: int, target_sz: float = 0.0) -> np.ndarray:
    """Keep bitstrings with correct N and S_z (drop-filter — no recovery).

    For closed-shell singlets, `n_alpha == n_beta == n_electrons // 2`.
    target_sz = (n_alpha − n_beta) / 2.
    """
    # Sz = (n_α - n_β) / 2 with n_α + n_β = n_e  →  n_α = (n_e + 2·Sz)/2
    n_alpha = (n_electrons + int(round(2 * target_sz))) // 2
    n_beta = n_electrons - n_alpha
    kept = []
    for occ in bitstrings:
        a, b = _split_alpha_beta(int(occ), n_orbitals)
        if _count_bits(a) == n_alpha and _count_bits(b) == n_beta:
            kept.append(int(occ))
    return _det_arr(kept)

def _recover_spin_sector(occ_spin: int, n_orbitals: int,
                          n_target: int,
                          mo_energies: Optional[np.ndarray] = None,
                          occupation_pref: Optional[np.ndarray] = None) -> Optional[int]:
    """Recover a valid spin-sector bitstring from a noisy sample.

    Greedy bit-flip toward the (N=n_target) sector. Two modes:

    1. **HF-energy mode (default, M4-D)**: ``occupation_pref=None``.
       Lower MO energy → "more likely occupied at HF". Drop highest-energy
       occupied bits if too many 1s; set lowest-energy unoccupied bits if
       too few. This is the M4-D single-shot baseline.

    2. **Marginal-occupation mode (M3 D2, Robledo-Moreno §III.A)**:
       ``occupation_pref`` is an array of per-orbital expected occupations
       ``n_p ∈ [0, 1]`` derived from the current subspace eigenvector.
       Drop bits at orbitals with LOWEST n_p; set bits at orbitals with
       HIGHEST n_p. This is the multi-round confidence-weighted path used
       by `_filter_with_iterative_recovery`.

    Args:
        occ_spin: Single-spin-sector bitstring (n_orbitals bits).
        n_orbitals: Number of spatial orbitals.
        n_target: Target occupation count.
        mo_energies: Optional ``np.ndarray`` of MO energies.
        occupation_pref: Optional per-orbital marginal occupations from
            an iterative-recovery round. **Takes priority over mo_energies.**

    Returns:
        Recovered bitstring with exactly ``n_target`` bits set, or ``None``
        if recovery is ambiguous.
    """
    n_cur = _count_bits(occ_spin)
    delta = n_cur - n_target
    if delta == 0:
        return occ_spin

    # Preference signal: higher pref → "more likely occupied"
    if occupation_pref is not None:
        # Marginal-occupation mode: pref = n_p directly
        pref = np.asarray(occupation_pref, dtype=float)
    elif mo_energies is not None:
        # HF-energy mode: pref = -ε (low ε ↔ high preference for occupation)
        pref = -np.asarray(mo_energies, dtype=float)
    else:
        # Default: PySCF index order (MO 0 most preferred)
        pref = -np.arange(n_orbitals, dtype=float)

    occupied = [p for p in range(n_orbitals) if (occ_spin >> p) & 1]
    unoccupied = [p for p in range(n_orbitals) if not (occ_spin >> p) & 1]

    if delta > 0:
        # Too many 1s — drop bits at LOWEST-preference occupied orbitals
        occupied.sort(key=lambda p: pref[p])     # ascending (lowest pref first)
        # Ambiguous if the orbital at the drop cut ties the first kept one:
        # there's no confident choice of which to drop. Report as ambiguous.
        if delta < len(occupied) and abs(pref[occupied[delta - 1]] - pref[occupied[delta]]) < 1e-9:
            return None
        bits_to_drop = occupied[:delta]
        new_occ = occ_spin
        for b in bits_to_drop:
            new_occ ^= (1 << b)
        return new_occ
    else:
        # Too few 1s — set bits at HIGHEST-preference unoccupied orbitals
        unoccupied.sort(key=lambda p: -pref[p])  # descending (highest pref first)
        n_set = -delta
        # Ambiguous if the orbital at the set cut ties the first skipped one.
        if n_set < len(unoccupied) and abs(pref[unoccupied[n_set - 1]] - pref[unoccupied[n_set]]) < 1e-9:
            return None
        bits_to_set = unoccupied[:n_set]
        new_occ = occ_spin
        for b in bits_to_set:
            new_occ ^= (1 << b)
        return new_occ

def _filter_with_recovery(bitstrings: np.ndarray, n_orbitals: int,
                          n_electrons: int, target_sz: float = 0.0,
                          mo_energies: Optional[np.ndarray] = None,
                          alpha_pref: Optional[np.ndarray] = None,
                          beta_pref: Optional[np.ndarray] = None) -> tuple:
    """Filter + recover noisy bitstrings (M4-D + M3 D2 extension).

    For each sampled bitstring:
    1. If (N, S_z) matches the target → keep as-is.
    2. Else → greedy bit-flip recovery in each spin sector, guided by
       either MO energies (HF mode) or per-orbital marginal occupations
       from a prior eigenvector (multi-round mode).

    Args:
        bitstrings, n_orbitals, n_electrons, target_sz: standard.
        mo_energies: HF orbital energies (used in default single-shot mode).
        alpha_pref, beta_pref: optional per-orbital marginal occupations
            from a previous SQD round's eigenvector (M3 D2 mode). When
            provided, they OVERRIDE mo_energies for recovery direction.

    Returns:
        ``(recovered_bitstrings, n_kept_directly, n_recovered, n_dropped)``.
    """
    n_alpha_target = (n_electrons + int(round(2 * target_sz))) // 2
    n_beta_target = n_electrons - n_alpha_target

    out = []
    n_kept_directly = 0
    n_recovered = 0
    n_dropped = 0

    for occ in bitstrings:
        a, b = _split_alpha_beta(int(occ), n_orbitals)
        if _count_bits(a) == n_alpha_target and _count_bits(b) == n_beta_target:
            out.append(int(occ))
            n_kept_directly += 1
            continue
        # Try recovery — pass marginal-occupation pref if available
        a_rec = _recover_spin_sector(a, n_orbitals, n_alpha_target,
                                       mo_energies, occupation_pref=alpha_pref)
        b_rec = _recover_spin_sector(b, n_orbitals, n_beta_target,
                                       mo_energies, occupation_pref=beta_pref)
        if a_rec is None or b_rec is None:
            n_dropped += 1
            continue
        # Re-interleave α (even qubits) + β (odd qubits)
        new_occ = 0
        for p in range(n_orbitals):
            if (a_rec >> p) & 1:
                new_occ |= (1 << (2 * p))
            if (b_rec >> p) & 1:
                new_occ |= (1 << (2 * p + 1))
        out.append(new_occ)
        n_recovered += 1

    return _det_arr(out), n_kept_directly, n_recovered, n_dropped

def _compute_orbital_marginals(determinants: list, weights: np.ndarray,
                                n_orbitals: int) -> tuple:
    """Per-orbital expected occupation under a subspace eigenvector.

    For each spatial orbital p, computes:
        n_α(p) = Σ_d |ψ_d|² · 1[α-orbital p occupied in det d]
        n_β(p) = Σ_d |ψ_d|² · 1[β-orbital p occupied in det d]

    These are the per-orbital marginal occupations under the current
    SQD eigenvector. Used by multi-round confidence-weighted recovery
    (M3 D2, Robledo-Moreno §III.A) to guide bit-flip choices toward
    orbitals with high expected occupation.

    Returns:
        ``(n_alpha, n_beta)`` — each an ``np.ndarray`` of length n_orbitals
        with entries in [0, 1]. Sum over orbitals equals (n_α_target,
        n_β_target) when the eigenvector is normalized.
    """
    w2 = np.abs(weights) ** 2
    w2 = w2 / max(w2.sum(), 1e-30)
    n_alpha = np.zeros(n_orbitals)
    n_beta = np.zeros(n_orbitals)
    for d, w in zip(determinants, w2):
        a, b = _split_alpha_beta(int(d), n_orbitals)
        for p in range(n_orbitals):
            if (a >> p) & 1:
                n_alpha[p] += w
            if (b >> p) & 1:
                n_beta[p] += w
    return n_alpha, n_beta

def _filter_with_iterative_recovery(
    bitstrings: np.ndarray, n_orbitals: int, n_electrons: int,
    diagonalize_callback, target_sz: float = 0.0,
    mo_energies: Optional[np.ndarray] = None,
    max_rounds: int = 5, energy_tol: float = 1e-5,
    record_history: bool = True,
) -> tuple:
    """Multi-round confidence-weighted iterative recovery (M3 D2,
    Robledo-Moreno §III.A).

    Round 0:
        Standard single-shot greedy recovery using MO energies as the
        preference signal (identical to `_filter_with_recovery`).
    Round k ≥ 1:
        1. Diagonalize H in the current subspace → eigenvector ψ_k.
        2. Compute per-orbital marginals (n_α(p), n_β(p)) from ψ_k.
        3. Re-process ORIGINALLY-DROPPED bitstrings using marginals as
           the preference signal. Recovered dets join the subspace.
        4. If no new dets added OR |E_k − E_{k-1}| < energy_tol, stop.

    Returns:
        ``(recovered_bitstrings, history_list)`` where history is a list of
        dicts with ``round``, ``energy``, ``n_dets``, ``n_valid_total``,
        ``n_added_this_round``.
    """
    n_alpha_target = (n_electrons + int(round(2 * target_sz))) // 2
    n_beta_target = n_electrons - n_alpha_target

    # Round 0: standard energy-guided greedy recovery
    valid_0, n_kept, n_rec_0, n_dropped_0 = _filter_with_recovery(
        bitstrings, n_orbitals, n_electrons, target_sz, mo_energies,
    )

    if not record_history:
        return valid_0, []

    history = []
    current_valid = list(int(d) for d in valid_0)
    current_dets = sorted(set(current_valid))
    last_energy = None

    # Build the originally-dropped pool (those that round 0 couldn't recover OR
    # that we want to try again with better signals)
    # For simplicity: re-process EVERY ORIGINAL bitstring with the marginal
    # preference; the keep-as-is path still keeps them; only the recovery
    # direction may change.
    for round_idx in range(max_rounds):
        if not current_dets:
            break
        # Diagonalize current subspace
        try:
            res = diagonalize_callback(current_dets)
        except Exception as exc:
            logger.warning(f"iterative recovery: diag round {round_idx} failed: {exc}")
            break
        energy = float(res['energy'])
        evec = res.get('eigenvector')
        if evec is None or len(evec) == 0:
            break
        history.append({
            'round': round_idx, 'energy': energy,
            'n_dets': len(current_dets), 'n_valid_total': len(current_valid),
        })
        if last_energy is not None and abs(energy - last_energy) < energy_tol:
            logger.info(
                f"iterative recovery converged at round {round_idx}: "
                f"|ΔE| = {abs(energy - last_energy):.2e}"
            )
            break
        last_energy = energy

        # Round k+1 prep: compute marginals
        n_a_marginal, n_b_marginal = _compute_orbital_marginals(
            current_dets, evec, n_orbitals,
        )

        # Re-process ORIGINAL bitstrings with marginal-guided recovery
        new_valid, _, _, _ = _filter_with_recovery(
            bitstrings, n_orbitals, n_electrons, target_sz,
            mo_energies=mo_energies,
            alpha_pref=n_a_marginal, beta_pref=n_b_marginal,
        )
        # Merge: dedupe with existing
        prev_set = set(current_dets)
        added = [int(d) for d in new_valid if int(d) not in prev_set]
        if not added:
            logger.info(f"iterative recovery: no new dets at round {round_idx+1}, stop")
            break
        current_valid.extend(added)
        current_dets = sorted(prev_set | set(added))
        # Record this round's additions on the entry for THIS round (after merge),
        # not on the next round's entry (which would mislabel the reference).
        history[-1]['n_added_this_round'] = len(added)

    return _det_arr(current_valid), history
