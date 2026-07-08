"""Quantum NMR magnetic shielding via the Ramsey sum-over-states.

σ_A = σ_A^dia + σ_A^para  (isotropic σ = Tr/3, in ppm)

- **Diamagnetic** (ground-state expectation of the correlated 1-RDM):
      σ^dia = ⟨0| (int1e_cg_a11part + int1e_a01gp) |0⟩         (× ALPHA²·1e6)
  contracted with the AO 1-RDM. With an HF density this reproduces PySCF's dia term to
  numerical precision; with the VQE/SQD (correlated) density it is the correlated dia.

- **Paramagnetic** (Ramsey sum-over-states — the term v1 of the protocol punted on):
      σ^para_uv = 2 · Σ_pq δγ^(u)_qp · (PSO_A,v)_pq                 (× ALPHA²·1e6)
  where δγ^(u) is the FIRST-ORDER 1-RDM response to the orbital-Zeeman perturbation
  h10_u = −½·(int1e_cg_irxp)_u, built from the excited states via
      δγ^(u)_pq = Σ_{n≠0} w_n^(u) (γ^{0n}_pq + γ^{n0}_pq)/(E_0 − E_n),
      w_n^(u) = Σ_rs (h10_u)_rs γ^{0n}_rs,
  with γ^{0n} the ground→excited transition 1-RDMs. This is the SOS form of the linear
  response that CPHF solves at the HF level — here evaluated with correlated (CASCI/FCI)
  states, so it is the correlated paramagnetic shielding.

Gauge: common origin at the nucleus of interest (GIAO-free). Absolute values are therefore
gauge-dependent in a finite basis; the correlated *shift* vs HF is the wavefunction signal.

The reference implementation below drives the states with PySCF FCI (== the framework's
CASCI in the full space) so the whole pipeline is validatable end to end; ``trans_rdms`` /
``e_states`` can equally come from the framework's excited-state solvers.
"""
from __future__ import annotations

import numpy as np

from kanad.core.integrals.property_integrals import compute_angular_momentum, compute_pso

_ALPHA = 1.0 / 137.035999084
_UNIT_PPM = _ALPHA ** 2 * 1e6


def _to_mo(ao_op, C):
    """(3,nao,nao) or (nao,nao) AO operator → MO basis via C."""
    a = np.asarray(ao_op)
    if a.ndim == 3:
        return np.einsum('xpq,pi,qj->xij', a, C, C)
    return C.T @ a @ C


def diamagnetic_shielding(mol, dm_ao, nucleus_index, gauge_origin):
    """σ^dia (3×3, ppm) for a nucleus from the AO 1-RDM ``dm_ao``.

    Common-gauge form, mirroring ``pyscf.prop.nmr.rhf.dia`` exactly: contract the
    ``int1e_cg_a11part`` integral with the density, then apply the traceless gauge
    correction ``e11 − I·Tr(e11)``. (The ``int1e_a01gp`` term belongs to the GIAO
    formulation only, not the common gauge.)
    """
    mol.set_common_origin(gauge_origin)
    with mol.with_rinv_origin(mol.atom_coord(nucleus_index)):
        h11 = mol.intor('int1e_cg_a11part', comp=9)
        e11 = np.einsum('xij,ij->x', h11, dm_ao).reshape(3, 3)
    e11 = e11 - np.eye(3) * e11.trace()
    return e11 * _UNIT_PPM


def paramagnetic_shielding_sos(mol, C, e0, e_states, trans_rdms_mo, nucleus_index,
                               gauge_origin):
    """σ^para (3×3, ppm) via the Ramsey SOS.

    Args:
        C: MO coefficients (nao, nmo).
        e0: ground-state energy (Ha).
        e_states: excited-state energies (Ha), ascending, EXCLUDING the ground state.
        trans_rdms_mo: list of ground→excited transition 1-RDMs γ^{0n} (nmo, nmo), MO basis,
            one per entry of ``e_states``.
        nucleus_index, gauge_origin: nucleus and common gauge origin (Bohr).
    """
    # Orbital-Zeeman perturbation h10_u = −½ (r_O × p)_u  (MO basis).
    h10_mo = -0.5 * _to_mo(compute_angular_momentum(mol, origin=gauge_origin), C)   # (3,nmo,nmo)
    pso_mo = _to_mo(compute_pso(mol, mol.atom_coord(nucleus_index)), C)             # (3,nmo,nmo)

    sigma = np.zeros((3, 3))
    for En, g0n in zip(e_states, trans_rdms_mo):
        dE = float(En) - float(e0)
        if dE < 1e-8:
            continue
        g0n = np.asarray(g0n)
        # The orbital-Zeeman perturbation h10 is ANTI-Hermitian (it represents i·L), so the
        # first-order 1-RDM response is the ANTI-symmetrized transition RDM γ^{n0}−γ^{0n} =
        # γ^{0n}ᵀ − γ^{0n} (NOT symmetrized — that would contract to zero against the
        # antisymmetric PSO). δγ^(u)_pq = w_u (γ^{0n}ᵀ − γ^{0n})_pq / (E0 − En).
        gdiff = g0n.T - g0n
        w = np.einsum('urs,rs->u', h10_mo, g0n)          # w_n^(u) = ⟨0|h10_u|n⟩
        contr = np.einsum('qp,vpq->v', gdiff, pso_mo)      # (3,)
        # Prefactor −1 (NOT +2): L and PSO are anti-Hermitian (i·L, i·(L/r³)) so their
        # product carries i² = −1 that the real integrals drop; and the antisymmetrized
        # response gdiff = γ^{n0}−γ^{0n} already contains both the term and its complex
        # conjugate, so PySCF's ×2-for-c.c. would double-count here.
        sigma += -1.0 * np.outer(w, contr) / (float(e0) - En)
    return sigma * _UNIT_PPM


def nmr_shielding_fci(mol, nucleus_index, *, n_states=25, gauge_origin=None):
    """Isotropic NMR shielding (ppm) for ``nucleus_index`` via FCI states + the Ramsey SOS.

    FCI == the framework's CASCI in the full active space, so this is the correlated shielding.
    Gauge origin defaults to the nucleus itself (common gauge at the nucleus).
    Returns a dict: ``sigma_iso``, ``sigma_dia``, ``sigma_para``, ``tensor``.
    """
    from pyscf import scf, fci
    mf = scf.RHF(mol).run(verbose=0)
    C = mf.mo_coeff
    norb = C.shape[1]
    nelec = mol.nelec
    if gauge_origin is None:
        gauge_origin = mol.atom_coord(nucleus_index)

    cis = fci.FCI(mf)
    nroots = min(n_states, _fci_dim(norb, nelec))
    e_all, civecs = cis.kernel(nroots=nroots)
    e_all = np.atleast_1d(e_all)
    e0, c0 = e_all[0], civecs[0]
    dm1_mo = cis.make_rdm1(c0, norb, nelec)
    dm1_ao = C @ dm1_mo @ C.T

    trans = [cis.trans_rdm1(c0, civecs[n], norb, nelec) for n in range(1, len(e_all))]
    sig_dia = diamagnetic_shielding(mol, dm1_ao, nucleus_index, gauge_origin)
    sig_para = paramagnetic_shielding_sos(mol, C, e0, e_all[1:], trans, nucleus_index, gauge_origin)
    tensor = sig_dia + sig_para
    return {
        'sigma_iso': float(np.trace(tensor) / 3.0),
        'sigma_dia': float(np.trace(sig_dia) / 3.0),
        'sigma_para': float(np.trace(sig_para) / 3.0),
        'tensor': tensor,
        'n_states': len(e_all),
    }


def _fci_dim(norb, nelec):
    from math import comb
    na, nb = (nelec if isinstance(nelec, (tuple, list)) else (nelec // 2 + nelec % 2, nelec // 2))
    return comb(norb, na) * comb(norb, nb)
