"""Tier-HARD Round 2 — spin-state energetics, unlocked by the SQD S² fix.

The spin_s fix lets SQD target a multiplicity, so we can now compute SINGLET-TRIPLET
GAPS — run the singlet (spin_s=0) and triplet (spin_s=1) on the SAME active orbitals
and difference them. References use fci.direct_spin0 (singlet) and direct_spin1
(triplet) — NOT the unreliable fix_spin_ that mis-converged on near-degenerate cases.

Also re-confirms the fixes (F2-stretched, twisted-ethylene) against RELIABLE references.

Cluster only:
    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier_hard2
"""
from __future__ import annotations
import time, traceback
import numpy as np

HA2KCAL = 627.509


def _atoms(geom):
    out = []
    for p in geom.strip().strip(';').split(';'):
        t = p.split()
        if len(t) >= 4:
            out.append((t[0], (float(t[1]), float(t[2]), float(t[3]))))
    return out


def cas_refs(geom, basis, ncas, ne):
    """Reliable singlet (direct_spin0) + lowest-triplet (direct_spin1) in the CAS
    built from the closed-shell RHF orbitals. Returns (E_singlet, E_triplet)."""
    from pyscf import gto, scf, mcscf, ao2mo, fci
    mol = gto.M(atom=geom, basis=basis, spin=0, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    cas = mcscf.CASCI(mf, ncas, ne)
    h1, ecore = cas.get_h1eff()
    h2 = ao2mo.restore(1, cas.get_h2eff(), ncas)
    e_s, v_s = fci.direct_spin0.kernel(h1, h2, ncas, ne, ecore=ecore)
    # triplet: (na, nb) = (ne/2+1, ne/2-1); take lowest root with S^2 ~ 2
    na, nb = ne // 2 + 1, ne // 2 - 1
    es, vs = fci.direct_spin1.kernel(h1, h2, ncas, (na, nb), ecore=ecore, nroots=3)
    if np.ndim(es) == 0:
        es, vs = [es], [vs]
    e_t = None
    for e, v in zip(es, vs):
        ss, _ = fci.spin_op.spin_square(v, ncas, (na, nb))
        if abs(ss - 2.0) < 0.3:
            e_t = float(e); break
    if e_t is None:
        e_t = float(es[0])
    return float(e_s), e_t


def sqd_energy(atoms, basis, n_occ, n_virt, spin, spin_s, n_samples=80000, charge=0):
    from kanad import MolecularBuilder
    b = MolecularBuilder.from_atoms(atoms).basis(basis)
    if charge:
        b = b.charge(charge)
    if spin:
        b = b.spin(spin)
    b = b.active_space('frontier', n_occ=n_occ, n_virt=n_virt)
    qs = b.solver('sqd', n_samples=n_samples, max_iterations=5, energy_tol=1e-6,
                  random_seed=0, spin_s=spin_s).build()
    r = qs.solve()
    return float(r['energy']), r.get('s_squared')


def st_gap(name, geom, basis, n_occ, n_virt, ne, ncas, exp_kcal=None, n_samples=80000):
    """Singlet-triplet gap: SQD(spin_s=0) and SQD(spin_s=1) vs direct_spin0/1 refs."""
    r = {'name': name}
    t0 = time.time()
    try:
        e_s_ref, e_t_ref = cas_refs(geom, basis, ncas, ne)
        ref_gap = (e_t_ref - e_s_ref) * HA2KCAL
        atoms = _atoms(geom)
        # Both multiplicities come from ONE closed-shell (spin=0) build: sample the
        # M_s=0 subspace, then the S² penalty extracts the singlet (spin_s=0) and the
        # M_s=0 triplet component (spin_s=1) from the SAME orbitals/subspace. This
        # sidesteps the open-shell active-space wall (a spin=2 build can't freeze core)
        # and guarantees the gap compares identical active spaces.
        e_s, ss_s = sqd_energy(atoms, basis, n_occ, n_virt, spin=0, spin_s=0.0, n_samples=n_samples)
        e_t, ss_t = sqd_energy(atoms, basis, n_occ, n_virt, spin=0, spin_s=1.0, n_samples=n_samples)
        sqd_gap = (e_t - e_s) * HA2KCAL
        r.update(dict(status='ok', e_s=e_s, e_t=e_t, ss_s=ss_s, ss_t=ss_t,
                      sqd_gap_kcal=sqd_gap, ref_gap_kcal=ref_gap,
                      gap_err_kcal=sqd_gap - ref_gap, exp_kcal=exp_kcal,
                      s_err_mha=(e_s - e_s_ref) * 1000, t_err_mha=(e_t - e_t_ref) * 1000))
    except Exception as e:
        r.update(dict(status='crash', error='%s: %s' % (type(e).__name__, str(e)[:130]),
                      trace=traceback.format_exc().splitlines()[-2:]))
    r['t'] = round(time.time() - t0, 1)
    return r


def confirm_singlet(name, geom, basis, n_occ, n_virt, ne, ncas, n_samples=100000):
    """Re-confirm a flagged 'variational violation' against the RELIABLE direct_spin0
    singlet ground (spin_s=0 should match it, NOT fall below)."""
    r = {'name': name}
    t0 = time.time()
    try:
        e_s_ref, _ = cas_refs(geom, basis, ncas, ne)
        e_s, ss_s = sqd_energy(_atoms(geom), basis, n_occ, n_virt, spin=0, spin_s=0.0, n_samples=n_samples)
        r.update(dict(status='ok', e_s=e_s, ss_s=ss_s, e_ref=e_s_ref,
                      gap_mha=(e_s - e_s_ref) * 1000))
    except Exception as e:
        r.update(dict(status='crash', error='%s: %s' % (type(e).__name__, str(e)[:130])))
    r['t'] = round(time.time() - t0, 1)
    return r


# Singlet-triplet gap experiments (exp gaps: + = triplet above singlet)
ST = [
    # methylene CH2: TRIPLET ground, S-T = +9.0 kcal/mol (singlet above). re geometry.
    ('CH2_methylene', 'C 0 0 0; H 0 0.99 0.59; H 0 -0.99 0.59', 'sto-3g', 3, 3, 6, 6, -9.0),
    # NH imidogen: TRIPLET ground, S-T ~ +35 kcal/mol
    ('NH_imidogen', 'N 0 0 0; H 0 0 1.036', 'sto-3g', 3, 3, 6, 6, -35.9),
    # O2: TRIPLET ground (the classic), singlet 1-delta-g ~ +22.5 kcal/mol above
    ('O2_dioxygen', 'O 0 0 0; O 0 0 1.208', 'sto-3g', 4, 4, 8, 8, -22.5),
    # trimethylenemethane TMM: TRIPLET ground diradical, S-T ~ +16 kcal/mol
    ('TMM_trimethylenemethane', 'C 0 0 0; C 0 1.40 0; C 1.21 -0.70 0; C -1.21 -0.70 0; '
     'H 0.93 1.94 0; H -0.93 1.94 0; H 2.16 -0.24 0; H 1.21 -1.79 0; H -2.16 -0.24 0; H -1.21 -1.79 0',
     'sto-3g', 3, 3, 6, 6, -16.0),
    # silylene SiH2: SINGLET ground (opposite of CH2!), S-T ~ -21 kcal/mol (singlet below)
    ('SiH2_silylene', 'Si 0 0 0; H 0 1.10 0.92; H 0 -1.10 0.92', 'sto-3g', 3, 3, 6, 6, +21.0),
]

# Fix-confirmations against reliable references
CONFIRM = [
    ('F2_stretched_spin0_recheck', 'F 0 0 0; F 0 0 2.8238', 'cc-pvdz', 7, 7, 14, 14),
    ('twisted_ethylene_spin0_recheck', 'C 0 0 0.667; C 0 0 -0.667; H 0.920 0 1.230; '
     'H -0.920 0 1.230; H 0 0.920 -1.230; H 0 -0.920 -1.230', 'sto-3g', 4, 4, 8, 8),
]


def main():
    print("=" * 110, flush=True)
    print("TIER-HARD ROUND 2 — spin-state energetics (S-T gaps via spin_s) + fix confirmations", flush=True)
    print("=" * 110, flush=True)
    print("\n--- Singlet-triplet gaps (SQD spin_s=0 vs spin_s=1; refs = direct_spin0/spin1) ---", flush=True)
    for args in ST:
        r = st_gap(*args)
        if r['status'] == 'ok':
            print("ST| %-28s | SQD gap=%+.2f kcal | ref=%+.2f | err=%+.2f | exp=%s | "
                  "S²(s/t)=%.3f/%.3f | E_err(s/t)=%+.2f/%+.2f mHa | t=%.0fs"
                  % (r['name'], r['sqd_gap_kcal'], r['ref_gap_kcal'], r['gap_err_kcal'],
                     r['exp_kcal'], r['ss_s'] or -1, r['ss_t'] or -1, r['s_err_mha'],
                     r['t_err_mha'], r['t']), flush=True)
        else:
            print("ST| %-28s | CRASH %s" % (r['name'], r.get('error')), flush=True)
    print("\n--- Fix confirmations (spin_s=0 vs reliable direct_spin0 singlet) ---", flush=True)
    for args in CONFIRM:
        r = confirm_singlet(*args)
        if r['status'] == 'ok':
            print("CONFIRM| %-30s | SQD(spin_s=0)=%.6f | direct_spin0=%.6f | gap=%+.3f mHa | "
                  "S²=%.2e | t=%.0fs"
                  % (r['name'], r['e_s'], r['e_ref'], r['gap_mha'], r['ss_s'] or 0, r['t']), flush=True)
        else:
            print("CONFIRM| %-30s | CRASH %s" % (r['name'], r.get('error')), flush=True)
    print("\nHARD2_DONE", flush=True)


if __name__ == "__main__":
    main()
