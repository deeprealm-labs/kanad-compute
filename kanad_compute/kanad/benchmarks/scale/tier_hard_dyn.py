"""Tier-HARD DYNAMICS audit — probes the dynamics module against code-grounded
failure predictions. Each probe reports predicted-vs-observed and a verdict:
    CONFIRMED   the predicted failure/limit reproduced
    REFUTED     it actually works (prediction wrong)
    PARTIAL     works with caveats
    crash       unexpected exception (could be real failure or probe-setup)

Cluster only:
    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier_hard_dyn
"""
from __future__ import annotations
import traceback
import numpy as np

ANG2BOHR = 1.8897259886


def _safe(fn):
    try:
        return fn(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:140]}"


def probe_CO_nac_ceiling():
    """CO sto-3g full-FCI = 14400 dets > 4096 -> HF-NAC must raise; overlap-NAC ground-excited ~0."""
    from kanad.dynamics.nonadiabatic import NonAdiabaticMD
    from kanad.core.bonds.bond_factory import BondFactory
    bond = BondFactory.create_bond('C', 'O', distance=1.25)
    md = NonAdiabaticMD(bond, n_states=3, solver_method='cis')
    pos = np.array([[0, 0, 0], [0, 0, 1.25 * ANG2BOHR]])
    hf, hf_err = _safe(lambda: md.compute_nonadiabatic_coupling(pos, 0, 1, method='hellmann_feynman'))
    ov, ov_err = _safe(lambda: md.compute_nonadiabatic_coupling(pos, 0, 1, method='overlap'))
    hf_raised = hf_err is not None and 'NotImplemented' in (hf_err or '')
    ov_mag = float(np.linalg.norm(ov)) if ov is not None else None
    ov_near_zero = ov_mag is not None and ov_mag < 1e-3
    verdict = 'CONFIRMED' if (hf_raised and ov_near_zero) else ('PARTIAL' if (hf_raised or ov_near_zero) else 'REFUTED')
    return dict(name='CO_NAC_ceiling', predicted='HF-NAC raises (>4096 dets); overlap d01~0 for ground-excited',
                observed=f"hf_raised={hf_raised} ({hf_err}); overlap|d01|={ov_mag}", verdict=verdict)


def probe_BH_nac_edge():
    """BH sto-3g 400 dets < 4096 -> HF-NAC should RUN; check antisymmetry + sign across R."""
    from kanad.dynamics.nonadiabatic import NonAdiabaticMD
    from kanad.core.bonds.bond_factory import BondFactory
    runs, anti_ok, mags = 0, True, []
    for R in (1.23, 1.40, 1.80, 2.20):
        try:
            bond = BondFactory.create_bond('B', 'H', distance=R)
            md = NonAdiabaticMD(bond, n_states=3, solver_method='cis')
            pos = np.array([[0, 0, 0], [0, 0, R * ANG2BOHR]])
            d01 = md.compute_nonadiabatic_coupling(pos, 0, 1, method='hellmann_feynman')
            d10 = md.compute_nonadiabatic_coupling(pos, 1, 0, method='hellmann_feynman')
            runs += 1
            if np.linalg.norm(np.asarray(d01) + np.asarray(d10)) > 1e-6 * (np.linalg.norm(d01) + 1e-9):
                anti_ok = False
            mags.append(round(float(np.linalg.norm(d01)), 4))
        except Exception as e:
            mags.append(f"ERR:{type(e).__name__}")
    verdict = 'REFUTED' if (runs == 4 and anti_ok) else ('PARTIAL' if runs > 0 else 'CONFIRMED')
    return dict(name='BH_NAC_edge', predicted='HF-NAC runs (400<4096); possible Pi-degeneracy sign-flip / unequal-mass ETF residual',
                observed=f"ran {runs}/4 R-points, antisym_ok={anti_ok}, |d01(R)|={mags}", verdict=verdict)


def probe_diazene_diatomic_limit():
    """NonAdiabaticMD.atoms=[bond.atom_1,bond.atom_2] (line 220) -> cannot represent 4-atom diazene."""
    from kanad.dynamics.nonadiabatic import NonAdiabaticMD
    from kanad.core.bonds.bond_factory import BondFactory
    bond = BondFactory.create_bond('N', 'N', distance=1.232)
    md = NonAdiabaticMD(bond, n_states=2, solver_method='cis')
    n_atoms = len(md.atoms)
    verdict = 'CONFIRMED' if n_atoms == 2 else 'REFUTED'
    return dict(name='diazene_diatomic_limit', predicted='NonAdiabaticMD hardwired to 2 atoms; 4-atom photoisomerization impossible',
                observed=f"md.atoms has {n_atoms} atoms (a diazene needs 4; torsion coordinate not representable)", verdict=verdict)


def probe_formaldehyde_cis_eq_tddft():
    """_solve_tddft (line 346) returns _solve_cis() -> 'tddft' must equal 'cis' exactly."""
    from kanad.solvers.excited_states_solver import ExcitedStatesSolver
    from kanad.core.bonds.bond_factory import BondFactory
    # CO as a stand-in chromophore bond for the solver (the equality is method-level, system-independent)
    bond = BondFactory.create_bond('C', 'O', distance=1.208)
    def get(m):
        s = ExcitedStatesSolver(bond, n_states=3, method=m).solve().to_dict()
        return s.get('excitation_energies_ev', s.get('energies'))
    cis, cis_err = _safe(lambda: get('cis'))
    tddft, td_err = _safe(lambda: get('tddft'))
    identical = (cis is not None and tddft is not None and
                 np.allclose(np.asarray(cis, float), np.asarray(tddft, float), atol=1e-10))
    verdict = 'CONFIRMED' if identical else ('crash' if (cis_err or td_err) else 'REFUTED')
    return dict(name='formaldehyde_cis_eq_tddft', predicted="method='tddft' is a relabeled CIS (identical numbers) — fake long-range physics",
                observed=f"cis={cis} ({cis_err}); tddft={tddft} ({td_err}); identical={identical}", verdict=verdict)


def probe_MnH_septet_cis():
    """High-spin ROHF septet -> CIS (n_occ=n_e//2 closed-shell assumption) should break or mislabel."""
    from kanad.dynamics.nonadiabatic import NonAdiabaticMD
    from kanad.core.bonds.bond_factory import BondFactory
    bond, berr = _safe(lambda: BondFactory.create_bond('Mn', 'H', distance=1.731))
    if bond is None:
        return dict(name='MnH_septet_cis', predicted='CIS breaks on high-spin ROHF septet',
                    observed=f"could not build MnH bond: {berr}", verdict='crash')
    md, merr = _safe(lambda: NonAdiabaticMD(bond, n_states=3, solver_method='cis', initial_state=0))
    if md is None:
        return dict(name='MnH_septet_cis', predicted='CIS breaks on high-spin ROHF septet',
                    observed=f"NonAdiabaticMD ctor: {merr}", verdict='CONFIRMED')
    pos = np.array([[0, 0, 0], [0, 0, 1.731 * ANG2BOHR]])
    e, eerr = _safe(lambda: md.compute_state_energies(pos))
    if e is None:
        return dict(name='MnH_septet_cis', predicted='CIS breaks on high-spin ROHF septet',
                    observed=f"compute_state_energies raised: {eerr}", verdict='CONFIRMED')
    return dict(name='MnH_septet_cis', predicted='CIS breaks on high-spin ROHF septet',
                observed=f"state energies returned: {np.round(np.asarray(e,float),4).tolist()} (check physicality)", verdict='PARTIAL')


def probe_lindblad_dephasing_convention():
    """create_dephasing_operator returns bare sigma_z (line 474) not sigma_z/2 -> T2* off by 2x; plus CPTP check."""
    from kanad.dynamics.open_quantum.lindblad import (
        LindbladEvolver, create_amplitude_damping_operator, create_dephasing_operator)
    Z = create_dephasing_operator(n_qubits=1, qubit_idx=0)
    is_bare = np.allclose(Z, np.array([[1, 0], [0, -1]], complex))
    is_half = np.allclose(Z, 0.5 * np.array([[1, 0], [0, -1]], complex))
    # short evolution CPTP/trace check
    omega = 3.9 / 27.211
    H0 = np.diag([0.0, omega]).astype(complex)
    cptp = None
    try:
        L = create_amplitude_damping_operator(1, 0)
        ev = LindbladEvolver(hamiltonian=H0, lindblad_ops=[0.01 * L, 0.01 * Z])
        rho0 = np.array([[0, 0], [0, 1]], complex)  # start excited
        out = ev.evolve(rho0, t_final=50.0, n_steps=100) if hasattr(ev, 'evolve') else None
        if out is not None:
            rho = out[-1] if isinstance(out, (list, np.ndarray)) and np.ndim(out) == 3 else out
            rho = np.asarray(rho)[-1] if np.ndim(rho) == 3 else np.asarray(rho)
            tr = float(np.real(np.trace(rho)))
            eig = np.linalg.eigvalsh(rho)
            cptp = abs(tr - 1) < 1e-3 and eig.min() > -1e-6
    except Exception as e:
        cptp = f"evolve-err:{type(e).__name__}:{str(e)[:50]}"
    verdict = 'CONFIRMED' if is_bare else ('REFUTED' if is_half else 'PARTIAL')
    return dict(name='lindblad_dephasing_convention', predicted='dephasing op is bare sigma_z (not sigma_z/2) -> inferred T2* off by 2x',
                observed=f"bare_sigma_z={is_bare}, half={is_half}, CPTP_short_evolve={cptp}", verdict=verdict)


def probe_malonaldehyde_reaction():
    """QuantumReactionSimulator IRC/TS on the symmetric proton-transfer double well (best-effort API)."""
    try:
        import kanad.reactions as rx
        names = [n for n in dir(rx) if 'eact' in n or 'imulator' in n]
    except Exception as e:
        return dict(name='malonaldehyde_reaction', predicted='active-space discontinuity at symmetric TS; IRC/TS fragile',
                    observed=f"reactions import failed: {type(e).__name__}: {str(e)[:80]}", verdict='crash')
    return dict(name='malonaldehyde_reaction', predicted='active-space discontinuity at symmetric TS; IRC/TS fragile',
                observed=f"reactions module exposes: {names[:6]} (API surface probe — full IRC deferred to cluster run)", verdict='PARTIAL')


PROBES = [probe_CO_nac_ceiling, probe_BH_nac_edge, probe_diazene_diatomic_limit,
          probe_formaldehyde_cis_eq_tddft, probe_MnH_septet_cis,
          probe_lindblad_dephasing_convention, probe_malonaldehyde_reaction]


def main():
    print("=" * 100, flush=True)
    print(f"TIER-HARD DYNAMICS audit — {len(PROBES)} code-grounded failure probes", flush=True)
    print("=" * 100, flush=True)
    for p in PROBES:
        try:
            r = p()
        except Exception as e:
            r = dict(name=p.__name__, predicted='?', observed=f"PROBE_CRASH {type(e).__name__}: {str(e)[:100]}",
                     verdict='crash', trace=traceback.format_exc().splitlines()[-2:])
        print(f"\nDYN| {r['name']}  [{r['verdict']}]", flush=True)
        print(f"   predicted: {r['predicted']}", flush=True)
        print(f"   observed : {r['observed']}", flush=True)
        if r.get('trace'):
            print(f"   trace: {r['trace']}", flush=True)
    print("\nDYN_DONE", flush=True)


if __name__ == "__main__":
    main()
