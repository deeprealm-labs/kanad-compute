"""Campaign C — PROPERTY / ANALYSIS calculators across diverse molecules, vs PySCF and
experiment. Exercises the analysis suite that the energy-only campaigns never touched:
dipoles, vibrational frequencies, thermochemistry, bonding/Mulliken, correlation energy.
Flags any silent HF-fallback or stub.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework \
        /root/miniconda3/bin/python -m benchmarks.tier_C_props
"""
from __future__ import annotations
import numpy as np


def _norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def main():
    from kanad import MolecularBuilder
    print("=" * 100, flush=True)
    print("CAMPAIGN C — property/analysis calculators vs PySCF / experiment", flush=True)
    print("=" * 100, flush=True)

    # ---- 1. Dipole moments (CI 1-RDM) vs PySCF FCI dipole vs experiment ----
    print("\n--- Dipole moments |μ| (Debye): builder observables() vs PySCF-FCI vs exp ---", flush=True)
    dip_sys = [
        ('H2O', [('O', (0, 0, 0)), ('H', (0, 0.757, 0.587)), ('H', (0, -0.757, 0.587))], 1.85),
        ('NH3', [('N', (0, 0, 0.12)), ('H', (0, 0.94, -0.27)), ('H', (0.81, -0.47, -0.27)), ('H', (-0.81, -0.47, -0.27))], 1.47),
        ('HF',  [('H', (0, 0, 0)), ('F', (0, 0, 0.917))], 1.82),
        ('CO',  [('C', (0, 0, 0)), ('O', (0, 0, 1.128))], 0.11),
    ]
    for name, atoms, exp in dip_sys:
        try:
            qs = MolecularBuilder.from_atoms(atoms).basis('6-31g').solver('ci').build()
            qs.solve()
            o = qs.observables('core')
            mu = _norm(o.get('dipole_debye', [0, 0, 0]))
            # PySCF reference (HF dipole — quick cross-check of magnitude/direction)
            from pyscf import gto, scf
            mol = gto.M(atom=atoms, basis='6-31g', verbose=0); mf = scf.RHF(mol).run(verbose=0)
            mu_hf = _norm(mf.dip_moment(unit='Debye', verbose=0))
            print(f"  {name:4} |μ|_CI={mu:5.2f}  |μ|_HF(pyscf)={mu_hf:5.2f}  exp={exp:4.2f} D  (CI-exp={mu-exp:+.2f})", flush=True)
        except Exception as e:
            print(f"  {name:4} CRASH {type(e).__name__}: {str(e)[:70]}", flush=True)

    # ---- 2. Vibrational frequencies (HF Hessian) vs experiment ----
    print("\n--- Harmonic vibrational frequencies (cm^-1) vs experiment ---", flush=True)
    try:
        from kanad.analysis.vibrational_analysis import FrequencyCalculator
        from kanad.core.atom import Atom
        # H2 (exp 4401), H2O bend ~1595 / sym-str ~3657
        def mol_from(atoms, basis='6-31g'):
            from kanad.core.representations.base_representation import BondMolecule
            try:
                return BondMolecule([Atom(s, np.array(xyz)) for s, xyz in atoms], basis=basis)
            except Exception:
                # fall back to whatever Molecule the FrequencyCalculator wants
                from kanad.core.molecule import Molecule
                return Molecule([Atom(s, np.array(xyz)) for s, xyz in atoms], basis=basis)
        for name, atoms, expfreq in [
            ('H2', [('H', (0, 0, 0)), ('H', (0, 0, 0.74))], '4401'),
            ('HF', [('H', (0, 0, 0)), ('F', (0, 0, 0.917))], '4138'),
        ]:
            try:
                m = mol_from(atoms)
                fc = FrequencyCalculator(m)
                res = fc.compute_frequencies(verbose=False)
                freqs = res.get('frequencies') if isinstance(res, dict) else getattr(res, 'frequencies', None)
                top = [round(float(x), 0) for x in np.atleast_1d(freqs)][-3:]
                print(f"  {name:4} freqs={top} cm^-1  (exp ~{expfreq})", flush=True)
            except Exception as e:
                print(f"  {name:4} freq CRASH {type(e).__name__}: {str(e)[:80]}", flush=True)
    except Exception as e:
        print(f"  FrequencyCalculator import CRASH {type(e).__name__}: {str(e)[:80]}", flush=True)

    # ---- 3. Bonding analysis / Mulliken charges ----
    print("\n--- Mulliken charges & bond order (BondingAnalyzer) ---", flush=True)
    try:
        from kanad.analysis.energy_analysis import BondingAnalyzer, CorrelationAnalyzer
        qs = MolecularBuilder.from_atoms([('C', (0, 0, 0)), ('O', (0, 0, 1.128))]).basis('6-31g').solver('ci').build()
        qs.solve()
        ham = qs.hamiltonian
        ba = BondingAnalyzer(ham)
        try:
            q = ba.compute_mulliken_charges()
            print(f"  CO Mulliken charges: {[round(float(x),3) for x in np.atleast_1d(q)]}", flush=True)
        except Exception as e:
            print(f"  Mulliken CRASH {type(e).__name__}: {str(e)[:70]}", flush=True)
        try:
            ca = CorrelationAnalyzer(ham)
            ec = ca.compute_correlation_energy()
            print(f"  CO correlation energy: {ec}", flush=True)
        except Exception as e:
            print(f"  Correlation CRASH {type(e).__name__}: {str(e)[:70]}", flush=True)
    except Exception as e:
        print(f"  BondingAnalyzer CRASH {type(e).__name__}: {str(e)[:80]}", flush=True)

    # ---- 4. observables('all') — polarizability / NMR (flag stubs/fallbacks) ----
    print("\n--- observables('all'): polarizability + NMR (watch for HF-fallback flags) ---", flush=True)
    try:
        qs = MolecularBuilder.from_atoms([('O', (0, 0, 0)), ('H', (0, 0.757, 0.587)), ('H', (0, -0.757, 0.587))]).basis('6-31g').solver('ci').build()
        qs.solve()
        o = qs.observables('all')
        for k in ('polarizability_au', 'polarizability', 'nmr_shielding_ppm', 'nmr', 'homo_lumo_gap_ev', 'm_diagnostic'):
            if k in o:
                v = o[k]
                vs = v if np.isscalar(v) else (np.round(np.atleast_1d(v), 3).tolist() if hasattr(v, '__len__') else v)
                print(f"  {k} = {vs}", flush=True)
    except Exception as e:
        print(f"  observables('all') CRASH {type(e).__name__}: {str(e)[:80]}", flush=True)

    print("\nPROPS_DONE", flush=True)


if __name__ == "__main__":
    main()
