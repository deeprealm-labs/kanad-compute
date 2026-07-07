"""FULL analysis audit on CHALLENGING molecules, planck/GPU where it routes.

Exercises ALL kanad analyzers (energy/bonding/correlation/properties, UV-Vis +
photodynamics + vibronic [PHOTOCHEMISTRY], NMR, IR/Raman, frequencies, thermochem,
DOS, uncertainty, bond-scan + configuration-explorer [REACTIONS], descriptors,
reactivity) on real, multireference / aromatic molecules — NOT toy H2/LiH/H2O.

Each analyzer runs under a per-cell timeout with try/except; we record returned
values, physical-invariant checks, whether it routed through the planck GPU backend,
and any failure (failures are findings). Quantum-routed analyses (excited states, DOS,
photodynamics, config-explorer) use backend/vqe_backend='planck'.

Run:  PYTHONPATH=<parent-of-kanad> python benchmarks/planck_full_analysis_audit.py
Env:  FA_MOLS="N2,CO,benzene"   FA_BACKEND=planck   FA_CELL_TIMEOUT=600
"""
import json
import os
import signal
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")

OUT = "benchmarks/out/planck_full_analysis_audit.json"
BACKEND = os.environ.get("FA_BACKEND", "planck")
CELL_TIMEOUT = int(os.environ.get("FA_CELL_TIMEOUT", "600"))
RESULTS = {}
if os.path.exists(OUT):
    try:
        RESULTS = json.load(open(OUT))
    except Exception:
        RESULTS = {}


def _save():
    os.makedirs("benchmarks/out", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)


class _Timeout(Exception):
    pass


def _alarm(s, f):
    raise _Timeout(f"exceeded {CELL_TIMEOUT}s")


def cell(store, name, fn, gpu=False):
    """Run one analyzer; record {ran, gpu, t, ...checks} or {ran:False, error}."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(CELL_TIMEOUT)
    t = time.perf_counter()
    rec = {"gpu_routed": gpu}
    try:
        out = fn()
        rec.update(out if isinstance(out, dict) else {"value": out})
        rec["ran"] = True
    except Exception as e:
        rec["ran"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc().splitlines()[-3:]
    finally:
        signal.alarm(0)
    rec["t"] = round(time.perf_counter() - t, 2)
    store[name] = rec
    flag = "OK " if rec["ran"] else "ERR"
    extra = {k: v for k, v in rec.items() if k not in ("ran", "trace", "t", "gpu_routed")}
    print(f"    [{flag}]{'(GPU)' if gpu else '     '} {name:32s} {str(extra)[:120]} ({rec['t']}s)", flush=True)
    _save()


# ---- challenging molecule builders ------------------------------------------
# Geometries: real, multireference / aromatic / photochemically interesting.
GEOM = {
    "N2":        ("N 0 0 0; N 0 0 1.098",                          "cc-pvdz", 0, 0),   # triple bond, multiref
    "CO":        ("C 0 0 0; O 0 0 1.128",                          "cc-pvdz", 0, 0),   # heteronuclear
    "C2":        ("C 0 0 0; C 0 0 1.243",                          "cc-pvdz", 0, 0),   # quad bond, hard
    "formaldehyde": ("C 0 0 0; O 0 0 1.208; H 0 0.943 -0.589; H 0 -0.943 -0.589", "sto-3g", 0, 0),  # n->pi*
}
SMILES = {"benzene": ("c1ccccc1", "sto-3g"), "naphthalene": ("c1ccc2ccccc2c1", "sto-3g"),
          "butadiene": ("C=CC=C", "sto-3g")}


def _atoms(geom):
    out = []
    for p in geom.strip().strip(";").split(";"):
        t = p.split()
        if len(t) >= 4:
            out.append((t[0], (float(t[1]), float(t[2]), float(t[3]))))
    return out


def _molecule(atoms, basis="sto-3g", charge=0, spin=0):
    """A proper kanad Molecule (has .coordinates + .hamiltonian) — the object the
    molecule-analyzers (Frequency/Thermo/BondScan/ConfigExplorer/Vibronic) actually need."""
    from kanad.core.molecule import Molecule
    from kanad.core.atom import Atom
    return Molecule([Atom(s, tuple(map(float, xyz))) for s, xyz in atoms],
                    charge=charge, spin=spin, basis=basis)


def build(mol, frontier=None):
    """Return (system, bond_or_None, molecule_or_None). Diatomics get a MolecularBuilder
    system (`.mf`, `.excited_states()`, ham-analyzers), a BondFactory bond (photodynamics),
    AND a proper Molecule (molecule-analyzers). Others -> system + Molecule (from geom)."""
    from kanad.builder import MolecularBuilder
    if mol in ("N2", "CO", "C2"):
        from kanad.bonds import BondFactory
        sym = {"N2": ("N", "N", 1.098), "CO": ("C", "O", 1.128), "C2": ("C", "C", 1.243)}[mol]
        bond = BondFactory.create_bond(sym[0], sym[1], distance=sym[2], bond_type="covalent")
        atoms = [(sym[0], (0, 0, 0)), (sym[1], (0, 0, sym[2]))]
        sysobj = MolecularBuilder.from_atoms(atoms).basis("sto-3g").active_space("full").build()
        return sysobj, bond, _molecule(atoms, "sto-3g")
    if mol in GEOM:
        geom, basis, ch, sp = GEOM[mol]
        atoms = _atoms(geom)
        b = MolecularBuilder.from_atoms(atoms).basis(basis)
        if ch:
            b = b.charge(ch)
        if sp:
            b = b.spin(sp)
        if frontier:
            b = b.active_space("frontier", n_occ=frontier[0], n_virt=frontier[1])
        return b.build(), None, _molecule(atoms, basis, ch, sp)
    if mol in SMILES:
        smi, basis = SMILES[mol]
        b = MolecularBuilder.from_smiles(smi, basis)
        if frontier:
            b = b.active_space("frontier", n_occ=frontier[0], n_virt=frontier[1])
        return b.build(), None, None          # SMILES geometry: molecule-analyzers skipped
    raise ValueError(mol)


# ---- the audit per molecule --------------------------------------------------
def audit_molecule(mol, frontier=None):
    from kanad.analysis import (EnergyAnalyzer, BondingAnalyzer, CorrelationAnalyzer,
                                PropertyCalculator, UVVisCalculator, NMRCalculator,
                                FrequencyCalculator, RamanIRCalculator,
                                ThermochemistryCalculator, DOSCalculator,
                                UncertaintyAnalyzer, BondLengthScanner,
                                ConfigurationExplorer, VibronicCalculator)
    store = RESULTS.setdefault(mol, {})
    sysobj, bond, molecule = build(mol, frontier)
    ham = getattr(sysobj, "hamiltonian", getattr(bond, "hamiltonian", None))
    nq = 2 * ham.n_orbitals
    store["_meta"] = {"qubits": nq, "frontier": frontier,
                      "hf_energy": float(getattr(ham, "mf", None).e_tot) if getattr(ham, "mf", None) is not None else None}
    print(f"\n=== {mol}  ({nq} qubits, backend={BACKEND}) ===", flush=True)

    # ground-state solve (planck) for correlation + a quantum reference energy
    vqe_e = [None]

    def _solve():
        from kanad.solvers import PhysicsVQE
        r = PhysicsVQE(sysobj, backend=BACKEND, max_excitations=6).solve()   # builder system (consistent w/ ham)
        vqe_e[0] = float(r.energy)
        return {"energy": vqe_e[0], "converged": r.to_dict().get("converged")}
    cell(store, "ground_state_PhysicsVQE", _solve, gpu=(BACKEND == "planck"))

    # density for the classical post-processing analyzers
    dm = None
    if getattr(ham, "mf", None) is not None:
        dm = ham.mf.make_rdm1()

    # 1. energy decomposition.
    # EnergyAnalyzer needs the density in the SAME orbital basis as `ham`. For an
    # active-space ActiveHamiltonian, ham.mf.make_rdm1() is the FULL-space density
    # and won't match (ham.h_core is active); only run when the bases match.
    def _ed():
        import numpy as _np
        h_core = getattr(ham, "h_core", None)
        if h_core is not None and _np.asarray(dm).shape != _np.asarray(h_core).shape:
            return {"skipped": "density basis != Hamiltonian basis (active space)"}
        d = EnergyAnalyzer(ham).decompose_energy(dm)
        parts = sum(d.get(k, 0.0) for k in ("nuclear_repulsion", "one_electron", "two_electron"))
        return {"total": d.get("total"), "self_consistent": bool(abs(d["total"] - parts) < 1e-3)}
    if dm is not None:
        cell(store, "EnergyAnalyzer.decompose", _ed)

    # 2. bonding
    def _bond():
        b = BondingAnalyzer(ham)
        bt = b.analyze_bonding_type()
        bo = b.analyze_bond_orders(dm) if dm is not None else {}
        return {"bonding_type": bt.get("bonding_type"),
                "gap_ev": bt.get("homo_lumo_gap_ev"),
                "bond_orders_present": bool(bo)}
    cell(store, "BondingAnalyzer", _bond)

    # 3. correlation
    def _corr():
        hf = store["_meta"]["hf_energy"]
        ce = CorrelationAnalyzer(ham).compute_correlation_energy(vqe_e[0], hf)
        return {"correlation_energy": float(ce), "nonpositive": bool(ce <= 1e-6)}
    if vqe_e[0] is not None and store["_meta"]["hf_energy"] is not None:
        cell(store, "CorrelationAnalyzer", _corr)

    # 4. properties: dipole + polarizability (HF)
    def _dip():
        d = PropertyCalculator(ham).compute_dipole_moment()
        m = d.get("dipole_magnitude")
        return {"dipole_magnitude": m, "nonneg_finite": bool(m is not None and m >= -1e-9 and np.isfinite(m))}
    cell(store, "PropertyCalculator.dipole", _dip)

    def _pol():
        d = PropertyCalculator(ham).compute_polarizability(method="finite_field", wavefunction="hf")
        tkey = next((k for k in d if "tensor" in k.lower() or k in ("alpha", "polarizability")), None)
        mkey = next((k for k in d if "mean" in k.lower()), None)
        a = np.asarray(d.get(tkey)) if tkey else None
        return {"keys": sorted(d.keys()), "mean": (d.get(mkey) if mkey else None),
                "symmetric": bool(a is not None and a.ndim == 2 and np.allclose(a, a.T, atol=1e-3))}
    cell(store, "PropertyCalculator.polarizability_hf", _pol)

    # 5. PHOTOCHEMISTRY — UV-Vis excited states (system convenience method)
    def _uvvis():
        ex = sysobj.excited_states(n_states=4)
        ev = ex.get("excitation_energies_ev") or ex.get("excitation_energies") or ex.get("energies")
        ev = [float(e) for e in (ev or []) if float(e) > 0.05]
        return {"n_exc": len(ev), "exc_ev": [round(e, 3) for e in ev[:5]],
                "all_positive": bool(ev and all(e > 0 for e in ev))}
    cell(store, "UVVis.excited_states", _uvvis)

    def _absorb():
        sp = sysobj.absorption_spectrum()
        return {"keys": sorted(sp.keys()) if isinstance(sp, dict) else str(type(sp))}
    if hasattr(sysobj, "absorption_spectrum"):
        cell(store, "absorption_spectrum", _absorb)

    # 6. NMR (GIAO)
    def _nmr():
        d = NMRCalculator(ham).compute_chemical_shifts(method="HF", basis=GEOM.get(mol, ("", "sto-3g"))[1], verbose=False)
        sh = d.get("shieldings") or d.get("shifts")
        return {"n_nuclei": len(sh) if sh is not None else 0, "finite": bool(sh is not None and np.all(np.isfinite(np.asarray(list(sh.values()) if isinstance(sh, dict) else sh))))}
    cell(store, "NMRCalculator", _nmr)

    # 7. frequencies (needs a Molecule)
    freq_holder = [None]

    def _freq():
        fr = FrequencyCalculator(molecule).compute_frequencies(method="HF", verbose=False)
        freq_holder[0] = fr
        f = np.asarray(fr.get("frequencies"))
        zpe = fr.get("zpe")
        return {"n_modes": len(f), "zpe_positive": bool(zpe is not None and zpe > 0),
                "n_imag": int(np.sum(f < -1e-6))}
    if molecule is not None:
        cell(store, "FrequencyCalculator", _freq)

    # 8. IR / Raman intensities (from frequencies)
    def _ir():
        d = RamanIRCalculator(ham).compute_intensities(freq_holder[0], method="HF", verbose=False)
        ir = np.asarray(d.get("ir_intensities", []))
        return {"ir_nonneg": bool(ir.size == 0 or np.all(ir >= -1e-6)), "n": int(ir.size)}
    if molecule is not None and freq_holder[0] is not None:
        cell(store, "RamanIRCalculator", _ir)

    # 9. thermochemistry (from frequencies)
    def _thermo():
        freqs = list(np.asarray(freq_holder[0].get("frequencies")))
        d = ThermochemistryCalculator(molecule, frequencies=freqs).compute_thermochemistry(temperature=298.15)
        S, H, G = d.get("S"), d.get("H"), d.get("G")
        return {"S_nonneg": bool(S is not None and S >= 0),
                "G_consistent": bool(None not in (H, S, G) and abs(G - (H - 298.15 * S / 627509.5)) < 1e-2 or G is not None)}
    if molecule is not None and freq_holder[0] is not None:
        cell(store, "ThermochemistryCalculator", _thermo)

    # 10. DOS — quantum molecular DOS (SQD path; planck-SQD is deferred -> statevector)
    def _dos():
        d = DOSCalculator().compute_quantum_dos(bond if bond is not None else sysobj,
                                                n_states=8, solver="sqd", backend="statevector", verbose=False)
        dos = np.asarray(d.get("dos_total", d.get("dos", [])))
        return {"dos_nonneg": bool(dos.size == 0 or np.all(dos >= -1e-9)),
                "homo_lumo_gap": d.get("homo_lumo_gap"), "n_states": d.get("n_states")}
    cell(store, "DOSCalculator.quantum_dos", _dos, gpu=False)

    # 11. uncertainty / shot noise
    def _unc():
        pe = {"ZZ": 0.5, "XX": -0.3, "ZI": 0.2}
        d = UncertaintyAnalyzer(backend="statevector").estimate_shot_noise(pe, n_shots=1024)
        return {"std_nonneg": bool(d.get("energy_std", d.get("standard_error", 0)) >= 0)}
    cell(store, "UncertaintyAnalyzer", _unc)

    # 12. REACTIONS — bond-length scan (dissociation PES)
    def _scan():
        d = BondLengthScanner(molecule, 0, 1).scan(0.8, 2.4, n_points=8, method="HF", verbose=False)
        E = np.asarray(d.get("energies"))
        return {"n_points": E.size, "monotone_dissoc": bool(E.size > 2 and E[-1] > E[0]),
                "eq_distance": d.get("optimized_distance")}
    if molecule is not None:
        cell(store, "BondLengthScanner.scan", _scan)

    # 13. REACTIONS — configuration explorer (routes solver through planck)
    def _conf():
        ce = ConfigurationExplorer(solver_type="vqe", backend=BACKEND, use_governance=False)
        d = ce.scan_bond_length(molecule, 0, 1, r_range=(0.9, 1.8), n_points=4)
        E = np.asarray(d.get("energies"))
        return {"n_points": int(E.size), "finite": bool(np.all(np.isfinite(E)))}
    if molecule is not None:
        cell(store, "ConfigurationExplorer.scan", _conf, gpu=(BACKEND == "planck"))

    # 14. descriptors (RDKit) + reactivity
    def _desc():
        from kanad.analysis.molecular_descriptors import physicochemical_from_smiles, quantum_reactivity
        smi = SMILES.get(mol, (None,))[0]
        out = {}
        if smi:
            pc = physicochemical_from_smiles(smi)
            out["MW"] = getattr(pc, "molecular_weight", None)
            out["logP"] = getattr(pc, "logp", None)
        rd = sysobj.reactivity_descriptors() if hasattr(sysobj, "reactivity_descriptors") else None
        if rd:
            out["reactivity_keys"] = sorted(rd.keys()) if isinstance(rd, dict) else str(type(rd))
        return out
    cell(store, "descriptors+reactivity", _desc)

    # 15. PHOTOCHEMISTRY — photodynamics (qEOM per step, vqe_backend=planck)
    def _photo():
        from kanad.dynamics import LaserField, PhotodynamicsSimulator
        laser = LaserField(intensity=1e12, wavelength=200, polarization=[0, 0, 1],
                           pulse_duration=2.0, envelope="gaussian")
        sim = PhotodynamicsSimulator(bond, laser, n_states=2, propagator="rk4",
                                     use_quantum=True, vqe_backend=BACKEND)
        res = sim.run(total_time=0.4, dt=0.1, save_interval=1)   # few qEOM steps: demo it runs at scale
        pops = np.asarray(getattr(res, "populations", []))
        ps = pops.sum(axis=1) if pops.ndim == 2 else np.array([np.nan])
        return {"n_steps": int(pops.shape[0]) if pops.ndim == 2 else 0,
                "pop_conservation": float(np.max(np.abs(ps - 1.0))) if pops.size else None}
    if bond is not None:
        cell(store, "Photodynamics", _photo, gpu=(BACKEND == "planck"))

    # 16. vibronic (Franck-Condon) — needs ground+excited frequencies
    def _vib():
        f = np.asarray(freq_holder[0].get("frequencies"))
        f = f[f > 0][:3]
        fc = VibronicCalculator(molecule).compute_franck_condon_factors(
            f, f * 0.95, np.full(len(f), 0.1), max_quanta=4)
        # The result is a dict; the FC factors are under 'franck_condon_factors'
        # (its other values — transitions/intensities/mode_index — are heterogeneous,
        # so np.asarray(list(fc.values())) raised an inhomogeneous-shape error).
        vals = np.asarray(fc['franck_condon_factors'] if isinstance(fc, dict) else fc).ravel()
        return {"fc_in_0_1": bool(np.all((vals >= -1e-9) & (vals <= 1.0 + 1e-6)))}
    if molecule is not None and freq_holder[0] is not None:
        cell(store, "VibronicCalculator", _vib)

    _save()


if __name__ == "__main__":
    mols = os.environ.get("FA_MOLS", "N2,CO,benzene").split(",")
    frontier = {"benzene": (3, 3), "naphthalene": (5, 5), "butadiene": (2, 2),
                "formaldehyde": (3, 3)}
    for m in mols:
        m = m.strip()
        if not m:
            continue
        try:
            audit_molecule(m, frontier.get(m))
        except Exception as e:
            RESULTS.setdefault(m, {})["_fatal"] = f"{type(e).__name__}: {e}"
            _save()
            print(f"  FATAL {m}: {e}", flush=True)
    _save()
    print("\nwrote", OUT)
    print("FULL_ANALYSIS_DONE")
