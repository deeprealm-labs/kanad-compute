"""Round 3 — ADME / druglikeness audit (RDKit physicochemical + rule filters +
quantum reactivity coupling). Validates against literature descriptor values.

Cluster (rdkit installed there):
    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier3_adme
"""
from __future__ import annotations
import dataclasses, traceback

# (name, smiles, {field: (ref, tol)}) — literature/RDKit-canonical reference values
DRUGS = [
    ('aspirin', 'CC(=O)Oc1ccccc1C(=O)O',
     dict(molecular_weight=(180.16, 0.2), tpsa=(63.6, 0.5), h_bond_donors=(1, 0),
          h_bond_acceptors=(3, 1), aromatic_rings=(1, 0), logP_crippen=(1.31, 0.8))),
    ('ibuprofen', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
     dict(molecular_weight=(206.28, 0.2), tpsa=(37.3, 0.5), h_bond_donors=(1, 0),
          h_bond_acceptors=(2, 1), aromatic_rings=(1, 0), logP_crippen=(3.07, 1.0))),
    ('caffeine', 'Cn1cnc2c1c(=O)n(C)c(=O)n2C',
     dict(molecular_weight=(194.19, 0.2), tpsa=(58.4, 0.5), h_bond_donors=(0, 0),
          h_bond_acceptors=(6, 2), aromatic_rings=(2, 1), logP_crippen=(-1.03, 1.0))),
    ('paracetamol', 'CC(=O)Nc1ccc(O)cc1',
     dict(molecular_weight=(151.16, 0.2), tpsa=(49.3, 0.5), h_bond_donors=(2, 0),
          h_bond_acceptors=(2, 1), aromatic_rings=(1, 0), logP_crippen=(1.35, 1.0))),
    ('atorvastatin', 'CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O',
     dict(molecular_weight=(558.64, 1.0), h_bond_donors=(4, 1), h_bond_acceptors=(7, 2),
          rotatable_bonds=(12, 3), aromatic_rings=(4, 1))),
    ('methanol', 'CO',
     dict(molecular_weight=(32.04, 0.1), tpsa=(20.2, 0.5), h_bond_donors=(1, 0),
          h_bond_acceptors=(1, 0), aromatic_rings=(0, 0))),
]

# expected druglikeness verdicts
RULE_EXPECT = {
    'aspirin': dict(lipinski=True, veber=True),
    'ibuprofen': dict(lipinski=True, veber=True),
    'caffeine': dict(lipinski=True, veber=True),
    'paracetamol': dict(lipinski=True, veber=True),
    'atorvastatin': dict(lipinski=False),   # MW>500, HBA>10-ish → violations
    'methanol': dict(lipinski=True, veber=True),
}


def main():
    from kanad.analysis.molecular_descriptors import (
        physicochemical_from_smiles, druglikeness_rules)
    print("=" * 100, flush=True)
    print("ROUND 3 — ADME / druglikeness audit", flush=True)
    print("=" * 100, flush=True)

    for name, smi, refs in DRUGS:
        try:
            p = physicochemical_from_smiles(smi)
            d = dataclasses.asdict(p)
            devs = []
            for f, (ref, tol) in refs.items():
                val = d.get(f)
                ok = (val is not None) and (abs(val - ref) <= tol)
                devs.append(f"{f}={val}{'' if ok else f'(exp~{ref} X)'}")
            rules = druglikeness_rules(p)
            exp = RULE_EXPECT.get(name, {})
            lip_ok = (rules.lipinski_pass == exp.get('lipinski', rules.lipinski_pass))
            n_bad = sum(1 for f, (ref, tol) in refs.items()
                        if d.get(f) is None or abs(d[f] - ref) > tol)
            status = 'PASS' if (n_bad == 0 and lip_ok) else f'{n_bad}-dev'
            print(f"ADME| {name:14} [{status}] | {' '.join(devs)} | "
                  f"rules(L/V/G viol)={rules.lipinski_violations}/{rules.veber_violations}/"
                  f"{rules.ghose_violations} lipinski_pass={rules.lipinski_pass} (exp {exp})", flush=True)
        except Exception as e:
            print(f"ADME| {name:14} [CRASH] {type(e).__name__}: {str(e)[:90]}", flush=True)

    # zwitterion / tautomer sensitivity
    print("\n--- glycine neutral vs zwitterion SMILES ---", flush=True)
    for label, smi in [('neutral', 'C(C(=O)O)N'), ('zwitterion', '[NH3+]CC(=O)[O-]')]:
        try:
            p = physicochemical_from_smiles(smi); d = dataclasses.asdict(p)
            print(f"ADME| glycine_{label:10} | MW={d['molecular_weight']:.2f} TPSA={d['tpsa']:.1f} "
                  f"HBD={d['h_bond_donors']} HBA={d['h_bond_acceptors']} logP={d['logP_crippen']:.2f}", flush=True)
        except Exception as e:
            print(f"ADME| glycine_{label} CRASH {type(e).__name__}: {str(e)[:70]}", flush=True)

    # SMILES parse-failure path
    print("\n--- error-path coverage ---", flush=True)
    for bad in ['C(C(=O', 'not_a_smiles', '']:
        try:
            physicochemical_from_smiles(bad)
            print(f"ADME| parse '{bad}' [FAIL] did NOT raise", flush=True)
        except ValueError as e:
            print(f"ADME| parse '{bad}' [PASS] clean ValueError: {str(e)[:60]}", flush=True)
        except Exception as e:
            print(f"ADME| parse '{bad}' [{type(e).__name__}] {str(e)[:60]}", flush=True)

    # quantum + physchem provenance coupling (aspirin: HF Koopmans reactivity + RDKit descriptors)
    print("\n--- quantum reactivity_descriptors(smiles=) provenance split ---", flush=True)
    try:
        from kanad import MolecularBuilder
        qs = (MolecularBuilder.from_smiles('CC(=O)Oc1ccccc1C(=O)O', 'sto-3g')
              .active_space('frontier', n_occ=3, n_virt=3).solver('ci').build())
        out = qs.reactivity_descriptors(smiles='CC(=O)Oc1ccccc1C(=O)O')
        qr = out['quantum_reactivity']; phys = out.get('physicochemical'); rules = out.get('druglikeness')
        import dataclasses as dc
        qrd = dc.asdict(qr) if dc.is_dataclass(qr) else vars(qr)
        print(f"ADME| aspirin quantum+physchem [PASS] | reactivity source={qrd.get('source')} "
              f"chi={qrd.get('electronegativity_ev'):.2f} eta={qrd.get('hardness_ev'):.2f} "
              f"gap={qrd.get('gap_ev'):.2f}eV | physchem present={phys is not None} rules present={rules is not None}", flush=True)
    except Exception as e:
        print(f"ADME| aspirin quantum+physchem [CRASH] {type(e).__name__}: {str(e)[:90]}", flush=True)
        traceback.print_exc()

    print("\nADME_DONE", flush=True)


if __name__ == "__main__":
    main()
