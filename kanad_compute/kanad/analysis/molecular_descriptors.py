"""Molecular descriptors for medicinal-chemistry triage — honest about provenance.

Two clearly-separated descriptor families, never blended:

1. **Physicochemical** (RDKit — validated 2D cheminformatics): molecular weight,
   Crippen logP, topological polar surface area (Ertl TPSA), H-bond donor/acceptor
   counts, rotatable-bond count, aromatic-ring count. These are standard,
   reproducible, literature-validated descriptors — they are *not* "quantum".
   Requires RDKit; raises a clear ImportError if it is missing (no fake fallback).

2. **Quantum reactivity** (conceptual DFT — from the solved wavefunction): the
   frontier-orbital gap and the Parr/Pearson global reactivity indices —
   electronegativity χ, chemical hardness η, global softness S, and the
   electrophilicity index ω — in Koopmans' approximation from the HOMO/LUMO
   energies. These *are* wavefunction-derived (the ``source`` field records
   whether the orbital energies are HF/Koopmans or correlated ΔSCF).

The previous ``ADMECalculator`` predicted logP from molecular weight and gave
Caco-2 / PAMPA / BBB / plasma-protein-binding numbers from hand-tuned step
functions with invented coefficients — empirical curve-fitting dressed as
"quantum ML". Those predictions are removed: they were neither quantum nor
validated. Drug-likeness *rule filters* (Lipinski, Veber, Ghose) are kept —
they are threshold checks on the real descriptors above, not predictive models.

References:
- Wildman & Crippen (1999) JCICS 39:868 — atomic-contribution logP.
- Ertl, Rohde & Selzer (2000) JMC 43:3714 — topological polar surface area.
- Lipinski et al. (1997) ADDR 23:3; Veber et al. (2002) JMC 45:2615;
  Ghose et al. (1999) J. Comb. Chem. 1:55 — drug-likeness rule filters.
- Parr & Pearson (1983) JACS 105:7512 — chemical hardness/softness.
- Parr, Szentpály & Liu (1999) JACS 121:1922 — electrophilicity index ω.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

HARTREE_TO_EV = 27.211386245988


@dataclass
class PhysicochemicalDescriptors:
    """Validated 2D descriptors from RDKit. Not quantum — standard cheminformatics."""
    molecular_weight: float           # g/mol
    heavy_atom_count: int
    logP_crippen: float               # Wildman-Crippen logP
    tpsa: float                       # Ertl topological polar surface area, Å²
    h_bond_donors: int
    h_bond_acceptors: int
    rotatable_bonds: int
    aromatic_rings: int
    ring_count: int


@dataclass
class QuantumReactivityDescriptors:
    """Conceptual-DFT global reactivity indices (eV) from the frontier orbitals.

    ``source`` records the orbital-energy provenance:
    ``'koopmans_hf'`` (HF canonical orbital energies, Koopmans' theorem) or
    ``'delta_scf'`` (finite-difference IP/EA from total energies).
    """
    homo_ev: float
    lumo_ev: float
    gap_ev: float
    electronegativity_ev: float       # χ = -(εH + εL)/2
    chemical_potential_ev: float      # μ = -χ
    hardness_ev: float                # η = (εL - εH)/2
    softness_per_ev: float            # S = 1/(2η)
    electrophilicity_ev: float        # ω = μ²/(2η) = χ²/(2η)
    source: str = 'koopmans_hf'
    dipole_debye: Optional[float] = None


@dataclass
class DrugLikenessRules:
    """Rule-filter violations (threshold checks on the real descriptors)."""
    lipinski_violations: int
    veber_violations: int
    ghose_violations: int

    @property
    def lipinski_pass(self) -> bool:        # Rule of Five tolerates ≤1 violation
        return self.lipinski_violations <= 1

    @property
    def veber_pass(self) -> bool:
        return self.veber_violations == 0

    @property
    def ghose_pass(self) -> bool:
        return self.ghose_violations == 0


def physicochemical_from_smiles(smiles: str) -> PhysicochemicalDescriptors:
    """Validated RDKit 2D descriptors for a SMILES string.

    Raises:
        ImportError: if RDKit is not installed (no fabricated fallback).
        ValueError: if the SMILES cannot be parsed.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    except ImportError as exc:                                    # pragma: no cover
        raise ImportError(
            "physicochemical_from_smiles requires RDKit (the validated descriptor "
            "engine). Install it with `pip install rdkit` (or conda-forge `rdkit`). "
            "There is no hand-tuned fallback by design — fake topology numbers were "
            "removed."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}.")

    return PhysicochemicalDescriptors(
        molecular_weight=float(Descriptors.MolWt(mol)),
        heavy_atom_count=int(mol.GetNumHeavyAtoms()),
        logP_crippen=float(Crippen.MolLogP(mol)),
        tpsa=float(Descriptors.TPSA(mol)),
        h_bond_donors=int(Lipinski.NumHDonors(mol)),
        h_bond_acceptors=int(Lipinski.NumHAcceptors(mol)),
        rotatable_bonds=int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        aromatic_rings=int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        ring_count=int(rdMolDescriptors.CalcNumRings(mol)),
    )


def quantum_reactivity(homo_ev: float, lumo_ev: float,
                       dipole_debye: Optional[float] = None,
                       source: str = 'koopmans_hf') -> QuantumReactivityDescriptors:
    """Parr/Pearson conceptual-DFT global reactivity indices from frontier energies.

    χ = -(εH+εL)/2, μ = -χ, η = (εL-εH)/2, S = 1/(2η), ω = μ²/(2η). All in eV.

    Args:
        homo_ev, lumo_ev: HOMO / LUMO energies in eV.
        dipole_debye: optional dipole magnitude to carry alongside.
        source: orbital-energy provenance label (see the dataclass docstring).
    """
    gap = lumo_ev - homo_ev
    chi = -(homo_ev + lumo_ev) / 2.0
    mu = -chi
    eta = gap / 2.0
    softness = float('inf') if eta == 0 else 1.0 / (2.0 * eta)
    omega = float('inf') if eta == 0 else (mu * mu) / (2.0 * eta)
    return QuantumReactivityDescriptors(
        homo_ev=homo_ev, lumo_ev=lumo_ev, gap_ev=gap,
        electronegativity_ev=chi, chemical_potential_ev=mu,
        hardness_ev=eta, softness_per_ev=softness, electrophilicity_ev=omega,
        source=source, dipole_debye=dipole_debye,
    )


def druglikeness_rules(phys: PhysicochemicalDescriptors) -> DrugLikenessRules:
    """Lipinski / Veber / Ghose rule-filter violations on the real descriptors."""
    lip = 0
    lip += phys.molecular_weight > 500
    lip += phys.logP_crippen > 5
    lip += phys.h_bond_donors > 5
    lip += phys.h_bond_acceptors > 10

    veb = 0
    veb += phys.rotatable_bonds > 10
    veb += phys.tpsa > 140

    gho = 0
    gho += not (-0.4 <= phys.logP_crippen <= 5.6)
    gho += not (160 <= phys.molecular_weight <= 480)

    return DrugLikenessRules(int(lip), int(veb), int(gho))
