"""`MolecularBuilder` — one fluent entry point for every Kanad workflow.

Construct a molecular system once, then run energy / analysis / dynamics /
reactions / photochemistry off the same `QuantumSystem`. Every stage has a
smart default and a full override (the "absolute freedom" requirement):

    # one-liner default (auto active space + auto solver)
    MolecularBuilder.from_smiles("O").build().solve()

    # tuned active space + explicit SQD on hardware
    (MolecularBuilder.from_xyz("fe2s2.xyz").basis("cc-pvdz")
        .active_space("mp2no", max_orbitals=10, occ_threshold=0.02)
        .solver("sqd", backend="ibm", recovery_rounds=3, n_samples=20000)
        .build().solve())

    # geometry-stable active space for a scan, then the dynamics closure
    qs = (MolecularBuilder.from_atoms(n2_atoms).basis("cc-pvdz")
            .active_space("manual", frozen=[0, 1], active=list(range(2, 12)))
            .build())
    e, warm = qs.energy_fn()(atoms_bohr, warm_state=None)

The builder holds a `SystemSpec`; `.build()` materializes it at the reference
geometry. `.spec()` exposes the raw spec for power users who want to mutate
fields the fluent API doesn't surface.
"""

from __future__ import annotations

from typing import Any, Sequence

from kanad.builder.system_spec import SystemSpec
from kanad.builder.quantum_system import QuantumSystem


def _normalize_atoms(atoms: Sequence[Any]):
    """Accept kanad `Atom` objects or ``(symbol, (x, y, z))`` tuples (Angstrom)."""
    out = []
    for a in atoms:
        if hasattr(a, 'symbol') and hasattr(a, 'position'):
            out.append((a.symbol, tuple(float(x) for x in a.position)))
        elif isinstance(a, (tuple, list)) and len(a) == 2:
            sym, pos = a
            out.append((str(sym), tuple(float(x) for x in pos)))
        else:
            raise ValueError(
                f"Cannot interpret atom {a!r}; expected a kanad Atom or a "
                "(symbol, (x, y, z)) tuple in Angstrom."
            )
    return tuple(out)


class MolecularBuilder:
    def __init__(self, atoms: Sequence[Any] = None):
        self._spec = SystemSpec(
            atoms=_normalize_atoms(atoms) if atoms is not None else ()
        )

    # ----- constructors -------------------------------------------------

    @classmethod
    def from_atoms(cls, atoms: Sequence[Any]) -> 'MolecularBuilder':
        """From kanad `Atom` objects or ``(symbol, (x, y, z))`` tuples."""
        return cls(atoms)

    @classmethod
    def from_molecule(cls, molecule) -> 'MolecularBuilder':
        """From a kanad `Molecule` (carries its charge / spin / basis)."""
        b = cls(_normalize_atoms(molecule.atoms))
        b._spec.charge = int(getattr(molecule, 'charge', 0))
        b._spec.spin = int(getattr(molecule, 'spin', 0))
        b._spec.basis = getattr(molecule, 'basis', 'sto-3g')
        return b

    @classmethod
    def from_smiles(cls, smiles: str, basis: str = 'sto-3g',
                    optimize_geometry: bool = True) -> 'MolecularBuilder':
        """From a SMILES string (RDKit 3D embedding)."""
        from kanad.core.io.smiles_parser import from_smiles as _from_smiles
        mol = _from_smiles(smiles, basis=basis, optimize_geometry=optimize_geometry)
        return cls.from_molecule(mol)

    @classmethod
    def from_xyz(cls, path: str, charge: int = 0, spin: int = 0,
                 basis: str = 'sto-3g') -> 'MolecularBuilder':
        """From an XYZ file."""
        from kanad.core.io.xyz_io import from_xyz as _from_xyz
        mol = _from_xyz(path, charge=charge, spin=spin, basis=basis)
        return cls.from_molecule(mol)

    # ----- fluent configuration (each returns self) ---------------------

    def basis(self, basis: str) -> 'MolecularBuilder':
        self._spec.basis = basis
        return self

    def charge(self, charge: int) -> 'MolecularBuilder':
        self._spec.charge = int(charge)
        return self

    def spin(self, spin: int) -> 'MolecularBuilder':
        self._spec.spin = int(spin)
        return self

    def active_space(self, strategy: str, *, policy: str = None,
                     **kwargs) -> 'MolecularBuilder':
        """Set the active-space strategy.

        ``strategy``: 'auto' | 'full' | 'frozen_core' | 'frontier' | 'mp2no' |
        'avas' | 'manual'. Strategy kwargs pass straight through (e.g. ``frozen=``,
        ``active=`` for manual; ``n_occ=``, ``n_virt=`` for frontier;
        ``max_orbitals=``, ``occ_threshold=`` for mp2no; ``ao_labels=``,
        ``threshold=`` for avas — works for closed- and open-shell ROHF).
        ``policy='freeze'``
        (default) keeps the active space continuous along a scan; ``'reselect'``
        re-picks at each geometry.
        """
        self._spec.active_space_strategy = strategy
        self._spec.active_space_kwargs = dict(kwargs)
        if policy is not None:
            self._spec.active_space_policy = policy
        return self

    def ansatz(self, ansatz_type: str, **kwargs) -> 'MolecularBuilder':
        """Set the ansatz ('auto' | 'lucj' | 'hardware_efficient' | 'givens_sd' | ...).

        ``mapper=`` here also sets the system mapper; other kwargs (``n_layers=``,
        ``target_sz=``) flow to the ansatz constructor.
        """
        self._spec.ansatz_type = ansatz_type
        kwargs = dict(kwargs)
        if 'mapper' in kwargs:
            self._spec.mapper = kwargs.pop('mapper')
        self._spec.ansatz_kwargs = kwargs
        return self

    def mapper(self, mapper: str) -> 'MolecularBuilder':
        self._spec.mapper = mapper
        return self

    def solver(self, solver: str, *, backend: str = None,
               **kwargs) -> 'MolecularBuilder':
        """Set the solver ('auto' | 'ci' | 'vqe' | 'sqd').

        ``backend=`` (statevector | qasm | bluequbit | ibm) routes here too.
        Remaining kwargs pass to the chosen solver (e.g. ``n_samples=``,
        ``recovery_rounds=`` for SQD; ``optimizer=``, ``max_iterations=`` for VQE).
        """
        self._spec.solver = solver
        if backend is not None:
            self._spec.backend = backend
        self._spec.solver_kwargs = dict(kwargs)
        return self

    def backend(self, backend: str) -> 'MolecularBuilder':
        self._spec.backend = backend
        return self

    def conditions(self, solvent: str = None, pH: float = None,
                   temperature: float = None, pressure: float = None,
                   thermal: bool = None, sites: list = None) -> 'MolecularBuilder':
        """Set environmental conditions applied to the solved energy.

        - ``solvent`` — name in the solvent database ('water', 'methanol',
          'hexane', …); routes through **real PCM** (`pyscf.solvent.pcm`), adding
          ΔG_solv to the energy.
        - ``thermal=True`` — add the **RRHO free-energy correction** (ZPE + thermal
          enthalpy − T·S) at ``temperature`` (K) and ``pressure`` (atm), from
          computed vibrational frequencies. Turns the electronic energy into a
          Gibbs free energy. (Needs a near-minimum geometry; computes a Hessian.)
        - ``pH`` — Henderson-Hasselbalch protonation free energy (`pHModulator`).
        - ``sites`` — explicit protonation sites so pH works on geometry-only
          molecules (no connectivity auto-detection). A list of dicts:
          ``[{'atom_index': 0, 'group_type': 'carboxylic_acid'}, …]``; an
          optional ``'pKa'`` per dict overrides the database value.

        Only the args you pass change; the rest keep their defaults.
        """
        c = dict(self._spec.conditions)
        if solvent is not None:
            c['solvent'] = solvent
        if pH is not None:
            c['pH'] = pH
        if temperature is not None:
            c['temperature'] = temperature
        if pressure is not None:
            c['pressure'] = pressure
        if thermal is not None:
            c['thermal'] = thermal
        if sites is not None:
            c['ph_sites'] = sites
        self._spec.conditions = c
        return self

    def spec(self) -> SystemSpec:
        """Escape hatch: the raw `SystemSpec` for fields the fluent API omits."""
        return self._spec

    # ----- materialize --------------------------------------------------

    def build(self) -> QuantumSystem:
        """Materialize at the reference geometry → `QuantumSystem`."""
        if not self._spec.atoms:
            raise ValueError(
                "MolecularBuilder has no atoms. Use from_smiles / from_xyz / "
                "from_atoms / from_molecule before build()."
            )
        mf, ham = self._spec.materialize_at(None)
        return QuantumSystem(self._spec, mf, ham)
