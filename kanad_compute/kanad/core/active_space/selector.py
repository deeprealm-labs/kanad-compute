"""Orbital-selection logic.

`ActiveSpace` is the result of a selection: which canonical (or rotated)
orbitals are frozen, active, virtual, with the n_active_electrons consistent
with `n_electrons_total - 2·|frozen|`.

`ActiveSpaceSelector` operates on a converged PySCF mean-field and exposes
the M1 selection methods. NOON / AVAS / CASSCF selectors are M2 work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveSpace:
    """Result of an active-space selection.

    Invariants
    ----------
    - ``frozen + active + virtual`` partition the full orbital set
      (their union has size ``n_orbitals_total`` with no overlap).
    - ``n_active_electrons == n_electrons_total - 2·|frozen|`` — valid for
      closed-shell and for open-shell (ROHF) references whose frozen orbitals
      are all doubly occupied (the singly-occupied SOMOs stay active).
    - ``n_active_orbitals == len(active_indices)``.
    - ``mo_coeff`` is the C matrix the indices refer to; downstream code uses
      this to build integrals consistently.
    """

    frozen_indices: Tuple[int, ...]
    active_indices: Tuple[int, ...]
    virtual_indices: Tuple[int, ...]
    n_active_electrons: int
    n_active_orbitals: int
    mo_coeff: np.ndarray = field(repr=False)
    method: str  # 'manual' | 'frozen_core' | 'frontier'

    def __post_init__(self):
        # Sanity checks — these throw if a selector is buggy.
        total = set(self.frozen_indices) | set(self.active_indices) | set(self.virtual_indices)
        if len(total) != len(self.frozen_indices) + len(self.active_indices) + len(self.virtual_indices):
            raise ValueError("frozen/active/virtual indices overlap")
        if self.n_active_orbitals != len(self.active_indices):
            raise ValueError(
                f"n_active_orbitals={self.n_active_orbitals} but len(active_indices)={len(self.active_indices)}"
            )
        # Electron-count invariant: an active space cannot hold more electrons than
        # 2·(orbitals). A manual partition that excludes OCCUPIED orbitals from both
        # frozen and active (e.g. open-shell FeO: frozen=[], active=8..20, so the 8
        # occupied core orbitals are dropped to "virtual" but their electrons are NOT
        # removed) assigns the full molecular electron count to a truncated space →
        # n_active_electrons > 2·n_active_orbitals. Catch it HERE with an actionable
        # message instead of a cryptic CircuitError deep in the ansatz HF-seed.
        if self.n_active_electrons < 0:
            raise ValueError(
                f"active space has negative n_active_electrons={self.n_active_electrons} "
                f"(too many frozen orbitals for the electron count)")
        if self.n_active_electrons > 2 * self.n_active_orbitals:
            raise ValueError(
                f"active space is over-filled: n_active_electrons={self.n_active_electrons} "
                f"> 2·n_active_orbitals={2 * self.n_active_orbitals}. This usually means "
                f"occupied orbitals were excluded from BOTH frozen and active (their "
                f"electrons are still counted). Add those occupied orbitals to `frozen` "
                f"(open- or closed-shell, as long as they are doubly occupied), or "
                f"include them in `active`. For open-shell systems the singly-occupied "
                f"SOMOs must be in the active space, never frozen.")


class ActiveSpaceSelector:
    """Pure orbital-selection logic — does not touch integrals.

    Operates on a converged PySCF mean-field. Construction is cheap;
    each selection method returns an `ActiveSpace`.
    """

    def __init__(self, mf):
        self.mf = mf
        self.mol = mf.mol
        # ROHF mo_coeff is a single (n_ao, n_mo) matrix (like RHF); UHF would
        # be a 2-tuple (α, β) which we don't support here.
        mo_coeff = np.asarray(mf.mo_coeff)
        if mo_coeff.ndim == 3:
            raise ValueError(
                "ActiveSpaceSelector does not support UHF (α/β mo_coeff). "
                "Use ROHF for open-shell systems."
            )
        self.n_orbitals_total = int(mo_coeff.shape[1])
        self.n_electrons_total = int(self.mol.nelectron)
        # Open-shell (odd-electron / ROHF) is supported as long as every FROZEN
        # orbital is doubly occupied — the frozen-core integral transform uses
        # the closed-shell 2·J−K mean field, which is exact for an ROHF core
        # whose singly-occupied orbitals (SOMOs) all live in the ACTIVE space.
        # `manual()` enforces this (per-frozen-orbital mo_occ≈2 check); `avas()`
        # enforces it via openshell_option=2 (SOMOs forced active) + an
        # even-inactive-electron assertion. Freezing a SOMO is the only thing
        # blocked (it would reintroduce the M1 variational-violation bug class).
        self._is_open_shell = (self.n_electrons_total % 2 != 0) or (
            getattr(self.mol, 'spin', 0) != 0
        )

    # ----- selection methods -----------------------------------------

    def manual(
        self,
        frozen: Sequence[int],
        active: Sequence[int],
        _method: str = 'manual',
        mo_coeff: Optional[np.ndarray] = None,
        _validate_openshell: bool = True,
    ) -> ActiveSpace:
        """User-specified partition.

        Any orbital not in ``frozen`` or ``active`` is treated as virtual
        (assumed empty). The private ``_method`` kwarg lets other selectors
        delegate construction to `manual` while preserving their own provenance
        tag on the returned `ActiveSpace`.

        ``mo_coeff`` overrides the orbital basis the indices refer to. It
        defaults to ``self.mf.mo_coeff`` (canonical MOs); selectors that rotate
        the basis (e.g. `mp2_natural_orbitals`) pass their own coefficient
        matrix so the indices index into the rotated orbitals.

        ``_validate_openshell`` (private) gates the open-shell frozen-core
        check below. Selectors that rotate the basis into an explicit
        [core | active | virtual] ordering (e.g. `avas`) set it False and do
        their own doubly-occupied-core validation, because ``self.mf.mo_occ``
        is not index-aligned with their rotated orbitals.
        """
        frozen_t = tuple(sorted(frozen))
        active_t = tuple(sorted(active))
        # Open-shell frozen-core guard. The frozen-core integral transform
        # (build_active_space_hamiltonian) uses the closed-shell 2·J−K mean
        # field, which is EXACT for an ROHF reference *as long as every frozen
        # orbital is doubly occupied* — the open-shell character (SOMOs) must
        # live in the ACTIVE space, not the frozen core. So we no longer block
        # open-shell+frozen wholesale; we block only a frozen orbital that is
        # NOT doubly occupied (a SOMO frozen by mistake would make 2·J−K wrong).
        if self._is_open_shell and len(frozen_t) > 0 and _validate_openshell:
            occ = np.asarray(self.mf.mo_occ)
            bad = [int(i) for i in frozen_t if not np.isclose(occ[i], 2.0, atol=1e-6)]
            if bad:
                raise ValueError(
                    f"open-shell frozen-core: frozen orbitals {bad} are not doubly "
                    f"occupied (mo_occ≈2 is required for the 2·J−K frozen-core "
                    f"transform). Keep singly-occupied (SOMO) orbitals in the active "
                    f"space, widen the frontier window (n_occ), or use frozen=[]."
                )
        all_idx = set(range(self.n_orbitals_total))
        virtual_t = tuple(sorted(all_idx - set(frozen_t) - set(active_t)))
        n_active_electrons = self.n_electrons_total - 2 * len(frozen_t)
        if n_active_electrons < 0:
            raise ValueError(
                f"Manual frozen list of size {len(frozen_t)} would consume "
                f"{2*len(frozen_t)} electrons, but molecule has only "
                f"{self.n_electrons_total}."
            )
        C = self.mf.mo_coeff if mo_coeff is None else mo_coeff
        return ActiveSpace(
            frozen_indices=frozen_t,
            active_indices=active_t,
            virtual_indices=virtual_t,
            n_active_electrons=n_active_electrons,
            n_active_orbitals=len(active_t),
            mo_coeff=np.asarray(C).copy(),
            method=_method,
        )

    def frozen_core(self) -> ActiveSpace:
        """Freeze the inner-shell orbitals using a chemical rule of thumb.

        - Atoms with Z ≤ 2 (H, He): no frozen orbitals.
        - Atoms with Z ≤ 10 (Li…Ne): freeze 1s (1 orbital per atom).
        - Atoms with Z > 10 (Na…): freeze 1s2s2p (5 orbitals per atom).

        Matches the heuristic used inline in `PhysicsVQE._determine_frozen_core`.
        """
        n_frozen = 0
        for atom_id in range(self.mol.natm):
            Z = int(self.mol.atom_charge(atom_id))
            if Z > 10:
                n_frozen += 5
            elif Z > 2:
                n_frozen += 1
        n_frozen = min(n_frozen, self.n_orbitals_total - 1)
        n_frozen = min(n_frozen, self.n_electrons_total // 2)
        frozen = tuple(range(n_frozen))
        active = tuple(range(n_frozen, self.n_orbitals_total))
        return self.manual(frozen=frozen, active=active, _method='frozen_core')

    def frontier(self, n_occ: int, n_virt: int) -> ActiveSpace:
        """HOMO−(n_occ-1) through LUMO+(n_virt-1).

        Always includes the formal HOMO and LUMO when both ``n_occ`` and
        ``n_virt`` are ≥1. The remaining orbitals are either frozen (occupied
        and below HOMO−n_occ+1) or virtual (above LUMO+n_virt−1).
        """
        if n_occ < 1 or n_virt < 1:
            raise ValueError("n_occ and n_virt must each be ≥ 1")
        # HOMO = highest orbital with nonzero occupation. Derived from mf.mo_occ so
        # open-shell ROHF (the SOMO counts as occupied) is centered correctly; the old
        # closed-shell formula n_electrons//2-1 mis-centered the window by ~spin//2 for
        # open-shell systems. For closed-shell the two agree. (CORE_BUGS B23.)
        mo_occ = np.asarray(self.mf.mo_occ)
        occ_idx = np.where(mo_occ > 0)[0]
        homo = int(occ_idx[-1]) if len(occ_idx) else 0  # 0-indexed HOMO/SOMO
        first_active = max(0, homo - n_occ + 1)
        last_active = min(self.n_orbitals_total - 1, homo + n_virt)
        frozen = tuple(range(0, first_active))
        active = tuple(range(first_active, last_active + 1))
        return self.manual(frozen=frozen, active=active, _method='frontier')

    def mp2_natural_orbitals(
        self,
        max_orbitals: Optional[int] = None,
        occ_threshold: float = 0.02,
    ) -> ActiveSpace:
        """Automatic active space from MP2 natural-orbital occupations (NOONs).

        Runs RMP2 on the converged mean field, diagonalizes the MP2 1-RDM to
        get natural orbitals and their occupations, and selects the
        *partially occupied* window ``occ_threshold < n < 2 − occ_threshold``
        as active. Orbitals at/above ``2 − occ_threshold`` are frozen (treated
        as doubly occupied); orbitals at/below ``occ_threshold`` are virtual.

        Unlike the index-based selectors, the returned `ActiveSpace` carries
        the **natural-orbital coefficient matrix** (a rotation of the canonical
        MOs) as ``mo_coeff`` — `build_active_space_hamiltonian` transforms
        integrals in that basis. The NO active space captures more correlation
        per orbital than the canonical frontier set, which is the point.

        Parameters
        ----------
        max_orbitals : int | None
            Cap on the active-orbital count. When the partially-occupied window
            exceeds it, keep the ``max_orbitals`` orbitals closest to half
            filling (``|n − 1|`` smallest); the rest spill to frozen (``n > 1``)
            or virtual (``n < 1``).
        occ_threshold : float
            Occupation tolerance defining the active window. 0.02 is the
            standard NOON cutoff (Keller/Reiher); orbitals more occupied than
            ``2 − 0.02`` or less than ``0.02`` are deemed inactive.

        Raises
        ------
        ValueError
            For open-shell systems — RMP2 NOONs assume a closed-shell RHF
            reference. Use `manual`/`frontier` for open-shell.
        """
        if self._is_open_shell:
            raise ValueError(
                "mp2_natural_orbitals supports closed-shell (RHF) systems only; "
                "the open-shell RMP2 1-RDM is not defined here. Use manual() or "
                "frontier() and place all occupied orbitals in the active space."
            )
        if not (0.0 < occ_threshold < 1.0):
            raise ValueError(f"occ_threshold must be in (0, 1); got {occ_threshold}")

        from pyscf import mp

        mp2 = mp.MP2(self.mf).run()
        # MP2 1-RDM in the canonical-MO basis (HF diagonal + correlation).
        dm1_mo = np.asarray(mp2.make_rdm1())
        # Natural orbitals = eigenvectors; NOONs = eigenvalues. eigh ascending.
        noons, rot = np.linalg.eigh(dm1_mo)
        order = np.argsort(noons)[::-1]          # sort by descending occupation
        noons = noons[order]
        rot = rot[:, order]
        # NO coefficients in AO basis, columns aligned with `noons`.
        C_no = np.asarray(self.mf.mo_coeff) @ rot

        hi = 2.0 - occ_threshold
        lo = occ_threshold
        frozen = [i for i, n in enumerate(noons) if n >= hi]
        active = [i for i, n in enumerate(noons) if lo < n < hi]
        virtual = [i for i, n in enumerate(noons) if n <= lo]

        if max_orbitals is not None and len(active) > max_orbitals:
            # Keep the most strongly correlated (closest to half filling).
            ranked = sorted(active, key=lambda i: abs(noons[i] - 1.0))
            keep = set(ranked[:max_orbitals])
            # A spilled NO may ONLY be demoted to frozen if it is essentially
            # doubly occupied (NOON ≥ hi) or to virtual if essentially empty
            # (NOON ≤ lo). Demoting a PARTIALLY-occupied NO (lo < NOON < hi) to
            # frozen made manual() subtract a full 2 e⁻ for an orbital holding
            # ~1.3–1.9 e⁻ (electron non-conservation + a frozen-core potential on
            # a strongly-correlated orbital); demoting it to virtual silently
            # drops its electrons. Neither conserves electrons, so we KEEP such
            # orbitals active (soft cap) and warn. (CORE_BUGS B5.)
            spill_partial, new_active = [], []
            for i in active:
                if i in keep:
                    new_active.append(i)
                elif noons[i] >= hi:
                    frozen.append(i)
                elif noons[i] <= lo:
                    virtual.append(i)
                else:
                    new_active.append(i)
                    spill_partial.append(i)
            active = new_active
            if spill_partial:
                logger.warning(
                    "mp2_natural_orbitals: max_orbitals=%d could not be honored "
                    "without dropping %d partially-occupied (correlated) natural "
                    "orbital(s) [NOON in (%.3f, %.3f)]; kept them active so the "
                    "electron count stays correct (n_active_orbitals=%d). Raise "
                    "max_orbitals or occ_threshold for a tighter active space.",
                    max_orbitals, len(spill_partial), lo, hi, len(active))

        if len(active) == 0:
            # Near single-reference: every NOON is ~2 or ~0, none in the
            # partially-occupied window. Rather than fail the whole build (C3),
            # fall back to a frontier (HOMO/LUMO) active space — for such systems
            # the canonical frontier orbitals ARE the right correlated window.
            mo_occ = np.asarray(self.mf.mo_occ)
            n_occ_avail = int(np.sum(mo_occ > 0))
            n_virt_avail = int(np.sum(mo_occ == 0))
            k = max(1, (max_orbitals // 2) if max_orbitals else 3)
            k_occ, k_virt = min(k, n_occ_avail), min(k, n_virt_avail)
            if k_occ < 1 or k_virt < 1:
                raise ValueError(
                    "mp2_natural_orbitals found no partially-occupied orbitals at "
                    f"occ_threshold={occ_threshold} and cannot form a frontier fallback "
                    f"(n_occ={n_occ_avail}, n_virt={n_virt_avail})."
                )
            logger.warning(
                "mp2_natural_orbitals: no partially-occupied NOs at occ_threshold=%s "
                "(near single-reference); falling back to frontier(n_occ=%d, n_virt=%d).",
                occ_threshold, k_occ, k_virt)
            return self.frontier(n_occ=k_occ, n_virt=k_virt)

        return self.manual(
            frozen=sorted(frozen),
            active=sorted(active),
            _method='mp2no',
            mo_coeff=C_no,
        )

    def avas(self, ao_labels, threshold: float = 0.2,
             minao: str = 'minao', openshell_option: int = 2) -> ActiveSpace:
        """Atom/orbital-targeted active space (AVAS — Sayfutyarova 2017).

        Selects the active orbitals by their projection onto a chosen set of
        target atomic valence orbitals (``ao_labels``, e.g. ``['Fe 3d']`` for a
        transition-metal center, ``['C 2pz']`` for a π system). This is the
        pyscf-standard automated picker for the chemically-relevant active space
        — the "bond-physics-aligned" selector — returning AVAS-rotated orbitals
        as ``mo_coeff``.

        Works for **closed-shell (RHF) and open-shell (ROHF)** references. For
        open shells, ``openshell_option=2`` (pyscf default) forces every
        singly-occupied orbital into the active space, so the frozen core stays
        doubly occupied and the closed-shell frozen-core transform is exact.

        Args:
            ao_labels: list of AO label strings selecting the target valence set.
            threshold: AVAS occupation/projection threshold (default 0.2).
            minao: minimal reference basis for the projection.
            openshell_option: pyscf AVAS open-shell handling. 2 (default) keeps
                all SOMOs active — the only mode compatible with the
                doubly-occupied frozen-core transform.

        Raises:
            ValueError: if AVAS leaves an odd number of inactive electrons
                (a SOMO would land in the frozen core — raise ``threshold`` or
                fix ``ao_labels`` so the singly-occupied valence set is active).
        """
        from pyscf.mcscf import avas as _avas
        # pyscf's module-level avas() returns FIVE values
        # (ncas, nelecas, mo, occ_weights, vir_weights) — only the AVAS.kernel()
        # METHOD returns 3. Unpacking into 3 names raised ValueError on every call,
        # killing the whole AVAS strategy. Catch-all the trailing weights. (CORE_BUGS B3.)
        ncas, nelecas, mo, *_ = _avas.avas(
            self.mf, ao_labels, threshold=threshold, minao=minao,
            canonicalize=True, openshell_option=openshell_option)
        # nelecas is an int (closed-shell) or an (nα, nβ) tuple (open-shell).
        ne_act = (int(sum(nelecas)) if isinstance(nelecas, (tuple, list, np.ndarray))
                  else int(nelecas))
        n_inactive = self.n_electrons_total - ne_act
        # The inactive electrons must fill a whole number of doubly-occupied core
        # orbitals. An odd remainder means a SOMO was left out of the AVAS active
        # space — the frozen-core transform would then be applied to a singly-
        # occupied orbital (wrong). openshell_option=2 prevents this; assert it.
        if n_inactive < 0 or n_inactive % 2 != 0:
            raise ValueError(
                f"AVAS returned {ne_act} active electrons for a "
                f"{self.n_electrons_total}-electron system, leaving {n_inactive} "
                f"inactive — not a whole number of doubly-occupied core orbitals. "
                f"A singly-occupied orbital must be inside the AVAS active space "
                f"(openshell_option=2 enforces this); check that `ao_labels` covers "
                f"the singly-occupied valence set.")
        ncore = n_inactive // 2
        mo = np.asarray(mo)
        # AVAS (openshell_option=2, canonicalize=True) is MEANT to order the returned MOs
        # as [doubly-occupied core | active | virtual] with every singly-occupied (SOMO)
        # orbital in the active block. For HIGH-SPIN references (>2 SOMOs) it can instead
        # leave some SOMOs in the inactive region — the [0:ncore] frozen slice then treats
        # those SOMOs as DOUBLY occupied, which corrupts the reference so badly that
        # CASCI-in-AS rises ABOVE the SCF energy (mathematically impossible for a valid CAS;
        # observed: Mn atom +205 mHa, Mn2(µ-O)2 +969 mHa, [Fe2S2Cl4]2- +298 mHa). Guard it:
        # a SOMO must never be frozen. Compute the ROHF occupation of each returned MO and
        # union any left-out SOMO into the active space, freezing only doubly-occ orbitals.
        occ = self._rohf_mo_occupations(mo)
        tol = 0.02   # occupation tolerance for "cleanly doubly-occupied"
        clean_double = set(np.where(occ > 2.0 - tol)[0])       # freeze-eligible: occ ~ 2
        partial = set(np.where((occ > tol) & (occ <= 2.0 - tol))[0])  # SOMOs + covalent leakage
        avas_active = set(range(ncore, ncore + int(ncas)))
        # A valid frozen core must be occupation-pure (~doubly occupied); every partially-
        # occupied orbital (open-shell SOMOs AND the fractionally-occupied covalent partners
        # that metal-ligand mixing produces in the AVAS-canonicalised basis) MUST be active,
        # or the frozen-as-doubly-occ reference rises above the SCF (invalid CAS). Union them.
        need_active = partial - avas_active
        if need_active:
            new_active = sorted(avas_active | partial)
            new_frozen = sorted(clean_double - set(new_active))
            new_virtual = sorted(set(range(mo.shape[1])) - set(new_active) - set(new_frozen))
            mo = mo[:, new_frozen + new_active + new_virtual]
            ncore = len(new_frozen)
            frozen = list(range(ncore))
            active = list(range(ncore, ncore + len(new_active)))
            logger.info(
                "AVAS open-shell rescue: %d partially-occupied orbital(s) (SOMOs / covalent "
                "partners) were in the frozen core; moved into the active space -> CAS(%de, %do).",
                len(need_active), self.n_electrons_total - 2 * ncore, len(active))
            if len(active) > 24:
                logger.warning(
                    "AVAS open-shell active space grew to %d orbitals (%d qubits): strong "
                    "metal-ligand covalency spreads occupation over many orbitals, so a "
                    "*valid* frozen-core CAS is necessarily large. For a compact valid space "
                    "use CASSCF orbital optimisation instead of a frozen-core AVAS reference.",
                    len(active), 2 * len(active))
        else:
            frozen = list(range(ncore))
            active = list(range(ncore, ncore + int(ncas)))
        return self.manual(frozen=frozen, active=active, _method='avas',
                           mo_coeff=mo, _validate_openshell=False)

    def _rohf_mo_occupations(self, mo: np.ndarray) -> np.ndarray:
        """ROHF total occupation (2/1/0) of each column of ``mo`` in the current MF basis,
        via projection of the mean-field 1-RDM: n_i = (S mo_i)^T D (S mo_i)."""
        S = self.mf.get_ovlp()
        dm = self.mf.make_rdm1()
        dm_tot = dm[0] + dm[1] if (isinstance(dm, np.ndarray) and dm.ndim == 3) else np.asarray(dm)
        Smo = S @ mo
        return np.einsum('pi,pq,qi->i', Smo, dm_tot, Smo)
