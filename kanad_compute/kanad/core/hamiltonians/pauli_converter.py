"""
Convert molecular Hamiltonians to Qiskit Pauli operators.

This module bridges the gap between the fermionic Hamiltonians
(h_core, ERI tensors) and Qiskit's SparsePauliOp representation
using the mappers (Jordan-Wigner, Bravyi-Kitaev, etc.).
"""

from typing import Dict, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PauliConverter:
    """
    Converts molecular Hamiltonians to Qiskit Pauli operators.

    Uses fermionic-to-qubit mappers to transform:
        H = Σ h_ij a†_i a_j + ½ Σ g_ijkl a†_i a†_j a_l a_k + E_nn

    into Pauli operator representation:
        H = Σ c_k P_k

    where P_k are Pauli strings (e.g., 'XXYZI') and c_k are coefficients.
    """

    @staticmethod
    def to_sparse_pauli_op(hamiltonian, mapper, use_qiskit_nature=True):
        """
        Convert molecular Hamiltonian to Qiskit SparsePauliOp.

        Args:
            hamiltonian: MolecularHamiltonian instance
            mapper: Fermionic-to-qubit mapper (JW, BK, etc.)
            use_qiskit_nature: If True, use Qiskit Nature for correct two-electron terms
                             (RECOMMENDED - mathematically correct!)

        Returns:
            qiskit.quantum_info.SparsePauliOp
        """
        try:
            from qiskit.quantum_info import SparsePauliOp
        except ImportError:
            raise ImportError(
                "Qiskit not installed. Install with: pip install qiskit>=2.0"
            )

        # Try Qiskit Nature approach
        # Always use it - even for charged systems we'll use the fermionic operators
        # (The bug is in ElectronicEnergy, but we can work around it)
        if use_qiskit_nature and hasattr(hamiltonian, 'eri') and hasattr(hamiltonian, 'h_core'):
            try:
                logger.info("Using Qiskit Nature fermionic operators for Pauli conversion")
                return PauliConverter._to_pauli_qiskit_nature(hamiltonian, mapper)
            except ImportError:
                logger.info("Qiskit Nature not installed, falling back")
                pass
            except Exception as e:
                logger.warning(f"Qiskit Nature approach failed ({e}), falling back")
                pass

        # Collect all Pauli terms
        pauli_dict = {}  # {pauli_string: coefficient}

        n_orbitals = hamiltonian.n_orbitals

        # CRITICAL FIX: Transform AO basis integrals to MO basis
        # VQE operates on MO basis, not AO basis!

        # Get MO coefficients from HF calculation
        mo_energies, C = hamiltonian.compute_molecular_orbitals()

        # AO->MO via indigenous core.integrals (reorg B3; identity C for an
        # ActiveHamiltonian => no-op on already-MO integrals).
        from kanad.core.integrals.transforms import ao2mo_transform, one_index_transform
        h_core = one_index_transform(hamiltonian.h_core, C)

        # Transform ERI to MO basis
        if hasattr(hamiltonian, 'eri') and hamiltonian.eri is not None:
            eri = ao2mo_transform(hamiltonian.eri, C, chemist=True)
        else:
            eri = None

        # For Jordan-Wigner and other mappers: need spin orbitals (alpha + beta)
        # Each spatial orbital has 2 spin orbitals (spin-up, spin-down)
        n_spin_orbitals = n_orbitals * 2
        n_qubits = mapper.n_qubits(n_spin_orbitals)

        # 1. One-body terms: Σ h_ij a†_i a_j
        for i in range(n_orbitals):
            for j in range(n_orbitals):
                if abs(h_core[i, j]) > 1e-10:
                    # Map fermionic operator a†_i a_j to Pauli operators
                    # Map for both spin-up and spin-down
                    for spin_offset in [0, n_orbitals]:  # spin-up (0), spin-down (n_orbitals)
                        i_spin = i + spin_offset
                        j_spin = j + spin_offset
                        pauli_terms = mapper.map_excitation_operator(i_spin, j_spin, n_spin_orbitals)

                        for pauli_string, coeff in pauli_terms.items():
                            full_coeff = h_core[i, j] * coeff

                            if pauli_string in pauli_dict:
                                pauli_dict[pauli_string] += full_coeff
                            else:
                                pauli_dict[pauli_string] = full_coeff

        # 2. Two-body terms: (1/2) Σ ⟨ij|kl⟩ a†_i a†_k a_l a_j
        # CRITICAL: Qiskit Nature convention (from SymmetricTwoBody):
        # For eri[i,j,k,l] = ⟨ij|kl⟩, the operator is a†_i a†_k a_l a_j
        # NOT a†_i a†_j a_l a_k!
        if eri is not None:
            for i in range(n_orbitals):
                for j in range(n_orbitals):
                    for k in range(n_orbitals):
                        for l in range(n_orbitals):
                            v_ijkl = eri[i, j, k, l]  # ⟨ij|kl⟩

                            if abs(v_ijkl) > 1e-10:
                                # Map operator: a†_i a†_k a_l a_j
                                # For all spin combinations
                                for spin_i_l in [0, n_orbitals]:  # Same spin for i, l
                                    for spin_k_j in [0, n_orbitals]:  # Same spin for k, j
                                        i_spin = i + spin_i_l
                                        k_spin = k + spin_k_j
                                        l_spin = l + spin_i_l
                                        j_spin = j + spin_k_j

                                        # Use mapper to convert a†_i a†_k a_l a_j to Pauli
                                        if hasattr(mapper, 'map_double_excitation'):
                                            # map_double_excitation(orb_from_1, orb_from_2, orb_to_1, orb_to_2)
                                            # for a†_i a†_k a_l a_j: annihilate j,l; create i,k
                                            pauli_terms = mapper.map_double_excitation(
                                                j_spin, l_spin, i_spin, k_spin, n_spin_orbitals
                                            )
                                        else:
                                            pauli_terms = PauliConverter._map_two_body_approximate(
                                                mapper, i_spin, k_spin, l_spin, j_spin, n_spin_orbitals
                                            )

                                        for pauli_string, coeff in pauli_terms.items():
                                            full_coeff = 0.5 * v_ijkl * coeff

                                            if pauli_string in pauli_dict:
                                                pauli_dict[pauli_string] += full_coeff
                                            else:
                                                pauli_dict[pauli_string] = full_coeff

        # 3. Nuclear repulsion (constant term - identity operator)
        # n_qubits already calculated above
        identity_string = 'I' * n_qubits

        if identity_string in pauli_dict:
            pauli_dict[identity_string] += hamiltonian.nuclear_repulsion
        else:
            pauli_dict[identity_string] = hamiltonian.nuclear_repulsion

        # Honor the requested mapper. This fallback previously ALWAYS returned a
        # Jordan-Wigner operator, silently ignoring a Bravyi-Kitaev request (the
        # returned operator was wrong for the requested mapper — CORE_BUGS B10); it
        # also dropped frozen_core_energy from the constant term (CORE_BUGS B19).
        # Mirror _to_pauli_qiskit_nature. (h_core/eri above are already MO-basis.)
        const = hamiltonian.nuclear_repulsion + getattr(hamiltonian, 'frozen_core_energy', 0.0)
        _name = (type(mapper).__name__ + ' ' + str(getattr(mapper, 'name', '') or '')
                 + (' ' + mapper if isinstance(mapper, str) else '')).lower()
        if 'bravyi' in _name:
            from kanad.core.operators.bravyi_kitaev import build_molecular_hamiltonian_bk
            return build_molecular_hamiltonian_bk(
                h_mo=h_core, eri_mo=eri, nuclear_repulsion=const,
                n_electrons=hamiltonian.n_electrons)
        from kanad.core.operators.jordan_wigner import build_molecular_hamiltonian_jw
        return build_molecular_hamiltonian_jw(
            h_mo=h_core, eri_mo=eri, nuclear_repulsion=const,
            n_electrons=hamiltonian.n_electrons)

    @staticmethod
    def _map_two_body_approximate(mapper, i, j, k, l, n_orbitals):
        """
        Approximate two-body operator using product of excitations.

        a†_i a†_j a_l a_k ≈ (a†_i a_k)(a†_j a_l)

        This is not exact due to anticommutation, but provides a starting point.
        """
        # Get single excitations
        exc_ik = mapper.map_excitation_operator(k, i, n_orbitals)
        exc_jl = mapper.map_excitation_operator(l, j, n_orbitals)

        # Multiply Pauli strings
        result = mapper.pauli_string_multiply(exc_ik, exc_jl)

        return result

    @staticmethod
    def to_pauli_dict(hamiltonian, mapper) -> Dict[str, complex]:
        """
        Convert Hamiltonian to dictionary of Pauli terms.

        Returns:
            Dictionary mapping Pauli strings to coefficients
        """
        sparse_pauli = PauliConverter.to_sparse_pauli_op(hamiltonian, mapper)

        pauli_dict = {}
        for pauli, coeff in zip(sparse_pauli.paulis, sparse_pauli.coeffs):
            pauli_dict[str(pauli)] = complex(coeff)

        return pauli_dict

    @staticmethod
    def count_pauli_terms(hamiltonian, mapper) -> int:
        """Count number of Pauli terms in Hamiltonian."""
        sparse_pauli = PauliConverter.to_sparse_pauli_op(hamiltonian, mapper)
        return len(sparse_pauli)

    @staticmethod
    def get_hamiltonian_info(hamiltonian, mapper) -> Dict:
        """
        Get information about the Pauli decomposition.

        Returns:
            Dictionary with statistics
        """
        sparse_pauli = PauliConverter.to_sparse_pauli_op(hamiltonian, mapper)

        coeffs = np.abs(sparse_pauli.coeffs)

        return {
            'num_terms': len(sparse_pauli),
            'max_coeff': np.max(coeffs),
            'min_coeff': np.min(coeffs[coeffs > 0]) if np.any(coeffs > 0) else 0,
            'mean_coeff': np.mean(coeffs),
            'total_weight': np.sum(coeffs),
        }

    @staticmethod
    def _to_pauli_qiskit_nature(hamiltonian, mapper):
        """
        Convert Hamiltonian to Pauli operators using native implementation.

        This method uses Kanad's native Jordan-Wigner transformation,
        eliminating external dependencies like OpenFermion or qiskit-nature.

        Args:
            hamiltonian: MolecularHamiltonian with h_core and eri
            mapper: Mapper type ('jordan_wigner' or 'bravyi_kitaev')

        Returns:
            qiskit.quantum_info.SparsePauliOp
        """
        from kanad.core.operators.jordan_wigner import build_molecular_hamiltonian_jw

        # Get MO coefficients
        mo_energies, C = hamiltonian.compute_molecular_orbitals()

        # Transform to MO basis via indigenous core.integrals (reorg B3)
        from kanad.core.integrals.transforms import ao2mo_transform, one_index_transform
        h_core_mo = one_index_transform(hamiltonian.h_core, C)
        eri_mo = ao2mo_transform(hamiltonian.eri, C, chemist=True)

        # Honor the requested mapper. Previously this method always emitted a
        # Jordan-Wigner operator and silently ignored a Bravyi-Kitaev request.
        # JW and BK are unitarily equivalent (same spectrum) but the operators
        # differ — return the one the caller asked for.
        _name = (type(mapper).__name__ + ' ' + str(getattr(mapper, 'name', '') or '')
                 + (' ' + mapper if isinstance(mapper, str) else '')).lower()
        if 'bravyi' in _name or 'bravyi_kitaev' in _name:
            from kanad.core.operators.bravyi_kitaev import build_molecular_hamiltonian_bk
            return build_molecular_hamiltonian_bk(
                h_mo=h_core_mo,
                eri_mo=eri_mo,
                # Include frozen-core energy in the constant term; CovalentHamiltonian
                # keeps it separate from nuclear_repulsion (covalent_hamiltonian.py:353),
                # and the convention sums both. getattr guards Hamiltonians (e.g.
                # ActiveHamiltonian) that fold core energy into nuclear_repulsion.
                nuclear_repulsion=hamiltonian.nuclear_repulsion + getattr(hamiltonian, 'frozen_core_energy', 0.0),
                n_electrons=hamiltonian.n_electrons,
            )

        return build_molecular_hamiltonian_jw(
            h_mo=h_core_mo,
            eri_mo=eri_mo,
            # Include frozen-core energy in the constant term (see BK branch above).
            nuclear_repulsion=hamiltonian.nuclear_repulsion + getattr(hamiltonian, 'frozen_core_energy', 0.0),
            n_electrons=hamiltonian.n_electrons
        )
