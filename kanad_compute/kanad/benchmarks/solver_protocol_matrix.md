# Solver-Protocol Conformance Matrix

Per-cell timeout: 75s. ✅=SolverResult+energy (Ha), ⊘=gated-infeasible, ⏱=timeout, ❌=error.

| solver | H2 | HeH+ | LiH | H2O | BeH2 |
|---|---|---|---|---|---|
| vqe | ✅ -1.1182 | ✅ -2.8230 | ✅ -5.0233 | ✅ -39.0807 | ✅ -5.7077 |
| ci | ✅ -1.1373 | ✅ -2.8627 | ⊘ skip | ⊘ skip | ⊘ skip |
| deterministic_ci | ✅ -1.1373 | ✅ -2.8627 | ⊘ skip | ⊘ skip | ⊘ skip |
| lanczos | ✅ -1.1373 | ✅ -2.8627 | ⊘ skip | ⊘ skip | ⊘ skip |
| excited_states | ✅ -1.1168 | ✅ -2.8543 | ✅ -7.8619 | ⊘ skip | ⊘ skip |
| smart | ✅ -1.1373 | ✅ -2.8627 | ✅ -7.8823 | ⊘ skip | ⊘ skip |
| physics_vqe | ✅ -1.1373 | ✅ -2.8622 | ✅ -7.8798 | ✅ -74.9861 | ✅ -15.5770 |
| hardware_vqe | ✅ -1.1369 | ✅ -3.1204 | ❌ fail | ❌ fail | ❌ fail |
| sampling_sqd | ✅ -1.1168 | ✅ -2.8543 | ✅ -7.8619 | ✅ -74.9627 | ✅ -15.5594 |
| varqite | ✅ -0.0486 | ✅ -1.8030 | ✅ -4.1862 | ⏱ t/o | ⏱ t/o |
| qeom_vqe | ✅ -1.1373 | ✅ -2.8627 | ⊘ skip | ⊘ skip | ⊘ skip |
| sampled_subspace_vqe | ✅ -1.1373 | ✅ -2.8627 | ✅ -7.8636 | ✅ -74.9628 | ✅ -15.5594 |

## Cell notes

- **ci × LiH** — SKIP: subspace diagonalizer IndexError ~12q
- **ci × H2O** — SKIP: classical CI subspace on 14q too slow (>min)
- **ci × BeH2** — SKIP: classical CI subspace on 14q too slow (>min)
- **deterministic_ci × LiH** — SKIP: dense 2^n Hamiltonian/operator out of reach (>=12 qubits)
- **deterministic_ci × H2O** — SKIP: dense 2^n Hamiltonian/operator out of reach (>=12 qubits)
- **deterministic_ci × BeH2** — SKIP: dense 2^n Hamiltonian/operator out of reach (>=12 qubits)
- **lanczos × LiH** — SKIP: dense 2^n Hamiltonian/operator out of reach (>=12 qubits)
- **lanczos × H2O** — SKIP: dense 2^n Hamiltonian/operator out of reach (>=12 qubits)
- **lanczos × BeH2** — SKIP: dense 2^n Hamiltonian/operator out of reach (>=12 qubits)
- **excited_states × H2O** — SKIP: classical CIS path needs solve_scf (builder ActiveHamiltonian lacks it)
- **excited_states × BeH2** — SKIP: classical CIS path needs solve_scf (builder ActiveHamiltonian lacks it)
- **smart × H2O** — SKIP: FCI subspace too large
- **smart × BeH2** — SKIP: FCI subspace too large
- **hardware_vqe × LiH** — FAIL: TypeError: object of type 'NoneType' has no len()
- **hardware_vqe × H2O** — FAIL: TypeError: object of type 'NoneType' has no len()
- **hardware_vqe × BeH2** — FAIL: TypeError: object of type 'NoneType' has no len()
- **varqite × H2O** — TIMEOUT: >75s
- **varqite × BeH2** — TIMEOUT: >75s
- **qeom_vqe × LiH** — SKIP: qEOM dense excitation operators capped near 8 qubits
- **qeom_vqe × H2O** — SKIP: qEOM dense excitation operators capped near 8 qubits
- **qeom_vqe × BeH2** — SKIP: qEOM dense excitation operators capped near 8 qubits
