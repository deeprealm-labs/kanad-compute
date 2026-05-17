//! Pauli operators on a qubit register.
//!
//! `PauliString` represents a tensor product of single-qubit Paulis
//! with a real coefficient. `PauliSum` is a weighted sum (the canonical
//! Hamiltonian representation after Jordan-Wigner / Bravyi-Kitaev
//! mapping). Expectation values are computed by direct amplitude
//! traversal — for an n-qubit state with N=2^n amplitudes a single
//! Pauli string is O(N) and never materializes the 2^n × 2^n matrix.

use crate::statevector::{StateVector, C};
use num_complex::Complex64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

/// A tensor product of Paulis, one per qubit. `paulis[q]` is the operator
/// acting on qubit `q`. Length equals the number of qubits in the
/// register (use `Pauli::I` for the identity factor).
#[derive(Debug, Clone)]
pub struct PauliString {
    pub paulis: Vec<Pauli>,
    pub coeff: f64,
}

impl PauliString {
    pub fn new(paulis: Vec<Pauli>, coeff: f64) -> Self {
        Self { paulis, coeff }
    }

    pub fn n_qubits(&self) -> usize {
        self.paulis.len()
    }

    /// ⟨ψ|P|ψ⟩ for a single Pauli string. Real-valued because all
    /// single-qubit Paulis are Hermitian.
    pub fn expectation(&self, sv: &StateVector) -> f64 {
        assert_eq!(self.paulis.len(), sv.n_qubits(), "qubit count mismatch");
        let amps = sv.amps();
        let len = amps.len();
        // Precompute per-qubit masks for fast X-flip / Y-phase / Z-sign.
        let mut x_mask = 0usize;
        let mut y_mask = 0usize;
        let mut z_mask = 0usize;
        for (q, p) in self.paulis.iter().enumerate() {
            let bit = 1usize << q;
            match p {
                Pauli::I => {}
                Pauli::X => x_mask |= bit,
                Pauli::Y => {
                    x_mask |= bit; // Y also flips the bit
                    y_mask |= bit;
                }
                Pauli::Z => z_mask |= bit,
            }
        }
        // P|j⟩ = phase(j) · |j ⊕ flip⟩  with phase(j) computed at the
        // column index:
        //   each Y qubit contributes +i if bit_q(j)=0 else -i
        //   each Z qubit contributes +1 if bit_q(j)=0 else -1
        // → phase(j) = i^n_Y · (-1)^popcount(j & (y_mask|z_mask))
        // ⟨ψ|P|ψ⟩ = Σ_j ψ*_{j⊕flip} · phase(j) · ψ_j
        let flip_mask = x_mask;
        let sign_mask = y_mask | z_mask;
        let n_y = (y_mask as u32).count_ones() as i32;
        let i_pow_ny = match n_y.rem_euclid(4) {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        };
        let mut acc = Complex64::new(0.0, 0.0);
        for j in 0..len {
            let i = j ^ flip_mask;
            let sign_parity = ((j & sign_mask) as u32).count_ones() & 1;
            let phase = if sign_parity == 1 { -i_pow_ny } else { i_pow_ny };
            acc += amps[i].conj() * phase * amps[j];
        }
        // For a Hermitian operator the imaginary part must vanish numerically.
        debug_assert!(acc.im.abs() < 1e-9, "non-Hermitian expectation: {acc}");
        self.coeff * acc.re
    }
}

/// Hermitian Hamiltonian as a sum of Pauli strings.
#[derive(Debug, Clone, Default)]
pub struct PauliSum {
    pub terms: Vec<PauliString>,
}

impl PauliSum {
    pub fn new(terms: Vec<PauliString>) -> Self {
        Self { terms }
    }

    pub fn push(&mut self, term: PauliString) {
        self.terms.push(term);
    }

    pub fn expectation(&self, sv: &StateVector) -> f64 {
        self.terms.iter().map(|t| t.expectation(sv)).sum()
    }
}

/// Convenience parser: build a `PauliString` from a label like `"IXZI"` where
/// `paulis[0]` is the rightmost character (Qiskit convention).
pub fn from_label(label: &str, coeff: f64) -> PauliString {
    let n = label.len();
    let mut paulis = vec![Pauli::I; n];
    for (idx, ch) in label.chars().rev().enumerate() {
        paulis[idx] = match ch {
            'I' => Pauli::I,
            'X' => Pauli::X,
            'Y' => Pauli::Y,
            'Z' => Pauli::Z,
            _ => panic!("invalid Pauli label char '{ch}' in {label:?}"),
        };
    }
    PauliString::new(paulis, coeff)
}

// We re-export C so downstream tests don't have to import num-complex.
pub use crate::statevector::C as Amp;

// silence unused if num_complex isn't dragged in elsewhere
#[allow(dead_code)]
fn _force_use(_: C) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statevector::{run_circuit, Op, StateVector};
    use approx::assert_abs_diff_eq;

    #[test]
    fn z_on_zero_state_is_one() {
        let sv = StateVector::zero(1);
        let z = from_label("Z", 1.0);
        assert_abs_diff_eq!(z.expectation(&sv), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn z_on_one_state_is_minus_one() {
        let sv = run_circuit(1, &[Op::X(0)]);
        let z = from_label("Z", 1.0);
        assert_abs_diff_eq!(z.expectation(&sv), -1.0, epsilon = 1e-12);
    }

    #[test]
    fn x_on_plus_state_is_one() {
        let sv = run_circuit(1, &[Op::H(0)]);
        let x = from_label("X", 1.0);
        assert_abs_diff_eq!(x.expectation(&sv), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn y_on_iplus_state_is_one() {
        // |i+⟩ = H S† |0⟩? Easier: prepare via Ry(π/2) and Rz(-π/2).
        // Skip — use the direct prep: |i+⟩ = (|0⟩ + i|1⟩)/√2 via S·H|0⟩.
        let sv = run_circuit(1, &[Op::H(0), Op::S(0)]);
        let y = from_label("Y", 1.0);
        assert_abs_diff_eq!(y.expectation(&sv), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn zz_on_bell_state_is_one() {
        // (|00⟩+|11⟩)/√2 → ⟨ZZ⟩ = 1.
        let sv = run_circuit(2, &[Op::H(0), Op::Cnot { control: 0, target: 1 }]);
        let zz = from_label("ZZ", 1.0);
        assert_abs_diff_eq!(zz.expectation(&sv), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn xx_on_bell_state_is_one() {
        let sv = run_circuit(2, &[Op::H(0), Op::Cnot { control: 0, target: 1 }]);
        let xx = from_label("XX", 1.0);
        assert_abs_diff_eq!(xx.expectation(&sv), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn yy_on_bell_state_is_minus_one() {
        let sv = run_circuit(2, &[Op::H(0), Op::Cnot { control: 0, target: 1 }]);
        let yy = from_label("YY", 1.0);
        assert_abs_diff_eq!(yy.expectation(&sv), -1.0, epsilon = 1e-12);
    }

    #[test]
    fn h2_minimal_hamiltonian_sanity() {
        // H2 in STO-3G after JW + Bravyi-Kitaev reduction to two qubits.
        // Coefficients vary by paper depending on what's absorbed into the
        // identity term; here we just check that the expectation value is
        // a real finite number on a reference computational-basis state.
        let h = PauliSum::new(vec![
            from_label("II", -1.0523732),
            from_label("IZ",  0.39793742),
            from_label("ZI", -0.39793742),
            from_label("ZZ", -0.01128010),
            from_label("XX",  0.18093119),
        ]);
        let hf = run_circuit(2, &[Op::X(0)]); // |01⟩
        let e_hf = h.expectation(&hf);
        assert!(e_hf.is_finite());
        assert!(e_hf > -3.0 && e_hf < 0.0, "unphysical energy: {e_hf}");

        // ⟨XX⟩ on a computational-basis state is 0, so e_hf should equal
        // the diagonal part exactly: -1.0524 + 0.3979·(-1) + (-0.3979)·1 + (-0.01128)·(-1)
        let diag = -1.0523732 - 0.39793742 - 0.39793742 + 0.01128010;
        assert_abs_diff_eq!(e_hf, diag, epsilon = 1e-9);
    }
}
