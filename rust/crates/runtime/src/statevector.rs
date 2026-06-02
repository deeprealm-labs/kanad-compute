//! Dense statevector simulator. Stores 2^n complex amplitudes in a flat
//! `Vec<Complex64>` indexed by the standard little-endian bitstring
//! (qubit 0 = least significant bit). All gate applications are in-place;
//! we walk the index space in strides so we touch each amplitude exactly
//! once per gate.

use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

pub type C = Complex64;

#[derive(Debug, Clone)]
pub struct StateVector {
    n_qubits: usize,
    amps: Vec<C>,
}

impl StateVector {
    /// |0…0⟩
    pub fn zero(n_qubits: usize) -> Self {
        assert!(n_qubits <= 30, "statevector limited to 30 qubits");
        let mut amps = vec![C::new(0.0, 0.0); 1 << n_qubits];
        amps[0] = C::new(1.0, 0.0);
        Self { n_qubits, amps }
    }

    pub fn from_amps(amps: Vec<C>) -> Self {
        let n = amps.len();
        assert!(n.is_power_of_two(), "amplitude count must be 2^n");
        Self {
            n_qubits: n.trailing_zeros() as usize,
            amps,
        }
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn amps(&self) -> &[C] {
        &self.amps
    }

    pub fn amps_mut(&mut self) -> &mut [C] {
        &mut self.amps
    }

    pub fn norm_sq(&self) -> f64 {
        self.amps.iter().map(|a| a.norm_sqr()).sum()
    }

    /// Apply a 2x2 unitary to a single qubit in-place.
    pub fn apply_1q(&mut self, gate: &[[C; 2]; 2], target: usize) {
        debug_assert!(target < self.n_qubits);
        let stride = 1usize << target;
        let block = stride << 1;
        let len = self.amps.len();
        let [[u00, u01], [u10, u11]] = *gate;
        let mut base = 0usize;
        while base < len {
            for off in 0..stride {
                let i0 = base + off;
                let i1 = i0 + stride;
                let a0 = self.amps[i0];
                let a1 = self.amps[i1];
                self.amps[i0] = u00 * a0 + u01 * a1;
                self.amps[i1] = u10 * a0 + u11 * a1;
            }
            base += block;
        }
    }

    /// Apply a controlled 2x2 unitary on `target` conditioned on `control` == |1⟩.
    pub fn apply_controlled_1q(&mut self, gate: &[[C; 2]; 2], control: usize, target: usize) {
        debug_assert!(control != target);
        debug_assert!(control < self.n_qubits && target < self.n_qubits);
        let c_mask = 1usize << control;
        let t_stride = 1usize << target;
        let t_block = t_stride << 1;
        let len = self.amps.len();
        let [[u00, u01], [u10, u11]] = *gate;
        let mut base = 0usize;
        while base < len {
            for off in 0..t_stride {
                let i0 = base + off;
                let i1 = i0 + t_stride;
                if i0 & c_mask != 0 {
                    let a0 = self.amps[i0];
                    let a1 = self.amps[i1];
                    self.amps[i0] = u00 * a0 + u01 * a1;
                    self.amps[i1] = u10 * a0 + u11 * a1;
                }
            }
            base += t_block;
        }
    }

    /// Swap two qubits.
    pub fn apply_swap(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let lo_mask = 1usize << lo;
        let hi_mask = 1usize << hi;
        let len = self.amps.len();
        for i in 0..len {
            // Only touch indices where lo=1, hi=0 to avoid double-swapping.
            if (i & lo_mask) != 0 && (i & hi_mask) == 0 {
                let j = (i & !lo_mask) | hi_mask;
                self.amps.swap(i, j);
            }
        }
    }
}

// ---- gate matrices ----

pub mod gates {
    use super::C;

    fn c(re: f64, im: f64) -> C {
        C::new(re, im)
    }
    pub fn pauli_x() -> [[C; 2]; 2] {
        [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]]
    }
    pub fn pauli_y() -> [[C; 2]; 2] {
        [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]]
    }
    pub fn pauli_z() -> [[C; 2]; 2] {
        [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)]]
    }
    pub fn hadamard() -> [[C; 2]; 2] {
        let h = super::FRAC_1_SQRT_2;
        [[c(h, 0.0), c(h, 0.0)], [c(h, 0.0), c(-h, 0.0)]]
    }
    pub fn s() -> [[C; 2]; 2] {
        [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, 1.0)]]
    }
    pub fn t() -> [[C; 2]; 2] {
        let r = super::FRAC_1_SQRT_2;
        [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(r, r)]]
    }
    pub fn rx(theta: f64) -> [[C; 2]; 2] {
        let h = theta * 0.5;
        let co = h.cos();
        let si = h.sin();
        [[c(co, 0.0), c(0.0, -si)], [c(0.0, -si), c(co, 0.0)]]
    }
    pub fn ry(theta: f64) -> [[C; 2]; 2] {
        let h = theta * 0.5;
        let co = h.cos();
        let si = h.sin();
        [[c(co, 0.0), c(-si, 0.0)], [c(si, 0.0), c(co, 0.0)]]
    }
    pub fn rz(theta: f64) -> [[C; 2]; 2] {
        let h = theta * 0.5;
        let co = h.cos();
        let si = h.sin();
        [[c(co, -si), c(0.0, 0.0)], [c(0.0, 0.0), c(co, si)]]
    }
    pub fn phase(theta: f64) -> [[C; 2]; 2] {
        [
            [c(1.0, 0.0), c(0.0, 0.0)],
            [c(0.0, 0.0), c(theta.cos(), theta.sin())],
        ]
    }
}

// ---- high-level circuit API ----

/// A single instruction in a quantum circuit. Compact enum so VQE ansätze
/// can hand the simulator a `Vec<Op>` and have it executed deterministically.
#[derive(Debug, Clone)]
pub enum Op {
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    T(usize),
    Rx(usize, f64),
    Ry(usize, f64),
    Rz(usize, f64),
    Cnot { control: usize, target: usize },
    Cz { control: usize, target: usize },
    Swap(usize, usize),
}

impl Op {
    pub fn apply(&self, sv: &mut StateVector) {
        match *self {
            Op::H(q) => sv.apply_1q(&gates::hadamard(), q),
            Op::X(q) => sv.apply_1q(&gates::pauli_x(), q),
            Op::Y(q) => sv.apply_1q(&gates::pauli_y(), q),
            Op::Z(q) => sv.apply_1q(&gates::pauli_z(), q),
            Op::S(q) => sv.apply_1q(&gates::s(), q),
            Op::T(q) => sv.apply_1q(&gates::t(), q),
            Op::Rx(q, t) => sv.apply_1q(&gates::rx(t), q),
            Op::Ry(q, t) => sv.apply_1q(&gates::ry(t), q),
            Op::Rz(q, t) => sv.apply_1q(&gates::rz(t), q),
            Op::Cnot { control, target } => {
                sv.apply_controlled_1q(&gates::pauli_x(), control, target)
            }
            Op::Cz { control, target } => {
                sv.apply_controlled_1q(&gates::pauli_z(), control, target)
            }
            Op::Swap(a, b) => sv.apply_swap(a, b),
        }
    }
}

/// Apply a sequence of operations to a fresh |0…0⟩ state.
pub fn run_circuit(n_qubits: usize, ops: &[Op]) -> StateVector {
    let mut sv = StateVector::zero(n_qubits);
    for op in ops {
        op.apply(&mut sv);
    }
    sv
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn close(a: C, b: C) {
        assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-10);
        assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-10);
    }

    #[test]
    fn zero_state_is_unit_norm() {
        let sv = StateVector::zero(3);
        assert_eq!(sv.amps().len(), 8);
        assert_abs_diff_eq!(sv.norm_sq(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn pauli_x_flips_zero_to_one() {
        let mut sv = StateVector::zero(1);
        sv.apply_1q(&gates::pauli_x(), 0);
        close(sv.amps()[0], C::new(0.0, 0.0));
        close(sv.amps()[1], C::new(1.0, 0.0));
    }

    #[test]
    fn hadamard_creates_plus_state() {
        let mut sv = StateVector::zero(1);
        sv.apply_1q(&gates::hadamard(), 0);
        let h = FRAC_1_SQRT_2;
        close(sv.amps()[0], C::new(h, 0.0));
        close(sv.amps()[1], C::new(h, 0.0));
    }

    #[test]
    fn rx_pi_is_minus_i_x() {
        // RX(π) |0⟩ = -i |1⟩
        let mut sv = StateVector::zero(1);
        sv.apply_1q(&gates::rx(std::f64::consts::PI), 0);
        close(sv.amps()[0], C::new(0.0, 0.0));
        close(sv.amps()[1], C::new(0.0, -1.0));
    }

    #[test]
    fn cnot_entangles_to_bell() {
        // |00⟩ -H_0-> (|00⟩+|10⟩)/√2 -CNOT(0,1)-> (|00⟩+|11⟩)/√2
        let sv = run_circuit(
            2,
            &[
                Op::H(0),
                Op::Cnot {
                    control: 0,
                    target: 1,
                },
            ],
        );
        let h = FRAC_1_SQRT_2;
        close(sv.amps()[0], C::new(h, 0.0)); // |00⟩
        close(sv.amps()[1], C::new(0.0, 0.0)); // |01⟩
        close(sv.amps()[2], C::new(0.0, 0.0)); // |10⟩
        close(sv.amps()[3], C::new(h, 0.0)); // |11⟩
    }

    #[test]
    fn swap_exchanges_qubits() {
        // |10⟩ in little-endian = index 1 ; swap(0,1) → |01⟩ = index 2
        let mut sv = StateVector::zero(2);
        Op::X(0).apply(&mut sv);
        assert_eq!(sv.amps()[1], C::new(1.0, 0.0));
        Op::Swap(0, 1).apply(&mut sv);
        close(sv.amps()[1], C::new(0.0, 0.0));
        close(sv.amps()[2], C::new(1.0, 0.0));
    }

    #[test]
    fn unitarity_preserved_under_random_circuit() {
        let sv = run_circuit(
            4,
            &[
                Op::H(0),
                Op::Ry(1, 0.7),
                Op::Cnot {
                    control: 0,
                    target: 2,
                },
                Op::Rz(3, -1.3),
                Op::Cz {
                    control: 1,
                    target: 3,
                },
                Op::Rx(2, 0.4),
                Op::Swap(0, 3),
            ],
        );
        assert_abs_diff_eq!(sv.norm_sq(), 1.0, epsilon = 1e-10);
    }
}
