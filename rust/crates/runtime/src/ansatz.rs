//! Parameterized quantum circuits ("ansätze") consumed by VQE.
//!
//! Each ansatz exposes `parameter_count()` and `build(&params) -> Vec<Op>`,
//! so VQE only sees a black-box `params -> circuit` mapping. Today we
//! ship one: a hardware-efficient ansatz (HEA) of RY+RZ rotations on each
//! qubit followed by a linear ladder of CNOT entanglers, repeated for a
//! configurable number of layers.

use crate::statevector::Op;

pub trait Ansatz: Send + Sync {
    fn n_qubits(&self) -> usize;
    fn parameter_count(&self) -> usize;
    fn build(&self, params: &[f64]) -> Vec<Op>;
}

/// Hardware-efficient ansatz: `n_layers` blocks of (RY ⊗ RZ on each qubit,
/// then a linear CNOT ladder `q→q+1`). Final block omits the entangler
/// because trailing CNOTs only renormalize globally — same expressibility
/// per parameter.
#[derive(Debug, Clone)]
pub struct HardwareEfficientAnsatz {
    n_qubits: usize,
    n_layers: usize,
}

impl HardwareEfficientAnsatz {
    pub fn new(n_qubits: usize, n_layers: usize) -> Self {
        assert!(n_qubits >= 1);
        assert!(n_layers >= 1);
        Self { n_qubits, n_layers }
    }
}

impl Ansatz for HardwareEfficientAnsatz {
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    fn parameter_count(&self) -> usize {
        // Each layer: 2 rotations per qubit (RY + RZ).
        2 * self.n_qubits * self.n_layers
    }

    fn build(&self, params: &[f64]) -> Vec<Op> {
        assert_eq!(params.len(), self.parameter_count(), "param count mismatch");
        let mut ops = Vec::with_capacity(self.parameter_count() + self.n_qubits * self.n_layers);
        let mut p = 0;
        for layer in 0..self.n_layers {
            for q in 0..self.n_qubits {
                ops.push(Op::Ry(q, params[p]));
                p += 1;
                ops.push(Op::Rz(q, params[p]));
                p += 1;
            }
            if layer + 1 < self.n_layers && self.n_qubits > 1 {
                for q in 0..self.n_qubits - 1 {
                    ops.push(Op::Cnot {
                        control: q,
                        target: q + 1,
                    });
                }
            }
        }
        ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statevector::run_circuit;
    use approx::assert_abs_diff_eq;

    #[test]
    fn parameter_count_is_2qn() {
        let a = HardwareEfficientAnsatz::new(3, 4);
        assert_eq!(a.parameter_count(), 24);
    }

    #[test]
    fn zero_params_gives_zero_state() {
        let a = HardwareEfficientAnsatz::new(2, 2);
        let params = vec![0.0; a.parameter_count()];
        let sv = run_circuit(a.n_qubits(), &a.build(&params));
        assert_abs_diff_eq!(sv.amps()[0].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sv.amps()[0].im, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sv.norm_sq(), 1.0, epsilon = 1e-10);
    }
}
