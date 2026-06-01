//! Variational Quantum Eigensolver — minimize ⟨ψ(θ)|H|ψ(θ)⟩ over the
//! parameter vector θ that defines the ansatz state.
//!
//! Today this drives the gradient-free `NelderMead` minimizer over an
//! `Ansatz`. The same `vqe()` entry point will swap in `argmin`-backed
//! L-BFGS / COBYLA / parameter-shift gradients without callers caring.

use crate::ansatz::Ansatz;
use crate::optim::{Lbfgs, Minimizer, NelderMead};
use crate::pauli::PauliSum;
use crate::statevector::run_circuit;

/// Which optimizer drives the VQE objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizerKind {
    /// Gradient-free downhill simplex. Robust default for low param counts.
    #[default]
    NelderMead,
    /// Limited-memory BFGS with analytic parameter-shift gradients. Scales
    /// to the larger parameter counts where the simplex method stalls.
    Lbfgs,
}

/// Per-iteration callback. Used by `Solver::run` to bridge into the
/// gateway's `ProgressSink`. Returning `true` halts the optimizer.
pub trait VqeCallback: Send {
    /// Called once per optimizer evaluation. Returns `true` to request
    /// cancellation (e.g. user pressed `c` in the TUI).
    fn on_eval(&mut self, iter: usize, energy: f64) -> bool;
}

pub struct NoCallback;
impl VqeCallback for NoCallback {
    fn on_eval(&mut self, _: usize, _: f64) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
pub struct VqeResult {
    pub energy: f64,
    pub params: Vec<f64>,
    pub iterations: usize,
    pub cancelled: bool,
}

pub struct VqeConfig {
    pub max_iters: usize,
    pub ftol: f64,
    pub initial_params: Option<Vec<f64>>,
    pub optimizer: OptimizerKind,
}

impl Default for VqeConfig {
    fn default() -> Self {
        Self {
            max_iters: 500,
            ftol: 1e-6,
            initial_params: None,
            optimizer: OptimizerKind::default(),
        }
    }
}

/// Run VQE.  `cb.on_eval(iter, energy)` is invoked after each objective
/// evaluation; returning `true` cancels the run early.
pub fn vqe<A: Ansatz, C: VqeCallback>(
    hamiltonian: &PauliSum,
    ansatz: &A,
    cfg: &VqeConfig,
    cb: &mut C,
) -> VqeResult {
    let n_params = ansatz.parameter_count();
    let x0 = cfg
        .initial_params
        .clone()
        .unwrap_or_else(|| vec![0.0; n_params]);
    assert_eq!(x0.len(), n_params, "initial param length mismatch");

    let mut evals = 0usize;
    let mut cancelled = false;
    let mut last_energy = f64::INFINITY;

    let (params, energy, iters) = {
        let mut objective = |theta: &[f64]| -> f64 {
            if cancelled {
                return last_energy;
            }
            let ops = ansatz.build(theta);
            let sv = run_circuit(ansatz.n_qubits(), &ops);
            let e = hamiltonian.expectation(&sv);
            evals += 1;
            last_energy = e;
            if cb.on_eval(evals, e) {
                cancelled = true;
            }
            e
        };
        // Both optimizers consume the identical objective; the parameter-shift
        // gradients L-BFGS needs are derived from it inside the optimizer.
        match cfg.optimizer {
            OptimizerKind::NelderMead => NelderMead {
                max_iters: cfg.max_iters,
                ftol: cfg.ftol,
                initial_step: 0.5,
            }
            .minimize(x0, &mut objective),
            OptimizerKind::Lbfgs => Lbfgs {
                max_iters: cfg.max_iters,
                ..Lbfgs::default()
            }
            .minimize(x0, &mut objective),
        }
    };

    VqeResult {
        energy,
        params,
        iterations: iters,
        cancelled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ansatz::HardwareEfficientAnsatz;
    use crate::pauli::from_label;
    use approx::assert_abs_diff_eq;

    #[test]
    fn vqe_finds_minus_one_on_z_hamiltonian() {
        // H = Z on 1 qubit → ground state |1⟩ with E = -1.
        let h = PauliSum::new(vec![from_label("Z", 1.0)]);
        let ansatz = HardwareEfficientAnsatz::new(1, 2);
        let cfg = VqeConfig {
            max_iters: 1000,
            ftol: 1e-8,
            initial_params: Some(vec![0.1; ansatz.parameter_count()]),
            ..VqeConfig::default()
        };
        let mut cb = NoCallback;
        let result = vqe(&h, &ansatz, &cfg, &mut cb);
        assert_abs_diff_eq!(result.energy, -1.0, epsilon = 1e-4);
        assert!(!result.cancelled);
    }

    #[test]
    fn vqe_minimizes_h2_minimal_hamiltonian() {
        // The 2-qubit H2 Hamiltonian (O'Malley et al. 2016 coefficients).
        // Exact ground state ≈ -1.857 Ha.
        let h = PauliSum::new(vec![
            from_label("II", -1.0523732),
            from_label("IZ", 0.39793742),
            from_label("ZI", -0.39793742),
            from_label("ZZ", -0.01128010),
            from_label("XX", 0.18093119),
            from_label("YY", 0.18093119),
        ]);
        let ansatz = HardwareEfficientAnsatz::new(2, 3);
        let cfg = VqeConfig {
            max_iters: 3000,
            ftol: 1e-8,
            // Small random-ish seed to break symmetries.
            initial_params: Some(
                (0..ansatz.parameter_count())
                    .map(|i| 0.1 * (i as f64).sin())
                    .collect(),
            ),
            ..VqeConfig::default()
        };
        let mut cb = NoCallback;
        let r = vqe(&h, &ansatz, &cfg, &mut cb);
        // Loose tolerance — Nelder-Mead on a 12-D landscape isn't going to
        // find the global min every time, but should beat any computational
        // basis state (which sit around -1.117 to -1.84).
        assert!(r.energy < -1.84, "VQE energy {} should beat HF", r.energy);
    }

    #[test]
    fn vqe_lbfgs_reaches_h2_ground_state() {
        // Same H2 Hamiltonian, but driven by L-BFGS with parameter-shift
        // gradients. Because the gradients are exact for this Pauli-rotation
        // ansatz, L-BFGS should converge to the true ground state (≈ -1.857)
        // tighter than Nelder-Mead, well below chemical accuracy.
        let h = PauliSum::new(vec![
            from_label("II", -1.0523732),
            from_label("IZ", 0.39793742),
            from_label("ZI", -0.39793742),
            from_label("ZZ", -0.01128010),
            from_label("XX", 0.18093119),
            from_label("YY", 0.18093119),
        ]);
        let ansatz = HardwareEfficientAnsatz::new(2, 3);
        let cfg = VqeConfig {
            max_iters: 500,
            ftol: 1e-10,
            initial_params: Some(
                (0..ansatz.parameter_count())
                    .map(|i| 0.1 * (i as f64).sin())
                    .collect(),
            ),
            optimizer: OptimizerKind::Lbfgs,
        };
        let mut cb = NoCallback;
        let r = vqe(&h, &ansatz, &cfg, &mut cb);
        assert!(
            r.energy < -1.8565,
            "L-BFGS VQE energy {} should reach the H2 ground state",
            r.energy
        );
        assert!(!r.cancelled);
    }

    #[test]
    fn vqe_callback_can_cancel() {
        struct CancelAt(usize);
        impl VqeCallback for CancelAt {
            fn on_eval(&mut self, iter: usize, _: f64) -> bool {
                iter >= self.0
            }
        }
        let h = PauliSum::new(vec![from_label("Z", 1.0)]);
        let ansatz = HardwareEfficientAnsatz::new(1, 2);
        let cfg = VqeConfig::default();
        let mut cb = CancelAt(5);
        let r = vqe(&h, &ansatz, &cfg, &mut cb);
        assert!(r.cancelled);
    }
}
