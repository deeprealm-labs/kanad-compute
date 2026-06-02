//! `Solver`-trait adapters that bridge the gateway's transport plumbing
//! (`ProgressSink` + `CancelToken`) to the native solver kernels in this
//! crate.
//!
//! `VqeSolver` is the first real solver wired into the gateway. Until the
//! native molecule→Hamiltonian lowering (integrals + Jordan-Wigner) lands
//! with the PhysicsVQE/HardwareVQE port, it consumes a **pre-mapped**
//! qubit Hamiltonian carried in `SolverSpec.extra["hamiltonian"]` as a list
//! of `{ "label": "IZ", "coeff": 0.39 }` terms (Qiskit label convention,
//! rightmost char = qubit 0). The cloud performs the mapping today; this
//! keeps the wire-up honest without blocking on the chemistry port.

use crate::ansatz::{Ansatz, HardwareEfficientAnsatz};
use crate::pauli::{from_label, PauliSum};
use crate::statevector::run_circuit;
use crate::vqe::{vqe, OptimizerKind, VqeCallback, VqeConfig};
use crate::{CancelToken, ProgressSink, Solver, SolverError};
use kanad_protocol::{ExperimentRequest, FinalResultPayload, ProgressPayload, SolverSpec};
use std::collections::HashMap;
use std::time::Instant;

/// Largest register we'll allocate a dense statevector for. 2^22 complex
/// amplitudes ≈ 64 MiB — past this a dense simulator is the wrong tool and
/// the job belongs on a Tier-2/Tier-3 path.
const MAX_QUBITS: usize = 22;

/// Native VQE over a hardware-efficient ansatz driven by Nelder-Mead.
pub struct VqeSolver;

impl Solver for VqeSolver {
    fn name(&self) -> &'static str {
        "vqe"
    }

    fn run(
        &mut self,
        request: &ExperimentRequest,
        progress: &mut dyn ProgressSink,
        cancel: &dyn CancelToken,
    ) -> Result<FinalResultPayload, SolverError> {
        let started = Instant::now();
        let hamiltonian = decode_hamiltonian(&request.solver)?;
        let n_qubits = hamiltonian
            .terms
            .first()
            .map(|t| t.n_qubits())
            .ok_or_else(|| SolverError::Failed("hamiltonian has no terms".into()))?;

        let n_layers = request.solver.n_layers.filter(|&l| l >= 1).unwrap_or(2) as usize;
        let ansatz = HardwareEfficientAnsatz::new(n_qubits, n_layers);

        // Deterministic non-zero seed: all-zeros is a saddle point for HEA,
        // so a small structured perturbation breaks the symmetry without
        // introducing run-to-run nondeterminism.
        let n_params = ansatz.parameter_count();
        let initial_params: Vec<f64> = (0..n_params).map(|i| 0.1 * (i as f64).sin()).collect();

        let optimizer = select_optimizer(request.solver.optimizer.as_deref());
        let cfg = VqeConfig {
            max_iters: request.solver.max_iterations.max(1) as usize,
            ftol: request
                .solver
                .convergence_threshold
                .filter(|t| *t > 0.0)
                .unwrap_or(1e-6),
            initial_params: Some(initial_params),
            optimizer,
        };

        // Hartree-Fock reference: ⟨0…0|H|0…0⟩, the all-zeros computational
        // basis state. Reported alongside the VQE energy so the cloud/UI can
        // show the correlation energy recovered.
        let hf_energy = hamiltonian.expectation(&run_circuit(n_qubits, &[]));

        let mut bridge = VqeBridge {
            sink: progress,
            cancel,
            total: cfg.max_iters as i64,
            best: f64::INFINITY,
            history: Vec::new(),
        };

        let result = vqe(&hamiltonian, &ansatz, &cfg, &mut bridge);

        if result.cancelled {
            return Err(SolverError::Cancelled);
        }

        let history = std::mem::take(&mut bridge.history);
        let wall_time_ms = started.elapsed().as_millis() as i64;

        let mut extra: HashMap<String, serde_json::Value> = HashMap::new();
        extra.insert("n_qubits".into(), serde_json::json!(n_qubits));
        extra.insert("n_layers".into(), serde_json::json!(n_layers));
        extra.insert("n_parameters".into(), serde_json::json!(n_params));
        extra.insert("ansatz".into(), serde_json::json!("hardware_efficient"));
        extra.insert(
            "optimizer".into(),
            serde_json::json!(match optimizer {
                OptimizerKind::NelderMead => "nelder_mead",
                OptimizerKind::Lbfgs => "lbfgs",
            }),
        );

        Ok(FinalResultPayload {
            energy: Some(result.energy),
            hf_energy: Some(hf_energy),
            fci_energy: None,
            error_mha: None,
            n_evaluations: Some(result.iterations as i64),
            converged: Some(result.iterations < cfg.max_iters),
            convergence_history: Some(history),
            wall_time_ms: Some(wall_time_ms),
            actual_backend: Some("kanad_compute_statevector".into()),
            extra,
        })
    }
}

/// Parse `SolverSpec.extra["hamiltonian"]` into a `PauliSum`.
fn decode_hamiltonian(spec: &SolverSpec) -> Result<PauliSum, SolverError> {
    let raw = spec.extra.get("hamiltonian").ok_or_else(|| {
        SolverError::Failed(
            "vqe requires a pre-mapped qubit Hamiltonian in \
             solver.extra['hamiltonian'] (list of {label, coeff}) until \
             native integrals land"
                .into(),
        )
    })?;

    let terms = raw.as_array().ok_or_else(|| {
        SolverError::Failed("hamiltonian must be a JSON array of {label, coeff}".into())
    })?;
    if terms.is_empty() {
        return Err(SolverError::Failed("hamiltonian is empty".into()));
    }

    let mut sum = PauliSum::default();
    let mut width: Option<usize> = None;
    for (i, term) in terms.iter().enumerate() {
        let label = term
            .get("label")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SolverError::Failed(format!("term {i} missing string 'label'")))?;
        let coeff = term
            .get("coeff")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| SolverError::Failed(format!("term {i} missing numeric 'coeff'")))?;

        let n = label.len();
        if n == 0 {
            return Err(SolverError::Failed(format!("term {i} has empty label")));
        }
        if n > MAX_QUBITS {
            return Err(SolverError::Failed(format!(
                "hamiltonian width {n} exceeds dense-simulator cap {MAX_QUBITS}"
            )));
        }
        match width {
            None => width = Some(n),
            Some(w) if w != n => {
                return Err(SolverError::Failed(format!(
                    "ragged hamiltonian: term {i} has width {n}, expected {w}"
                )));
            }
            _ => {}
        }
        if !label.chars().all(|c| matches!(c, 'I' | 'X' | 'Y' | 'Z')) {
            return Err(SolverError::Failed(format!(
                "term {i} label {label:?} has non-Pauli characters"
            )));
        }
        sum.push(from_label(label, coeff));
    }
    Ok(sum)
}

/// Map the wire-level `SolverSpec.optimizer` string onto a native optimizer.
/// Any L-BFGS spelling selects the gradient-based path; everything else
/// (including `None`) falls back to the robust Nelder-Mead default.
fn select_optimizer(name: Option<&str>) -> OptimizerKind {
    match name.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("lbfgs") | Some("l-bfgs") | Some("l_bfgs") | Some("bfgs") => OptimizerKind::Lbfgs,
        _ => OptimizerKind::NelderMead,
    }
}

/// Bridges the VQE optimizer's per-evaluation callback onto the gateway's
/// `ProgressSink` (live convergence curve) and `CancelToken` (cooperative
/// stop). To avoid flooding the wire on a Nelder-Mead run, a Progress event
/// is emitted only when the best energy improves (the curve is monotone) —
/// every evaluation is still recorded for the final convergence history.
struct VqeBridge<'a> {
    sink: &'a mut dyn ProgressSink,
    cancel: &'a dyn CancelToken,
    total: i64,
    best: f64,
    history: Vec<HashMap<String, serde_json::Value>>,
}

impl VqeCallback for VqeBridge<'_> {
    fn on_eval(&mut self, iter: usize, energy: f64) -> bool {
        self.history.push(HashMap::from([
            ("iteration".to_string(), serde_json::json!(iter)),
            ("energy".to_string(), serde_json::json!(energy)),
        ]));

        if energy < self.best - 1e-9 || iter == 1 {
            self.best = energy.min(self.best);
            self.sink.emit_progress(ProgressPayload {
                iteration: iter as i64,
                total: Some(self.total),
                energy: Some(self.best),
                gradient_norm: None,
                message: None,
            });
        }
        self.cancel.is_cancelled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NeverCancelled;
    use kanad_protocol::{MoleculeSpec, SolverSpec};

    struct CollectSink(Vec<ProgressPayload>);
    impl ProgressSink for CollectSink {
        fn emit_progress(&mut self, p: ProgressPayload) {
            self.0.push(p);
        }
    }

    fn h2_request(extra: HashMap<String, serde_json::Value>) -> ExperimentRequest {
        ExperimentRequest {
            experiment_id: "e1".into(),
            user_id: "u1".into(),
            molecule: MoleculeSpec {
                atoms: vec![],
                basis: "sto-3g".into(),
                charge: 0,
                multiplicity: 1,
            },
            solver: SolverSpec {
                type_: "vqe".into(),
                ansatz_type: "hardware_efficient".into(),
                max_iterations: 3000,
                max_excitations: 5,
                optimizer: None,
                mapper_type: None,
                convergence_threshold: Some(1e-8),
                n_layers: Some(3),
                shots: None,
                frozen_core: false,
                include_singles: true,
                include_doubles: true,
                extra,
            },
            backend: "kanad_compute".into(),
            backend_credentials: None,
            deadline_ms: 600_000,
        }
    }

    fn h2_hamiltonian() -> serde_json::Value {
        serde_json::json!([
            {"label": "II", "coeff": -1.0523732},
            {"label": "IZ", "coeff":  0.39793742},
            {"label": "ZI", "coeff": -0.39793742},
            {"label": "ZZ", "coeff": -0.01128010},
            {"label": "XX", "coeff":  0.18093119},
            {"label": "YY", "coeff":  0.18093119}
        ])
    }

    #[test]
    fn runs_h2_and_beats_hf() {
        let req = h2_request(HashMap::from([("hamiltonian".into(), h2_hamiltonian())]));
        let mut sink = CollectSink(Vec::new());
        let cancel = NeverCancelled;
        let out = VqeSolver.run(&req, &mut sink, &cancel).unwrap();
        let energy = out.energy.unwrap();
        assert!(energy < -1.84, "VQE energy {energy} should beat HF");
        assert!(out.hf_energy.unwrap() > energy, "HF should be above VQE");
        // At least one progress frame emitted, and the history is full.
        assert!(!sink.0.is_empty());
        assert!(out.convergence_history.unwrap().len() >= sink.0.len());
        assert_eq!(out.extra["n_qubits"], serde_json::json!(2));
    }

    #[test]
    fn lbfgs_optimizer_selected_and_converges() {
        let mut req = h2_request(HashMap::from([("hamiltonian".into(), h2_hamiltonian())]));
        req.solver.optimizer = Some("lbfgs".into());
        req.solver.convergence_threshold = Some(1e-10);
        let mut sink = CollectSink(Vec::new());
        let cancel = NeverCancelled;
        let out = VqeSolver.run(&req, &mut sink, &cancel).unwrap();
        let energy = out.energy.unwrap();
        // Exact parameter-shift gradients let L-BFGS reach the true H2 ground
        // state (≈ -1.857) to within chemical accuracy.
        assert!(
            energy < -1.8565,
            "L-BFGS VQE energy {energy} should reach ground state"
        );
        assert_eq!(out.extra["optimizer"], serde_json::json!("lbfgs"));
    }

    #[test]
    fn unknown_optimizer_falls_back_to_nelder_mead() {
        let mut req = h2_request(HashMap::from([("hamiltonian".into(), h2_hamiltonian())]));
        req.solver.optimizer = Some("cobyla".into());
        let mut sink = CollectSink(Vec::new());
        let cancel = NeverCancelled;
        let out = VqeSolver.run(&req, &mut sink, &cancel).unwrap();
        assert_eq!(out.extra["optimizer"], serde_json::json!("nelder_mead"));
    }

    #[test]
    fn progress_curve_is_monotone() {
        let req = h2_request(HashMap::from([("hamiltonian".into(), h2_hamiltonian())]));
        let mut sink = CollectSink(Vec::new());
        let cancel = NeverCancelled;
        VqeSolver.run(&req, &mut sink, &cancel).unwrap();
        let energies: Vec<f64> = sink.0.iter().filter_map(|p| p.energy).collect();
        for w in energies.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "progress energy must not increase");
        }
    }

    #[test]
    fn missing_hamiltonian_is_a_clear_error() {
        let req = h2_request(HashMap::new());
        let mut sink = CollectSink(Vec::new());
        let cancel = NeverCancelled;
        let err = VqeSolver.run(&req, &mut sink, &cancel).unwrap_err();
        match err {
            SolverError::Failed(m) => assert!(m.contains("hamiltonian")),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn ragged_hamiltonian_rejected() {
        let bad = serde_json::json!([
            {"label": "ZZ", "coeff": 1.0},
            {"label": "Z",  "coeff": 1.0}
        ]);
        let req = h2_request(HashMap::from([("hamiltonian".into(), bad)]));
        let mut sink = CollectSink(Vec::new());
        let cancel = NeverCancelled;
        let err = VqeSolver.run(&req, &mut sink, &cancel).unwrap_err();
        assert!(matches!(err, SolverError::Failed(_)));
    }

    #[test]
    fn cancellation_propagates() {
        struct AlwaysCancel;
        impl CancelToken for AlwaysCancel {
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        let req = h2_request(HashMap::from([("hamiltonian".into(), h2_hamiltonian())]));
        let mut sink = CollectSink(Vec::new());
        let cancel = AlwaysCancel;
        let err = VqeSolver.run(&req, &mut sink, &cancel).unwrap_err();
        assert!(matches!(err, SolverError::Cancelled));
    }
}
