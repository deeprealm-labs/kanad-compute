//! Native solver runtime.
//!
//! Phase 3.3 in progress: ships the dense statevector simulator
//! (`statevector`) and Pauli-sum Hamiltonians (`pauli`) that VQE will
//! drive. The `Solver` trait + `ProgressSink` + `CancelToken` shape was
//! locked in by Phase 3.2.

pub mod statevector;
pub mod pauli;

use kanad_protocol::{ExperimentRequest, FinalResultPayload, ProgressPayload};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SolverError {
    #[error("solver `{0}` not implemented yet")]
    NotImplemented(String),
    #[error("cancelled by user")]
    Cancelled,
    #[error("solver failed: {0}")]
    Failed(String),
}

/// Sink the runtime emits progress events into. The gateway implements
/// this against the WS outbox; tests can implement a `Vec`-collecting
/// version.
pub trait ProgressSink: Send {
    fn emit_progress(&mut self, p: ProgressPayload);
}

/// Cooperative cancel check the runtime polls between phases.
pub trait CancelToken: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

pub struct NeverCancelled;
impl CancelToken for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }
}

pub trait Solver: Send {
    fn name(&self) -> &'static str;
    fn run(
        &mut self,
        request: &ExperimentRequest,
        progress: &mut dyn ProgressSink,
        cancel: &dyn CancelToken,
    ) -> Result<FinalResultPayload, SolverError>;
}

/// Placeholder solver that always returns NotImplemented — used until
/// 3.3 lands real implementations, so dispatch wiring can be tested
/// end-to-end without solver math.
pub struct UnimplementedSolver(pub &'static str);

impl Solver for UnimplementedSolver {
    fn name(&self) -> &'static str {
        self.0
    }
    fn run(
        &mut self,
        _request: &ExperimentRequest,
        _progress: &mut dyn ProgressSink,
        _cancel: &dyn CancelToken,
    ) -> Result<FinalResultPayload, SolverError> {
        Err(SolverError::NotImplemented(self.0.into()))
    }
}
