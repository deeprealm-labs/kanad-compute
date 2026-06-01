//! Outbound WebSocket gateway client.
//!
//! Phase 3.2 ships the full reconnect loop, handshake, outbox-backed emit,
//! cancel handling, and ping/pong watchdog. Solver execution is plugged in
//! through the `Solver` trait from `kanad-runtime`; Phase 3.3 supplies
//! real implementations.

pub mod backoff;
pub mod client;
pub mod outbox;
pub mod seq_state;

pub use backoff::Backoff;
pub use client::{
    default_factory, unimplemented_factory, ClientConfig, GatewayClient, SolverFactory,
};
pub use outbox::{Outbox, OutboxRow};
pub use seq_state::SeqState;
