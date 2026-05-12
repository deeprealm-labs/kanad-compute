//! Outbound WebSocket gateway client.
//!
//! Initial Phase-3.1 scope: the durable outbox + reconnect-backoff
//! primitives, both proven against unit tests, plus the protocol-level
//! send/recv loop scaffolded around a `Solver` trait that the runtime
//! crate will plug into in Phase 3.3.

pub mod backoff;
pub mod outbox;

pub use backoff::Backoff;
pub use outbox::{Outbox, OutboxRow};
