//! Wire protocol for kanad-compute ↔ kanad-app WebSocket gateway.
//!
//! Mirror of `kanad_compute/protocol.py`. Both sides serialize / parse
//! the same JSON; the Rust side uses Serde's internally-tagged enum
//! representation with `deny_unknown_fields` everywhere so a typo
//! surfaces as a parse error instead of silently dropping a field.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

pub const PROTOCOL_VERSION: &str = "1.0";

fn major(v: &str) -> &str {
    v.split_once('.').map(|(m, _)| m).unwrap_or(v)
}

pub fn is_compatible(server_version: &str) -> bool {
    major(server_version) == major(PROTOCOL_VERSION)
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("unknown message type: {0}")]
    UnknownType(String),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("payload shape mismatch for kind {kind}: {detail}")]
    PayloadShape { kind: String, detail: String },
}

// ── Event payloads (compute → app) ─────────────────────────────────────

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "lowercase")]
pub enum LogLevel {
    Debug,
    #[default]
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogPayload {
    #[serde(default)]
    pub level: LogLevel,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgressPayload {
    pub iteration: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient_norm: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialResultPayload {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_energy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fci_energy: Option<f64>,
    #[serde(default)]
    pub fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinalResultPayload {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_energy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fci_energy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_mha: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_evaluations: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub converged: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub convergence_history: Option<Vec<HashMap<String, serde_json::Value>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_time_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_backend: Option<String>,
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorPayload {
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub traceback: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Discriminator + payload for `ExperimentEvent.payload`.
///
/// On the wire, `kind` and `payload` are two separate fields on the parent
/// `ExperimentEvent`; this enum lets Rust ensure they line up by construction.
#[derive(Debug, Clone, PartialEq)]
pub enum EventPayload {
    Log(LogPayload),
    Progress(ProgressPayload),
    PartialResult(PartialResultPayload),
    FinalResult(FinalResultPayload),
    Error(ErrorPayload),
}

impl EventPayload {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Log(_) => "Log",
            Self::Progress(_) => "Progress",
            Self::PartialResult(_) => "PartialResult",
            Self::FinalResult(_) => "FinalResult",
            Self::Error(_) => "Error",
        }
    }

    fn to_json(&self) -> Result<serde_json::Value, serde_json::Error> {
        match self {
            Self::Log(p) => serde_json::to_value(p),
            Self::Progress(p) => serde_json::to_value(p),
            Self::PartialResult(p) => serde_json::to_value(p),
            Self::FinalResult(p) => serde_json::to_value(p),
            Self::Error(p) => serde_json::to_value(p),
        }
    }

    fn from_kind_and_value(kind: &str, v: serde_json::Value) -> Result<Self, ProtocolError> {
        let parsed = match kind {
            "Log" => Self::Log(serde_json::from_value(v)?),
            "Progress" => Self::Progress(serde_json::from_value(v)?),
            "PartialResult" => Self::PartialResult(serde_json::from_value(v)?),
            "FinalResult" => Self::FinalResult(serde_json::from_value(v)?),
            "Error" => Self::Error(serde_json::from_value(v)?),
            other => return Err(ProtocolError::UnknownType(other.into())),
        };
        Ok(parsed)
    }
}

// ── Typed sub-models for ExperimentRequest ─────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Atom {
    pub symbol: String,
    pub position: [f64; 3],
}

fn default_basis() -> String {
    "sto-3g".into()
}
fn default_multiplicity() -> i32 {
    1
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoleculeSpec {
    pub atoms: Vec<Atom>,
    #[serde(default = "default_basis")]
    pub basis: String,
    #[serde(default)]
    pub charge: i32,
    #[serde(default = "default_multiplicity")]
    pub multiplicity: i32,
}

fn default_ansatz() -> String {
    "hardware_efficient".into()
}
fn default_max_iter() -> i32 {
    100
}
fn default_max_exc() -> i32 {
    5
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverSpec {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(default = "default_ansatz")]
    pub ansatz_type: String,
    #[serde(default = "default_max_iter")]
    pub max_iterations: i32,
    #[serde(default = "default_max_exc")]
    pub max_excitations: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapper_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub convergence_threshold: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_layers: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shots: Option<i64>,
    #[serde(default)]
    pub frozen_core: bool,
    #[serde(default = "default_true")]
    pub include_singles: bool,
    #[serde(default = "default_true")]
    pub include_doubles: bool,
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

// ── Top-level messages ─────────────────────────────────────────────────

fn default_protocol_version() -> String {
    PROTOCOL_VERSION.into()
}
fn default_client_version() -> String {
    "0.1.0".into()
}
fn default_server_version() -> String {
    "0.1.0".into()
}
fn default_deadline_ms() -> i64 {
    600_000
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hello {
    #[serde(default = "default_protocol_version")]
    pub protocol_version: String,
    pub node_id: String,
    #[serde(default = "default_client_version")]
    pub client_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kanad_core_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_info: Option<HashMap<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vault: Option<HashMap<String, bool>>,
    #[serde(default)]
    pub last_ack_seq: HashMap<String, i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Registered {
    #[serde(default = "default_protocol_version")]
    pub protocol_version: String,
    pub node_id: String,
    pub session_id: String,
    #[serde(default = "default_server_version")]
    pub server_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentRequest {
    pub experiment_id: String,
    pub user_id: String,
    pub molecule: MoleculeSpec,
    pub solver: SolverSpec,
    pub backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_credentials: Option<HashMap<String, String>>,
    #[serde(default = "default_deadline_ms")]
    pub deadline_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CancelExperiment {
    pub experiment_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ack {
    pub experiment_id: String,
    pub last_seq: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ping {
    pub ts_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pong {
    pub ts_ms: i64,
}

/// `ExperimentEvent` is special: `kind` and `payload` are split fields on
/// the wire, but they must agree. We round-trip through a private "raw"
/// struct so callers see the typed `EventPayload`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentEvent {
    pub experiment_id: String,
    pub seq: i64,
    pub ts_ms: i64,
    pub payload: EventPayload,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawExperimentEvent {
    #[serde(rename = "type")]
    type_: String,
    experiment_id: String,
    seq: i64,
    ts_ms: i64,
    kind: String,
    payload: serde_json::Value,
}

impl Serialize for ExperimentEvent {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let raw = RawExperimentEvent {
            type_: "ExperimentEvent".into(),
            experiment_id: self.experiment_id.clone(),
            seq: self.seq,
            ts_ms: self.ts_ms,
            kind: self.payload.kind().into(),
            payload: self.payload.to_json().map_err(serde::ser::Error::custom)?,
        };
        raw.serialize(s)
    }
}

impl<'de> Deserialize<'de> for ExperimentEvent {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = RawExperimentEvent::deserialize(d)?;
        if raw.type_ != "ExperimentEvent" {
            return Err(serde::de::Error::custom(format!(
                "expected type=ExperimentEvent, got {}",
                raw.type_
            )));
        }
        let payload = EventPayload::from_kind_and_value(&raw.kind, raw.payload)
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            experiment_id: raw.experiment_id,
            seq: raw.seq,
            ts_ms: raw.ts_ms,
            payload,
        })
    }
}

// ── Discriminated unions ───────────────────────────────────────────────

/// Messages the server (kanad-app) sends down the WS.
///
/// `ExperimentRequest` carries a `MoleculeSpec` whose atom list can grow,
/// so we box it to keep the enum compact for the other variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    Registered(Registered),
    ExperimentRequest(Box<ExperimentRequest>),
    CancelExperiment(CancelExperiment),
    Ack(Ack),
    Ping(Ping),
    Pong(Pong),
}

/// Messages the client (kanad-compute) sends up the WS.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientMessage {
    Hello(Hello),
    ExperimentEvent(ExperimentEvent),
    Ping(Ping),
    Pong(Pong),
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum ClientMessageWire {
    Hello(Hello),
    ExperimentEvent(ExperimentEvent),
    Ping(Ping),
    Pong(Pong),
}

impl Serialize for ClientMessage {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let wire = match self {
            Self::Hello(h) => ClientMessageWire::Hello(h.clone()),
            Self::ExperimentEvent(e) => ClientMessageWire::ExperimentEvent(e.clone()),
            Self::Ping(p) => ClientMessageWire::Ping(p.clone()),
            Self::Pong(p) => ClientMessageWire::Pong(p.clone()),
        };
        wire.serialize(s)
    }
}

impl<'de> Deserialize<'de> for ClientMessage {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let wire = ClientMessageWire::deserialize(d)?;
        Ok(match wire {
            ClientMessageWire::Hello(h) => Self::Hello(h),
            ClientMessageWire::ExperimentEvent(e) => Self::ExperimentEvent(e),
            ClientMessageWire::Ping(p) => Self::Ping(p),
            ClientMessageWire::Pong(p) => Self::Pong(p),
        })
    }
}

pub fn parse_server_message(raw: &str) -> Result<ServerMessage, ProtocolError> {
    Ok(serde_json::from_str(raw)?)
}

pub fn parse_client_message(raw: &str) -> Result<ClientMessage, ProtocolError> {
    Ok(serde_json::from_str(raw)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_compat() {
        assert!(is_compatible("1.0"));
        assert!(is_compatible("1.99"));
        assert!(!is_compatible("2.0"));
        assert!(!is_compatible("garbage"));
    }

    #[test]
    fn hello_roundtrip() {
        let h = Hello {
            protocol_version: "1.0".into(),
            node_id: "node-1".into(),
            client_version: "0.1.0".into(),
            kanad_core_version: Some("0.3.2".into()),
            system_info: None,
            vault: Some(HashMap::from([("ibm".into(), true)])),
            last_ack_seq: HashMap::from([("exp-1".into(), 42)]),
        };
        let msg = ClientMessage::Hello(h.clone());
        let s = serde_json::to_string(&msg).unwrap();
        assert!(s.contains("\"type\":\"Hello\""));
        let back: ClientMessage = serde_json::from_str(&s).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn experiment_event_progress_roundtrip() {
        let ev = ExperimentEvent {
            experiment_id: "exp-1".into(),
            seq: 7,
            ts_ms: 1_700_000_000_000,
            payload: EventPayload::Progress(ProgressPayload {
                iteration: 3,
                total: Some(100),
                energy: Some(-1.137),
                gradient_norm: Some(1e-3),
                message: Some("iter 3".into()),
            }),
        };
        let s = serde_json::to_string(&ev).unwrap();
        assert!(s.contains("\"kind\":\"Progress\""));
        let back: ExperimentEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(back, ev);
    }

    #[test]
    fn experiment_event_rejects_unknown_kind() {
        let raw = r#"{
            "type":"ExperimentEvent","experiment_id":"e","seq":1,"ts_ms":0,
            "kind":"NotAKind","payload":{}
        }"#;
        let err = serde_json::from_str::<ExperimentEvent>(raw).unwrap_err();
        assert!(err.to_string().contains("NotAKind"));
    }

    #[test]
    fn server_message_dispatch() {
        let raw = r#"{"type":"Ack","experiment_id":"e","last_seq":5}"#;
        let msg = parse_server_message(raw).unwrap();
        match msg {
            ServerMessage::Ack(a) => {
                assert_eq!(a.experiment_id, "e");
                assert_eq!(a.last_seq, 5);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn unknown_field_rejected() {
        let raw = r#"{"type":"Ping","ts_ms":1,"bogus":42}"#;
        assert!(parse_server_message(raw).is_err());
    }

    #[test]
    fn experiment_request_full_payload() {
        let raw = r#"{
          "type":"ExperimentRequest",
          "experiment_id":"e1",
          "user_id":"u1",
          "molecule":{"atoms":[{"symbol":"H","position":[0.0,0.0,0.0]},
                                {"symbol":"H","position":[0.0,0.0,0.74]}]},
          "solver":{"type":"vqe","max_iterations":50},
          "backend":"kanad_compute"
        }"#;
        let msg = parse_server_message(raw).unwrap();
        match msg {
            ServerMessage::ExperimentRequest(boxed) => {
                let r = *boxed;
                assert_eq!(r.molecule.atoms.len(), 2);
                assert_eq!(r.molecule.basis, "sto-3g"); // default
                assert_eq!(r.solver.type_, "vqe");
                assert_eq!(r.solver.max_iterations, 50);
                assert_eq!(r.solver.ansatz_type, "hardware_efficient");
                assert_eq!(r.deadline_ms, 600_000);
            }
            _ => panic!("wrong variant"),
        }
    }
}
