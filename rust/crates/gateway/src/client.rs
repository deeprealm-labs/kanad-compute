//! Async WebSocket gateway client — Rust port of `ComputeWSClient`.
//!
//! Connect lifetime:
//!   1. Open ws(s)://…/api/compute/connect with `Authorization: Bearer …`.
//!   2. Send Hello carrying `last_ack_seq` from `SeqState`.
//!   3. Receive Registered, verify protocol-version compatibility.
//!   4. Replay every row in the outbox.
//!   5. Run three concurrent tasks: reader (server → us), writer
//!      (mpsc → ws sink), and ping watchdog. Any of them returning
//!      causes the connection to tear down; the outer reconnect loop
//!      reopens with `Backoff`.

use crate::backoff::Backoff;
use crate::outbox::Outbox;
use crate::seq_state::SeqState;
use anyhow::{anyhow, Context};
use futures_util::stream::SplitStream;
use futures_util::{SinkExt, StreamExt};
use http::header::{AUTHORIZATION, HOST};
use kanad_protocol::{
    is_compatible, parse_server_message, Ack, CancelExperiment, ErrorPayload, EventPayload,
    ExperimentEvent, ExperimentRequest, Hello, LogLevel, LogPayload, ProgressPayload,
    ServerMessage, PROTOCOL_VERSION,
};
use kanad_runtime::{CancelToken, ProgressSink, Solver, SolverError};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio::time::{interval, timeout};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

pub const PING_INTERVAL: Duration = Duration::from_secs(15);
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
pub const PONG_GRACE_MULTIPLIER: u32 = 2; // 2 missed pongs → drop

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;
type WsReader = SplitStream<WsStream>;

/// Factory that produces a solver for a given solver name (`vqe`, `sqd`, …).
/// Returns `None` if the solver isn't registered — callers emit an Error.
pub type SolverFactory = Arc<dyn Fn(&str) -> Option<Box<dyn Solver>> + Send + Sync>;

#[derive(Clone)]
pub struct ClientConfig {
    pub kanad_url: String,
    pub api_key: String,
    pub node_id: String,
    pub client_version: String,
    pub kanad_core_version: Option<String>,
    pub state_dir: PathBuf,
}

impl ClientConfig {
    pub fn new(
        kanad_url: impl Into<String>,
        api_key: impl Into<String>,
        node_id: impl Into<String>,
    ) -> Self {
        Self {
            kanad_url: kanad_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            node_id: node_id.into(),
            client_version: env!("CARGO_PKG_VERSION").into(),
            kanad_core_version: None,
            state_dir: default_state_dir(),
        }
    }

    pub fn ws_url(&self) -> String {
        let u = self.kanad_url.as_str();
        if let Some(rest) = u.strip_prefix("https://") {
            format!("wss://{rest}/api/compute/connect")
        } else if let Some(rest) = u.strip_prefix("http://") {
            format!("ws://{rest}/api/compute/connect")
        } else {
            format!("{u}/api/compute/connect")
        }
    }
}

fn default_state_dir() -> PathBuf {
    if let Some(home) = dirs_home() {
        home.join(".kanad-compute").join("state")
    } else {
        PathBuf::from(".kanad-compute-state")
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

pub struct GatewayClient {
    pub config: ClientConfig,
    outbox: Arc<Outbox>,
    seq_state: Arc<SeqState>,
    cancelled: Arc<RwLock<HashSet<String>>>,
    solver_factory: SolverFactory,
}

impl GatewayClient {
    pub fn new(config: ClientConfig, solver_factory: SolverFactory) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&config.state_dir)
            .with_context(|| format!("create state dir {}", config.state_dir.display()))?;
        let outbox = Arc::new(Outbox::open(config.state_dir.join("outbox.db"))?);
        let seq_state = Arc::new(SeqState::open(crate::seq_state::path_in(
            &config.state_dir,
        ))?);
        // Opportunistic 24h cleanup at startup.
        let _ = outbox.gc(crate::outbox::GC_DEFAULT_AGE);
        Ok(Self {
            config,
            outbox,
            seq_state,
            cancelled: Arc::new(RwLock::new(HashSet::new())),
            solver_factory,
        })
    }

    /// Run the reconnect loop forever. Returns only on unrecoverable error
    /// (e.g. authentication-rejected close codes, which we still surface as
    /// a regular reconnect today — TODO when device-revoke ships).
    pub async fn run_forever(&self) -> anyhow::Result<()> {
        let mut backoff = Backoff::new();
        loop {
            match self.connect_once().await {
                Ok(()) => {
                    tracing::info!("gateway: connection closed cleanly; reconnecting");
                    backoff.reset();
                }
                Err(e) => {
                    let delay = backoff.next_delay();
                    tracing::warn!(error = %e, ?delay, "gateway: connection error; reconnecting");
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    /// One connection lifetime: handshake → replay → run until disconnect.
    pub async fn connect_once(&self) -> anyhow::Result<()> {
        let url = self.config.ws_url();
        tracing::info!(%url, "gateway: connecting");

        let req = build_request(&url, &self.config.api_key)?;
        let (ws, _resp) = tokio_tungstenite::connect_async(req)
            .await
            .context("ws connect")?;
        let (mut sink, mut reader) = ws.split();

        // ── handshake ────────────────────────────────────────────────────
        let hello = self.build_hello();
        let frame = serde_json::to_string(&kanad_protocol::ClientMessage::Hello(hello))?;
        sink.send(Message::Text(frame)).await?;

        let registered = match timeout(HANDSHAKE_TIMEOUT, reader.next()).await {
            Ok(Some(Ok(Message::Text(t)))) => parse_server_message(&t)?,
            Ok(Some(Ok(Message::Binary(b)))) => {
                let t = std::str::from_utf8(&b)
                    .map_err(|e| anyhow!("binary handshake frame not utf8: {e}"))?;
                parse_server_message(t)?
            }
            Ok(Some(Ok(other))) => {
                return Err(anyhow!("unexpected handshake frame: {other:?}"));
            }
            Ok(Some(Err(e))) => return Err(anyhow!("ws read during handshake: {e}")),
            Ok(None) => return Err(anyhow!("ws closed before Registered")),
            Err(_) => return Err(anyhow!("handshake timed out after {HANDSHAKE_TIMEOUT:?}")),
        };
        let registered = match registered {
            ServerMessage::Registered(r) => r,
            other => return Err(anyhow!("expected Registered, got {other:?}")),
        };
        if !is_compatible(&registered.protocol_version) {
            return Err(anyhow!(
                "protocol version mismatch: server={}, client={PROTOCOL_VERSION}",
                registered.protocol_version
            ));
        }
        tracing::info!(
            session_id = %registered.session_id,
            server_proto = %registered.protocol_version,
            "gateway: registered"
        );

        // ── writer task (drains mpsc → ws sink) ──────────────────────────
        let (tx, mut rx) = mpsc::unbounded_channel::<Message>();
        let mut tasks = JoinSet::new();
        tasks.spawn(async move {
            while let Some(msg) = rx.recv().await {
                if sink.send(msg).await.is_err() {
                    break;
                }
            }
            // Sink dropped here closes the write half.
            let _ = sink.close().await;
        });

        // ── replay unacked rows from the outbox ──────────────────────────
        self.replay_outbox(&tx).await?;

        // ── ping watchdog ────────────────────────────────────────────────
        let last_pong = Arc::new(AtomicI64::new(now_ms()));
        {
            let tx = tx.clone();
            let last_pong = last_pong.clone();
            tasks.spawn(async move {
                let mut tick = interval(PING_INTERVAL);
                tick.tick().await; // skip immediate fire
                loop {
                    tick.tick().await;
                    let last = last_pong.load(Ordering::Relaxed);
                    let now = now_ms();
                    let deadline = PING_INTERVAL.as_millis() as i64 * PONG_GRACE_MULTIPLIER as i64;
                    if last > 0 && (now - last) > deadline {
                        tracing::warn!(
                            stale_ms = now - last,
                            "gateway: no pong within deadline; dropping"
                        );
                        return;
                    }
                    let ping =
                        kanad_protocol::ClientMessage::Ping(kanad_protocol::Ping { ts_ms: now });
                    let frame = match serde_json::to_string(&ping) {
                        Ok(s) => s,
                        Err(_) => return,
                    };
                    if tx.send(Message::Text(frame)).is_err() {
                        return;
                    }
                }
            });
        }

        // ── reader loop ──────────────────────────────────────────────────
        self.read_loop(&mut reader, tx, last_pong).await?;

        // Reader returned → close everything else.
        tasks.shutdown().await;
        Ok(())
    }

    fn build_hello(&self) -> Hello {
        Hello {
            protocol_version: PROTOCOL_VERSION.into(),
            node_id: self.config.node_id.clone(),
            client_version: self.config.client_version.clone(),
            kanad_core_version: self.config.kanad_core_version.clone(),
            system_info: None,
            vault: None,
            last_ack_seq: self.seq_state.last_ack_seq_map(),
        }
    }

    async fn replay_outbox(&self, tx: &mpsc::UnboundedSender<Message>) -> anyhow::Result<()> {
        let rows = self.outbox.pending()?;
        if rows.is_empty() {
            return Ok(());
        }
        tracing::info!(count = rows.len(), "gateway: replaying unacked events");
        for row in rows {
            // The row's payload_json is the *full ExperimentEvent JSON*
            // — see emit() below. Send it verbatim.
            tx.send(Message::Text(row.payload_json))
                .map_err(|_| anyhow!("writer channel closed during replay"))?;
        }
        Ok(())
    }

    async fn read_loop(
        &self,
        reader: &mut WsReader,
        tx: mpsc::UnboundedSender<Message>,
        last_pong: Arc<AtomicI64>,
    ) -> anyhow::Result<()> {
        while let Some(frame) = reader.next().await {
            let frame = frame.context("ws read")?;
            let text = match frame {
                Message::Text(t) => t.to_string(),
                Message::Binary(b) => match std::str::from_utf8(&b) {
                    Ok(s) => s.to_owned(),
                    Err(e) => {
                        tracing::warn!(%e, "gateway: dropping non-utf8 binary frame");
                        continue;
                    }
                },
                Message::Ping(payload) => {
                    let _ = tx.send(Message::Pong(payload));
                    continue;
                }
                Message::Pong(_) => {
                    last_pong.store(now_ms(), Ordering::Relaxed);
                    continue;
                }
                Message::Close(_) => return Ok(()),
                Message::Frame(_) => continue,
            };

            let msg = match parse_server_message(&text) {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!(%e, raw = %text, "gateway: invalid server message");
                    continue;
                }
            };

            match msg {
                ServerMessage::Ping(p) => {
                    let pong = kanad_protocol::ClientMessage::Pong(kanad_protocol::Pong {
                        ts_ms: p.ts_ms,
                    });
                    let frame = serde_json::to_string(&pong)?;
                    let _ = tx.send(Message::Text(frame));
                }
                ServerMessage::Pong(_) => {
                    last_pong.store(now_ms(), Ordering::Relaxed);
                }
                ServerMessage::Ack(ack) => self.handle_ack(ack),
                ServerMessage::CancelExperiment(c) => self.handle_cancel(c),
                ServerMessage::ExperimentRequest(req) => {
                    let req = *req;
                    let tx = tx.clone();
                    let outbox = self.outbox.clone();
                    let seq_state = self.seq_state.clone();
                    let cancelled = self.cancelled.clone();
                    let factory = self.solver_factory.clone();
                    tokio::spawn(async move {
                        handle_experiment(req, tx, outbox, seq_state, cancelled, factory).await;
                    });
                }
                ServerMessage::Registered(_) => {
                    tracing::warn!("gateway: unexpected second Registered frame; ignoring");
                }
            }
        }
        Ok(())
    }

    fn handle_ack(&self, ack: Ack) {
        if let Err(e) = self.outbox.ack(&ack.experiment_id, ack.last_seq) {
            tracing::warn!(%e, "outbox.ack failed");
        }
        if self.seq_state.record_ack(&ack.experiment_id, ack.last_seq) {
            if let Err(e) = self.seq_state.save() {
                tracing::warn!(%e, "seq_state.save failed");
            }
        }
    }

    fn handle_cancel(&self, c: CancelExperiment) {
        self.cancelled.write().insert(c.experiment_id);
    }
}

fn build_request(url: &str, api_key: &str) -> anyhow::Result<http::Request<()>> {
    let mut req = url.into_client_request().context("parse ws url")?;
    let host = req
        .uri()
        .host()
        .ok_or_else(|| anyhow!("ws url missing host"))?
        .to_owned();
    let bearer = format!("Bearer {api_key}");
    req.headers_mut()
        .insert(AUTHORIZATION, bearer.parse().context("auth header")?);
    if !req.headers().contains_key(HOST) {
        req.headers_mut()
            .insert(HOST, host.parse().context("host header")?);
    }
    Ok(req)
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

// ── Experiment dispatch ────────────────────────────────────────────────

/// Per-task state for a single experiment.
async fn handle_experiment(
    req: ExperimentRequest,
    tx: mpsc::UnboundedSender<Message>,
    outbox: Arc<Outbox>,
    seq_state: Arc<SeqState>,
    cancelled: Arc<RwLock<HashSet<String>>>,
    factory: SolverFactory,
) {
    let exp_id = req.experiment_id.clone();
    let solver_name = req.solver.type_.clone();
    let backend = req.backend.clone();

    let _ = emit(
        &exp_id,
        EventPayload::Log(LogPayload {
            level: LogLevel::Info,
            message: format!("Starting {solver_name} on {backend}"),
            detail: None,
        }),
        &tx,
        &outbox,
        &seq_state,
    );

    let solver = match factory(&solver_name) {
        Some(s) => s,
        None => {
            let _ = emit(
                &exp_id,
                EventPayload::Error(ErrorPayload {
                    message: format!("solver `{solver_name}` not registered"),
                    traceback: None,
                    code: Some("not_implemented".into()),
                }),
                &tx,
                &outbox,
                &seq_state,
            );
            return;
        }
    };

    // Solvers are synchronous Rust code (Phase 3.3); run them on a blocking
    // worker so they don't stall the async runtime. We bridge progress
    // events back through the mpsc/outbox plumbing.
    let cancel = SetCancelToken {
        experiment_id: exp_id.clone(),
        set: cancelled.clone(),
    };
    let tx_for_progress = tx.clone();
    let outbox_for_progress = outbox.clone();
    let seq_state_for_progress = seq_state.clone();
    let exp_for_progress = exp_id.clone();

    let result = tokio::task::spawn_blocking(move || {
        let mut sink = ChannelSink {
            experiment_id: exp_for_progress,
            tx: tx_for_progress,
            outbox: outbox_for_progress,
            seq_state: seq_state_for_progress,
        };
        let mut solver = solver;
        solver.run(&req, &mut sink, &cancel)
    })
    .await;

    match result {
        Ok(Ok(final_payload)) => {
            let _ = emit(
                &exp_id,
                EventPayload::FinalResult(final_payload),
                &tx,
                &outbox,
                &seq_state,
            );
        }
        Ok(Err(SolverError::Cancelled)) => {
            cancelled.write().remove(&exp_id);
            let _ = emit(
                &exp_id,
                EventPayload::Error(ErrorPayload {
                    message: "Cancelled by user".into(),
                    traceback: None,
                    code: Some("cancelled".into()),
                }),
                &tx,
                &outbox,
                &seq_state,
            );
        }
        Ok(Err(e)) => {
            let code = match &e {
                SolverError::NotImplemented(_) => "not_implemented",
                SolverError::Failed(_) => "failed",
                SolverError::Cancelled => "cancelled",
            };
            let _ = emit(
                &exp_id,
                EventPayload::Error(ErrorPayload {
                    message: e.to_string(),
                    traceback: None,
                    code: Some(code.into()),
                }),
                &tx,
                &outbox,
                &seq_state,
            );
        }
        Err(join_err) => {
            let _ = emit(
                &exp_id,
                EventPayload::Error(ErrorPayload {
                    message: format!("solver task panicked: {join_err}"),
                    traceback: None,
                    code: Some("panic".into()),
                }),
                &tx,
                &outbox,
                &seq_state,
            );
        }
    }
}

struct SetCancelToken {
    experiment_id: String,
    set: Arc<RwLock<HashSet<String>>>,
}

impl CancelToken for SetCancelToken {
    fn is_cancelled(&self) -> bool {
        self.set.read().contains(&self.experiment_id)
    }
}

struct ChannelSink {
    experiment_id: String,
    tx: mpsc::UnboundedSender<Message>,
    outbox: Arc<Outbox>,
    seq_state: Arc<SeqState>,
}

impl ProgressSink for ChannelSink {
    fn emit_progress(&mut self, p: ProgressPayload) {
        let _ = emit(
            &self.experiment_id,
            EventPayload::Progress(p),
            &self.tx,
            &self.outbox,
            &self.seq_state,
        );
    }
}

/// Persist event to the outbox THEN push to the writer channel. Mirrors
/// the Python `_emit`: record-before-send is what gives crash resilience.
fn emit(
    experiment_id: &str,
    payload: EventPayload,
    tx: &mpsc::UnboundedSender<Message>,
    outbox: &Outbox,
    seq_state: &SeqState,
) -> anyhow::Result<()> {
    let seq = seq_state.next_seq(experiment_id);
    let ev = ExperimentEvent {
        experiment_id: experiment_id.to_owned(),
        seq,
        ts_ms: now_ms(),
        payload,
    };
    let kind = ev.payload.kind().to_owned();
    let frame = serde_json::to_string(&ev)?;
    outbox.record(experiment_id, seq, &kind, &frame)?;
    if tx.send(Message::Text(frame)).is_err() {
        return Err(anyhow!("writer channel closed"));
    }
    Ok(())
}

/// Convenience factory that always returns an `UnimplementedSolver`. Used
/// in tests and as the default before Phase 3.3 wires real solvers in.
pub fn unimplemented_factory() -> SolverFactory {
    Arc::new(|name: &str| -> Option<Box<dyn Solver>> {
        // Leak the &str into a 'static lifetime for the error message —
        // safe because we own the string for the lifetime of the solver.
        let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
        Some(Box::new(kanad_runtime::UnimplementedSolver(leaked)))
    })
}

/// Production factory: dispatches solver names to native Tier-1 solvers,
/// falling back to `UnimplementedSolver` (which surfaces a clean
/// `not_implemented` Error event) for anything not yet ported.
pub fn default_factory() -> SolverFactory {
    Arc::new(|name: &str| -> Option<Box<dyn Solver>> {
        match name {
            "vqe" => Some(Box::new(kanad_runtime::VqeSolver)),
            other => {
                let leaked: &'static str = Box::leak(other.to_owned().into_boxed_str());
                Some(Box::new(kanad_runtime::UnimplementedSolver(leaked)))
            }
        }
    })
}
