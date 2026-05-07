# kanad-compute — Master TODO

Living plan for the architecture migration tracked by [`mk0dz/kanad-app#16`](https://github.com/mk0dz/kanad-app/issues/16):
*"New kanad compute for entire backend is needed"*.

The thesis of issue #16: move execution, credentials, logging, and (eventually) the compute runtime out of the cloud (`kanad-app`) and into a user-controlled local binary (`kanad-compute`). The cloud reduces to molecule-design UX, auth/billing, history, and a thin dispatch/proxy layer. Browsers either talk **directly** to localhost (Path A) or are **proxied** through the cloud (Path B). The compute node owns the calculations and the credentials.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked / decision needed

---

## 0. Architecture north star

- **Two repos, two responsibilities.**
  - `kanad-app` (cloud, Next.js + FastAPI): UX, auth, history, dispatch, browser fan-out. NO solver code, NO user backend credentials at rest.
  - `kanad-compute` (local, Python today → Rust tomorrow): solvers, vault, telemetry, persistent outbound WS to the cloud.
- **Wire format:** JSON over WebSocket for v1 (browser-friendly, proxy-friendly, debuggable). Optional Protobuf-over-WS in v2 once the schema is stable.
- **Topology:** persistent outbound WS from compute → app (NAT-friendly, no inbound port), and from browser → app (existing) or browser → compute direct (Path A).
- **Credentials:** OS keyring (Keychain / Credential Manager / secret-service) primary; AES-256-GCM + Argon2id encrypted file fallback. **Never stored in cloud DB.**
- **Auth:** RFC 8628 device authorization grant (Phase 2) — same UX as `gh auth login` / `claude login`. Bearer API key today.
- **Solver tiering (Rust migration):**
  - Tier 1 (native Rust): statevector, basic VQE, PhysicsVQE, HardwareVQE.
  - Tier 2 (PyO3-wrapped Python): SQD, qEOM, advanced ansätze.
  - Tier 3 (cloud delegate): anything that genuinely needs the cluster.

---

## 1. Phase 1 — WebSocket gateway (Python compute, replace polling)

Goal: kill the 2-second `/api/compute/jobs` polling loop in `remote_worker.py`. Compute holds one persistent outbound WS to the cloud. Cloud pushes experiments down, compute streams events back, cloud fans out to the browser via the existing `connection_manager`.

### 1.1 Wire protocol (Pydantic)
- [x] `kanad-app/api/protocol.py` — server-side schema
- [x] `kanad-compute/kanad_compute/protocol.py` — client-side mirror
- [x] Discriminated unions: `Hello | Registered | ExperimentRequest | CancelExperiment | Ack | Ping | Pong` (server) and `Hello | ExperimentEvent | Ping | Pong` (client)
- [x] `ExperimentEvent` carries `experiment_id`, monotonic `seq`, `ts_ms`, `kind ∈ {Log, Progress, PartialResult, FinalResult, Error}`, `payload` (validated against `kind`)
- [x] Versioning: `protocol_version` in `Hello`/`Registered`; server rejects MAJOR mismatch with close 1003
- [x] `Ack` server→client message carrying `last_seq` per experiment (drives outbox trim on the client)
- [x] `model_config = ConfigDict(extra="forbid")` on every message — typos surface as parse errors
- [x] Typed `MoleculeSpec` / `SolverSpec` / `Atom` sub-models on `ExperimentRequest`

### 1.2 Server endpoint (kanad-app)
- [x] `api/routes/compute_ws.py` — `WebSocket /api/compute/connect`
- [x] `NodeSession` dataclass + in-memory `NODE_REGISTRY: dict[user_id, NodeSession]`
- [x] Bearer-token auth resolving `User.kanad_compute_key → user_id`
- [x] Handshake (`Hello` first, then `Registered`), 15-s `Ping`, 2-miss disconnect
- [x] Public API `dispatch_experiment_to_user(user_id, ExperimentRequest) -> bool` and `dispatch_cancel_to_user`
- [x] Per-experiment `seq` dedup; fan-out via `connection_manager.send_to_job(experiment_id, ...)` with lowercased `type`
- [x] Terminal-event persistence via centralized `transition_job` (race-safe vs CANCELLED)
- [x] Router registered in `api/main.py`
- [x] Resume on reconnect: honour `Hello.last_ack_seq`; suppress already-delivered events
- [x] `Ack` sent back to client on every successful event handle (drives client outbox trim)
- [x] Metrics: `/api/admin/compute_status` returns `{user_id, node_id, session_id, connected_since, last_rtt_ms, in_flight_count}`
- [x] Single-worker startup warning when `WEB_CONCURRENCY > 1`
- [x] Fan-out failures wrapped — dead browser doesn't kill the WS handler
- [ ] Multi-node-per-user support (Phase 2 with device-token migration; current behaviour: bump previous session)

### 1.3 Client (kanad-compute)
- [x] `kanad_compute/ws_client.py` — `ComputeWSClient` with reconnect (1 s → 30 s exponential backoff, ±20 % jitter)
- [x] Hello → Registered handshake with `protocol_version` check; `Authorization: Bearer …` header
- [x] `_handle_experiment` consumes typed `MoleculeSpec` / `SolverSpec`, runs `worker.run_calculation` in `ThreadPoolExecutor`
- [x] Phase-1 events: `Log` start + `FinalResult` / `Error` / cancelled
- [x] `_send` lock for atomic frame writes; `_emit` increments per-experiment seq
- [x] `_ping_loop` 15 s; URL translation `http(s)://` → `ws(s)://`
- [x] `kanad-compute connect [--url URL]` CLI subcommand
- [x] `websockets>=12.0`, `pytest`, `pytest-asyncio` in `pyproject.toml`
- [x] Persist `last_ack_seq` to `~/.kanad-compute/state/seq.json` (atomic tmpfile+rename); populates `Hello.last_ack_seq` on reconnect
- [x] In-memory ring buffer of unacked events (RING_MAX=100), replayed on reconnect, trimmed on `Ack`
- [x] Backpressure: drop oldest non-terminal events; `FinalResult`/`Error` never dropped
- [x] Honour `CancelExperiment` via cooperative `cancel_check` callable threaded through `worker.run_calculation`
- [x] 22 pytest tests including a real WS server smoke test

### 1.4 Dispatch hook (kanad-app `calculations.py`)
- [x] New module `api/routes/_compute_dispatch.py` with `_build_experiment_request` + `_try_ws_dispatch`
- [x] Edits `submit_calculation` (L474-489): WS dispatch first, polling-task fallback if no live session
- [x] On success: `Job` transitions PENDING → RUNNING via `transition_job` in a separate `get_db_context()` write
- [x] Credentials resolved via existing `_resolve_credential` (Phase 2 will move them into the local vault)
- [x] py_compile clean on every touched file; pytest covers dispatch + persist + dedup + cancel race
- [ ] **Manual end-to-end** per `kanad-app/docs/compute_ws_smoke.md` — needs a running Postgres + a real molecule; not yet executed by this session

### 1.5 Graceful coexistence with old polling path
- [ ] Keep `/api/compute/jobs` polling endpoints alive throughout Phase 1 — old `remote_worker.py` must still function for users on stale `kanad-compute` versions
- [ ] Add deprecation log in the polling handlers
- [ ] Decide cutoff version: drop polling N minor releases after WS lands

---

## 2. Phase 2 — Live progress, outbox, device-token auth

### 2.1 Solver callback hooks (live `Progress` events) — Phase 2.1a (VQE + SQD)
- [x] Survey existing solvers: VQESolver and SQDSolver already accept `solve(callback=...)` kwargs in kanad-core; PhysicsVQE / HardwareVQE / VarQITE / qEOM / EfficientVQE do not (carved into 2.1b)
- [x] Protocol: `ProgressPayload.note` → `message`; new optional `gradient_norm` field
- [x] Thread `progress_cb` kwarg through `worker.run_calculation` and into `_run_vqe` / `_run_sqd`
- [x] Compute-side: `_make_progress_cb` factory bridges worker thread → asyncio loop via `run_coroutine_threadsafe`; throttle floor `PROGRESS_MIN_INTERVAL_MS=100`; energy-delta bypass `PROGRESS_ENERGY_DELTA=1e-4 Ha`; terminal flush before `FinalResult`
- [x] Front-end already renders `progress` messages via `ExperimentMonitor.tsx` — no UI change required
- [x] 28 (compute) + 15 (app) = 43 tests passing

### 2.1b Solver callback hooks — remaining solvers (kanad-core PR)
- [x] Adapter scaffolding in `kanad_compute/worker.py`: `_adapt_generic_progress` + `progress_cb` plumbed through every `_run_*` helper. Today these no-op (TypeError swallowed by `_solve_with_optional_callback`); they start producing Progress events automatically once kanad-core lands the kwarg
- [ ] kanad-core PR: add `callback` kwarg to PhysicsVQE.solve()
- [ ] kanad-core PR: add `callback` kwarg to HardwareVQE.solve_local() / solve_hardware() (positional `callback=cb` already attempted with TypeError fallback)
- [ ] kanad-core PR: add `callback` kwarg to VarQITESolver.solve() (per-step or every-N-steps)
- [ ] kanad-core PR: add `callback` kwarg to qEOMVQE.solve() (delegate to underlying VQE)
- [ ] kanad-core PR: expose internal callback as user-configurable param on EfficientVQE.solve()
- [ ] Tier requirement: kanad-core ≥ X.Y to unlock per-iteration progress for these solvers

### 2.2 SQLite outbox (crash-resilient delivery)
- [x] `kanad_compute/outbox.py` — `Outbox(path)` with WAL journaling, threading lock, `record / ack / pending / pending_count / gc / close`
- [x] `_emit`: insert THEN send (durable on disk before going on the wire)
- [x] `_handle_ack`: `outbox.ack(exp_id, last_seq)`; `last_ack_seq` mirrored to seq.json so Hello populates fast
- [x] `_replay_unacked`: drains `outbox.pending()` ordered by `(exp_id, seq)` after Registered
- [x] Retention: `outbox.gc(older_than_seconds=86400)` runs once at startup; defensive cleanup against runaway state
- [x] Replaces the in-memory ring buffer entirely; `_ExperimentBuffer.unacked` removed; `_evict_old` removed
- [x] 7 outbox tests (durability, concurrent record, ack semantics, gc, ordering)

### 2.3 RFC 8628 device authorization grant
- [ ] `POST /api/auth/device/code` → returns `device_code, user_code, verification_uri, interval, expires_in`
- [ ] `POST /api/auth/device/token` → polled by client; returns `{access_token, refresh_token}` once user approves
- [ ] Browser flow at `/connect-device` showing `user_code` + Approve button (gated by login)
- [ ] CLI: `kanad-compute login` opens browser, polls `/token`, persists tokens to OS keyring
- [ ] Migrate WS auth from raw `User.kanad_compute_key` to short-lived JWT access_token + refresh
- [ ] Revocation: dashboard page listing connected devices with "revoke" button

### 2.4 Local credential vault
- [x] `kanad_compute/vault.py` wraps `keyring` (OS keychain on macOS / Credential Manager on Windows / Secret Service on Linux). API: `set/get/has/clear/status/all/list_present`
- [x] CLI: `kanad-compute creds set/get/list/clear` (Click subgroup); reveal-flag controls full-token print vs `****<last4>` redaction
- [x] Vault state surfaced in `Hello.vault` from real keyring presence (config-dict fallback merged in for backwards-compat)
- [x] Worker prefers vaulted credentials over wire-provided ones during dispatch; cloud can stop sending creds once all users have populated vaults
- [x] 8 vault tests with an in-memory keyring backend
- [ ] Encrypted file fallback when no keyring backend exists: AES-256-GCM, Argon2id-derived key (deferred — `keyring` already covers macOS / Windows / mainstream Linux desktop; headless Linux is a separate UX problem)
- [ ] Cloud-side enforcement: if `ExperimentRequest.backend in {"ibm", "ionq"}` and `Hello.vault[backend]` is False, fail-fast with a clear error before dispatching

### 2.5 Cancellation
- [x] Cooperative interrupt on compute side: `cancel_check` callable threaded through `worker.run_calculation`, checked between major phases (Phase 1)
- [x] App-side cancel propagation: `POST /api/calculations/{id}/cancel` now also calls `dispatch_cancel_to_user(user_id, calc_id)` (no-op if no live session) and transitions Job to CANCELLED via `transition_job` for terminal-state safety against late FinalResult arrivals
- [ ] Per-iteration `should_stop()` flag inside solvers themselves — covered by the kanad-core 2.1b PR (each solver's iteration loop checks the same flag passed via `cancel_check`)

---

## 3. Phase 3 — Rust runtime (the actual rewrite)

### 3.1 Workspace bootstrap
- [ ] `kanad-compute` crate workspace: `vault/`, `auth/`, `protocol/`, `gateway/`, `runtime/`, `cli/`, `tui/`
- [ ] Build: `cargo workspace`, MSRV 1.75
- [ ] CI: `cargo test`, `cargo clippy -- -D warnings`, cross-compile macOS (arm64, x86_64), Linux (x86_64, aarch64), Windows (x86_64)

### 3.2 Native crates first (no solver math yet)
- [ ] `vault` — keyring + AES-GCM file fallback. Public API: `set/get/rotate/clear`.
- [ ] `auth` — RFC 8628 client; token refresh; revocation hook.
- [ ] `protocol` — Serde structs mirroring `kanad-compute/kanad_compute/protocol.py`. JSON for v1, optional Protobuf feature flag.
- [ ] `gateway` — async WS client (`tokio-tungstenite`); reconnect; outbox (`rusqlite`); same backoff as Python client.
- [ ] `cli` — replace Click commands one at a time: `connect`, `login`, `creds`, `status`.

### 3.3 Solver migration (Tier 1 → Tier 2)
- [ ] Build the statevector simulator natively (faer / ndarray + LAPACK)
- [ ] Port basic VQE (parameter-shift gradients, COBYLA / L-BFGS via `argmin`)
- [ ] Port PhysicsVQE governance layer
- [ ] Port HardwareVQE (transpilation pipeline; backend abstraction)
- [ ] Wrap remaining Python solvers via PyO3 shim (Tier 2): SQD, qEOM, advanced ansätze. Hide them behind the same `Solver` trait so callers don't care which tier handled the job.
- [ ] Tier-3 cloud-delegate: `ExperimentEvent { kind: "DelegatedToCloud", payload: {cloud_job_id} }` and stream from there

### 3.4 TUI (`ratatui`)
- [ ] Tabs: Status (connection, queue, throughput) · Jobs (live list w/ progress bars) · Vault · Logs
- [ ] Keybindings: `q` quit, `c` cancel selected job, `l` tail logs
- [ ] Background: continues running headless if TUI exits

---

## 4. Phase 4 — Cloud reduction & cleanup

Once Rust compute is the default and stable:

- [ ] Delete the ~6.2k LOC of solver code from `kanad-app/api/` (keep only validation / shape-checks needed for dispatch)
- [ ] Delete `kanad-app` polling endpoints (`/api/compute/jobs` family)
- [ ] Drop unused columns from `Job` model — anything that was for in-app execution state
- [ ] Audit `User` table: remove plaintext credential columns once vault migration is verified for all active users
- [ ] Document in `kanad-app/CLAUDE.md` that solver code lives in `kanad-compute` only

---

## 5. Cross-cutting / not-tied-to-a-phase

### 5.1 Testing
- [ ] Protocol round-trip tests on both sides — random `ExperimentEvent` fuzzing through `model_dump_json` → `parse_*_message`
- [ ] Compute integration test: in-process FastAPI app + `ComputeWSClient` connecting to it; submit a fake `ExperimentRequest`; assert browser-fanout payload arrives at a mock `connection_manager`
- [ ] Reconnect + replay test: kill the server mid-experiment, assert events redelivered
- [ ] Backpressure test: slow the server reader, assert outbox grows, no events lost

### 5.2 Observability
- [ ] Structured logging on both sides with `experiment_id` correlation
- [ ] Cloud: per-user "compute" dashboard page — connected status, last ping, in-flight count, last error
- [ ] Compute TUI mirrors the same data locally

### 5.3 Distribution
- [ ] PyPI: `kanad-compute` already has `pyproject.toml`; verify wheel builds + publishes from CI
- [ ] Homebrew tap (Phase 3): `brew install deeprealm-labs/tap/kanad-compute`
- [ ] Windows installer (Phase 3): MSI via `cargo wix` or similar
- [ ] Auto-update channel — opt-in, prompted on connect if version mismatch in `Registered`

### 5.4 Security review
- [ ] Threat model document: what does compromise of the cloud DB now expose? (Should be: nothing material — no creds, no results-in-flight.)
- [ ] Threat model: what does compromise of a user's machine expose? (Their own vault — same as today, but now also session tokens.)
- [ ] Pen-test the device-auth flow before announcing
- [ ] Rate-limit `compute_connect` per user, per IP

### 5.5 Docs
- [ ] User-facing README in `kanad-compute`: install, login, connect, creds
- [ ] Architecture doc in `kanad-app`: cloud responsibilities, compute responsibilities, where the line is
- [ ] Migration guide for users on the old polling worker

---

## 6. Decisions still open

- [x] **Multi-node per user (decided 2026-05).** "Bump previous session." When a new connection arrives for a user with an existing session, the old WS is closed (1001 GOING_AWAY, reason=`superseded`) and replaced. Rationale: reconnect after a network blip arrives BEFORE the dead session's heartbeat times out (~30 s), so rejecting the new connection would force the user to wait through that window. Matches `gh auth login` / `claude login` conventions. Multi-device support proper arrives with device-token auth (2.3).
- [!] **Browser Path A (direct localhost) vs Path B (proxy).** Path A needs CORS + self-signed-cert UX or a localhost-trusted-origin trick; Path B is a non-decision (works today). Default: ship Path B in Phase 1, evaluate Path A demand.
- [!] **Where do TUI logs come from?** Same outbox table read by gateway, or a separate `logs` ring buffer? Decide before starting 3.4.
- [x] **Solver versioning (decided 2026-05).** `kanad_core_version` is now sent in `Hello` and surfaced in `/api/admin/compute_status`. Logged on connect. Server-side refusal of incompatible jobs is deferred until a real version-skew incident; today's failure mode (mismatched fields) surfaces clearly via solver TypeError on dispatch.

---

## 7. Immediate next actions

Phase 1 (issue #16, WS gateway) is **code-complete and test-covered**. Three PRs landed across both repos:
- `kanad-compute`: `add ws gateway client` → `harden ws client outbox` → `add ws client tests`
- `kanad-app`: `add ws gateway endpoint` → `add job transition helper` → `harden compute ws endpoint` → `wire dispatch hook` → `add ws gateway tests`

Tests: 22 (compute) + 14 (app) = **36 passing**. Includes real-uvicorn WS smoke, dispatch round-trip, dedup-on-reconnect, cancel-race, and outbox/seq persistence.

Remaining before declaring Phase 1 done:
1. Manual e2e against a real Postgres + real molecule per `kanad-app/docs/compute_ws_smoke.md`
2. File the Phase-2 epic on `mk0dz/kanad-app` so issue #16 has trackable children for live `Progress`, SQLite outbox, RFC 8628 device auth, local vault
3. Decide §6 open questions (multi-node-per-user, Path A vs B, log source for TUI, solver versioning) before starting Phase 2 work
