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
- [x] kanad-core PR (`deeprealm-labs/kanad#2`, branch `solver-callbacks-phase2-1b`): adds `callback` kwarg to PhysicsVQE.solve, HardwareVQE.solve_local/solve_hardware, VarQITESolver.solve, qEOMVQE.solve, EfficientVQE.solve. Signature: `callback(iteration, energy[, message])`. Exceptions inside the callback are swallowed at each call site so a buggy consumer can't break optimization.
- [ ] Tier requirement: kanad-core ≥ X.Y once `#2` is merged. Compute-side `_solve_with_optional_callback` keeps the TypeError fallback so a node running an older kanad-core still works (no Progress, but no breakage).

### 2.2 SQLite outbox (crash-resilient delivery)
- [x] `kanad_compute/outbox.py` — `Outbox(path)` with WAL journaling, threading lock, `record / ack / pending / pending_count / gc / close`
- [x] `_emit`: insert THEN send (durable on disk before going on the wire)
- [x] `_handle_ack`: `outbox.ack(exp_id, last_seq)`; `last_ack_seq` mirrored to seq.json so Hello populates fast
- [x] `_replay_unacked`: drains `outbox.pending()` ordered by `(exp_id, seq)` after Registered
- [x] Retention: `outbox.gc(older_than_seconds=86400)` runs once at startup; defensive cleanup against runaway state
- [x] Replaces the in-memory ring buffer entirely; `_ExperimentBuffer.unacked` removed; `_evict_old` removed
- [x] 7 outbox tests (durability, concurrent record, ack semantics, gc, ordering)

### 2.3 RFC 8628 device authorization grant
- [x] `POST /api/auth/device/code` (cloud, unauth): mints `(device_code, user_code)` with 8-char dashed user_code from a vowel-free alphabet. Returns `verification_uri`, `verification_uri_complete?code=…`, `interval`, `expires_in=900s`. Opportunistic GC sweeps expired PENDING rows on every mint.
- [x] `POST /api/auth/device/token` (cloud, unauth): standard RFC 8628 polling endpoint with error states `authorization_pending`, `slow_down`, `expired_token`, `access_denied`, `invalid_grant`. On APPROVED, mints a 30-day JWT and flips row to REDEEMED so it can't be re-used.
- [x] `POST /api/auth/device/approve` (cloud, auth): user types `user_code` at `/connect-device?code=…`; this flips the row to APPROVED and binds the device to the calling user's id. Idempotent for the same user.
- [x] Browser flow at `/connect-device` (Next.js, `web/src/app/connect-device/page.tsx`): reads `?code=` from the URL, prompts the signed-in user to confirm, calls `/api/auth/device/approve` with the user's existing session token. Shows success state with `client_id`.
- [x] CLI: `kanad-compute login [--no-browser] [--timeout SECS]` mints a code, prints it + URL, opens the browser, polls `/api/auth/device/token` honouring the `interval` (and bumps on `slow_down`). On success, stores the access token in `Vault` under canonical key `kanad_access_token`.
- [x] WS auth: `compute_ws.py::_authenticate` now decodes a JWT first; falls back to the legacy `User.kanad_compute_key` so older nodes don't break. `ws_client.py` prefers the vaulted `kanad_access_token` over the config `api_key` on connect.
- [ ] **Refresh tokens.** Not minted yet — the 30-day window covers the migration period; rotation lands when we shorten access tokens to ~24h.
- [ ] **Revocation UI.** A `/dashboard/devices` page listing approved sessions with a "revoke" button. Deferred until first multi-device user; today a server-side `UPDATE device_codes SET status='denied' WHERE …` does the job.

### 2.4 Local credential vault
- [x] `kanad_compute/vault.py` wraps `keyring` (OS keychain on macOS / Credential Manager on Windows / Secret Service on Linux). API: `set/get/has/clear/status/all/list_present`
- [x] CLI: `kanad-compute creds set/get/list/clear` (Click subgroup); reveal-flag controls full-token print vs `****<last4>` redaction
- [x] Vault state surfaced in `Hello.vault` from real keyring presence (config-dict fallback merged in for backwards-compat)
- [x] Worker prefers vaulted credentials over wire-provided ones during dispatch; cloud can stop sending creds once all users have populated vaults
- [x] 8 vault tests with an in-memory keyring backend
- [ ] Encrypted file fallback when no keyring backend exists: AES-256-GCM, Argon2id-derived key (deferred — `keyring` already covers macOS / Windows / mainstream Linux desktop; headless Linux is a separate UX problem)
- [x] Cloud-side enforcement: `_try_ws_dispatch` runs a pre-flight credential check via `_check_credentials_available` keyed on solver type. When the requested solver needs cloud creds (today: `hardware_vqe` → IBM) and neither the node's vault nor the User row / .env has them, the dispatch refuses, marks the Job `FAILED` with a user-readable `error_message`, and skips the polling fallback. Mapping in `_SOLVER_VAULT_REQUIREMENTS` is the extension point for ionq/bluequbit once those solvers land in the WS path.

### 2.5 Cancellation
- [x] Cooperative interrupt on compute side: `cancel_check` callable threaded through `worker.run_calculation`, checked between major phases (Phase 1)
- [x] App-side cancel propagation: `POST /api/calculations/{id}/cancel` now also calls `dispatch_cancel_to_user(user_id, calc_id)` (no-op if no live session) and transitions Job to CANCELLED via `transition_job` for terminal-state safety against late FinalResult arrivals
- [ ] Per-iteration `should_stop()` flag inside solvers themselves — covered by the kanad-core 2.1b PR (each solver's iteration loop checks the same flag passed via `cancel_check`)

---

## 3. Phase 3 — Rust runtime (the actual rewrite)

### 3.1 Workspace bootstrap
- [x] `kanad-compute` Cargo workspace under `rust/` with seven crates: `protocol`, `vault`, `auth`, `gateway`, `runtime`, `cli`, `tui`
- [x] Build: `cargo check --workspace` clean; MSRV bumped to 1.85 (clap 4.6 + edition2024 floor — 1.75 in the original plan was no longer reachable from current crates.io)
- [x] `cargo test --workspace` — 24 tests passing (7 protocol + 6 vault + 3 auth + 8 gateway)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` — clean
- [ ] CI: GitHub Actions workflow running the above on each push, plus cross-compile matrix for macOS arm64/x86_64, Linux x86_64/aarch64, Windows x86_64 (deferred to 3.2)

### 3.2 Native crates first (no solver math yet)
- [x] `protocol` — Serde mirror of `kanad_compute/protocol.py`. Tagged-enum `ServerMessage` / `ClientMessage`, `deny_unknown_fields` on every struct, custom (de)serialize for `ExperimentEvent` so `kind` + `payload` stay synchronized by construction. Round-trip + reject-unknown-field tests cover Hello, ExperimentEvent (Progress), ExperimentRequest, and Ack.
- [x] `vault` — keyring wrapper with canonical-key allowlist (`ibm_api_token`, `ibm_crn`, `ionq_api_key`, `bluequbit_api_key`, `kanad_access_token`). Backend trait + `MemoryBackend` for tests. Headless-Linux AES-GCM file fallback still deferred (Phase 2 tail item).
- [x] `auth` — RFC 8628 client: `DeviceFlow::request_code` + `poll_token` loop honouring `authorization_pending` / `slow_down` (+5 s) / `access_denied` / `expired_token` / `invalid_grant`. Returns `AccessToken { access_token, expires_in, ... }`. Refresh-token rotation still deferred (same as the Python side).
- [x] `gateway` — Phase 3.1 ships the foundation only: durable `Outbox` (`rusqlite`, WAL, `synchronous=NORMAL`) that mirrors the Python schema byte-for-byte (`events(id, experiment_id, seq, kind, payload_json, created_at)`) and a `Backoff` (1 s → 30 s exponential, ±20 % jitter). The async WS send/recv loop wires up in 3.2.
- [x] `cli` — `kanad-compute` binary with subcommands `status`, `creds {set,get,list,clear}`, `login [--no-browser] [--timeout SECS]`, `connect [--node-id]`, `version`.
- [x] `gateway` WS send/recv loop (`tokio-tungstenite`): Hello/Registered handshake (10s timeout, version-major check), reader/writer/ping tasks under a `JoinSet`, mpsc-fanned writer drains a single `WsSink`, `emit()` records-to-outbox-then-sends, Ack-driven outbox trim + `seq.json` persist, ping watchdog (15s tick, 2-miss disconnect via atomic `last_pong_at`), reconnect via `Backoff`. Solver execution plugs in through `kanad-runtime::Solver` + `ProgressSink` + `CancelToken`. Integration test (`crates/gateway/tests/ws_smoke.rs`) spins up an in-process tungstenite server and proves Hello → Registered → ExperimentRequest → Log + Error events → Ack → outbox trim end-to-end.

### 3.3 Solver migration (Tier 1 → Tier 2)
- [x] Build the statevector simulator natively. `runtime/statevector.rs` implements a dense `StateVector<Complex64>` with single-qubit gates (I/X/Y/Z/H/S/T/RX/RY/RZ/phase), controlled-1q gates (CNOT/CZ), SWAP, in-place stride-based application, and a compact `Op` enum so ansätze can hand the simulator a `Vec<Op>`. `runtime/pauli.rs` adds `PauliString` / `PauliSum` with Hermitian expectation values computed in O(N) per term (no matrix materialization), Qiskit-convention `from_label` parser, and Bell/H2-minimal sanity tests. 15 unit tests passing.
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
- [x] Protocol round-trip fuzz tests on both sides (`tests/test_protocol_fuzz.py`): identical `_FUZZ_SEED=0xCAFEBABE`, 200 iterations across all five kinds. If a payload field is renamed on one side, both tests fail in lock-step.
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

Phase 1 + **all** of Phase 2 (2.1a / 2.1b / 2.2 / 2.3 / 2.4 except headless-Linux fallback / 2.5) + §6.1 / §6.4 are **code-complete and test-covered**.

Open PRs (stacked on existing branches, not yet merged):
- `deeprealm-labs/kanad-compute#2` — Phase 2.1 + 2.2 + 2.4 + 2.3-CLI (login command + vault token + WS prefers JWT)
- `mk0dz/kanad-app#18` — Phase 2.1 + 2.5 + 2.3-server (device-auth routes + /connect-device page + WS JWT auth)
- `deeprealm-labs/kanad#2` — Phase 2.1b: `callback` kwarg added to PhysicsVQE / HardwareVQE / VarQITE / qEOM / EfficientVQE in kanad-core.

Phase 1 PRs already merged:
- `kanad-compute`: `add ws gateway client` → `harden ws client outbox` → `add ws client tests`
- `kanad-app`: `add ws gateway endpoint` → `add job transition helper` → `harden compute ws endpoint` → `wire dispatch hook` → `add ws gateway tests`

Tests (current): 47 (compute) + 32 (app) = **79 passing**.

Remaining before declaring Phase 2 done:
1. **Manual e2e** against a real Postgres + real molecule per `kanad-app/docs/compute_ws_smoke.md` — golden path (VQE/SQD progress + cancel + reconnect) AND the new device-auth login flow (`kanad-compute login` → browser approval → JWT in vault → WS connects with JWT bearer). Not yet run by any session.
2. Merge the three open PRs once the manual smoke passes.

Phase 2 deferred items (not blocking Phase 3 kickoff):
- **Refresh-token rotation** — 30-day access tokens cover the migration period; revisit if/when we shorten the window.
- **`/dashboard/devices` revocation UI** — defer until first multi-device user. Today: `UPDATE device_codes SET status='denied' WHERE user_code='…'` does the job.
- **Encrypted file vault fallback** (2.4 tail) — only needed for headless Linux; punted until a real user hits it.

§6 status:
- §6.1 (multi-node) — **decided**: bump previous session, structured log.
- §6.4 (solver versioning) — **decided**: advertise `kanad_core_version`; no server refusal yet.
- §6.2 Path A vs B — still `[!]`, decide before Phase 3.
- §6.3 TUI log source — still `[!]`, decide before 3.4.
