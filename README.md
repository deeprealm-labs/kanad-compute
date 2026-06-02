> ⚠️ **ARCHIVED (2026-06).** This repo is no longer the live compute path.
> Compute moved in-process into the **kanad-app** FastAPI backend and streams
> over `/ws/jobs/{id}`; the external node (WS gateway, device-auth, polling,
> workers, wire protocol) is retired. The code is preserved in git history for
> possible future in-process acceleration. See [ARCHIVED.md](ARCHIVED.md).

# kanad-compute

Turn your computer into a quantum-chemistry compute node for [Kanad](https://kanad.xyz).

Run VQE calculations, molecular dynamics, and quantum analyses on **your own hardware** — the
Kanad cloud keeps only molecule design, auth, history, and a thin dispatch layer. Your machine
owns the solvers **and** your backend credentials; nothing sensitive sits in the cloud database.

## Quick Start (recommended — WebSocket gateway)

```bash
# Install
pip install kanad-compute        # or: pip install -e .  (from source)

# One-time setup
kanad-compute init               # creates ~/.kanad-compute/config.json

# Log in (browser-mediated device authorization, like `gh auth login`)
kanad-compute login

# Connect — holds one persistent outbound WebSocket to the cloud and
# streams live progress back as your experiments run
kanad-compute connect
```

That's it. Open [kanad.xyz](https://kanad.xyz), pick **Kanad Compute** as your backend, and
submit an experiment — it runs on your machine and the convergence curve updates live in the
browser. No inbound ports, no API keys to copy-paste.

## How it works

```
Your machine                                   Kanad cloud (kanad.xyz)
┌───────────────────────────────┐             ┌──────────────────────────────┐
│ kanad-compute connect          │  outbound   │ /api/compute/connect (WS)    │
│  • persistent WebSocket  ──────┼────WSS──────▶  dispatch experiments down   │
│  • runs Kanad solvers          │             │  fan out events to browser   │
│  • streams Log/Progress/Result │◀────────────┼──  (live convergence chart)  │
│  • OS-keychain credential vault│             │  auth · history · molecule UX │
└───────────────────────────────┘             └──────────────────────────────┘
```

- **Wire format:** JSON over WebSocket — browser-friendly, proxy-friendly, debuggable.
- **Topology:** the compute node dials *out* to the cloud (NAT-friendly, no port forwarding).
  The browser talks to the cloud, which proxies events to/from your node (Path B).
- **Credentials:** stored in your OS keychain (Keychain / Credential Manager / Secret Service),
  never in the cloud DB. See `kanad-compute creds` below.
- **Auth:** RFC 8628 device authorization grant (`kanad-compute login`) → a 30-day access token
  stored in the vault. A legacy bearer API key still works for older deployments.
- **Crash-resilience:** every event is written to a local SQLite outbox before going on the wire
  and replayed on reconnect, so a dropped connection never loses results.

## What it does

kanad-compute executes quantum-chemistry calculations using the
[Kanad framework](https://github.com/mk0dz/kanad). When you select "Kanad Compute" as a backend
in the web app, your calculations run on YOUR machine instead of cloud services.

**Supported solvers:** PhysicsVQE, HardwareVQE, HybridSubspaceVQE, SQD, KrylovSQD, VQE, VarQITE,
qEOM, EfficientVQE, ExcitedStates

**Supported backends:** Statevector (local), Qiskit Aer (CPU/GPU), IBM Quantum (your credentials),
IonQ / BlueQubit (your credentials)

## Requirements

- Python 3.11+ (3.11/3.12 recommended — pyscf / qiskit-aer have no 3.14 wheels yet)
- 8 GB+ RAM recommended
- [Kanad](https://github.com/mk0dz/kanad) library installed (`pip install kanad` or from source)

## Installation

```bash
# From source (recommended today)
git clone https://github.com/mk0dz/kanad-compute.git
cd kanad-compute
pip install -e .                 # base
pip install -e ".[gpu]"          # + GPU (qiskit-aer-gpu)
pip install -e ".[ibm]"          # + IBM Quantum
pip install -e ".[ionq]"         # + IonQ
pip install -e ".[all]"          # everything

# From PyPI (coming soon)
pip install kanad-compute
```

> You also need the Kanad framework:
> ```bash
> git clone https://github.com/mk0dz/kanad.git && cd kanad && pip install -e .
> ```

## CLI

### `kanad-compute init`
Initialize configuration (`~/.kanad-compute/config.json`) with a unique node ID.

```bash
kanad-compute init --port 7440 --max-qubits 20 --gpu
```

### `kanad-compute login`
Log in via RFC 8628 device authorization — mints a device code, opens the verification URL in
your browser, polls until you approve, and stores the access token in your OS keychain.

```bash
kanad-compute login                  # opens browser
kanad-compute login --no-browser     # prints the URL + code instead
kanad-compute login --timeout 600    # max seconds to wait for approval
```

### `kanad-compute connect`
Hold a persistent outbound WebSocket to the cloud, run dispatched experiments, and stream live
`Log` / `Progress` / `FinalResult` events back. Prefers the vaulted access token from `login`.

```bash
kanad-compute connect
kanad-compute connect --url http://localhost:8000   # point at a local cloud dev server
```

### `kanad-compute creds` — local credential vault
Store backend credentials in the OS keychain so they never transit the cloud. The compute node
prefers vaulted credentials over anything sent over the wire.

```bash
kanad-compute creds set ibm_api_token            # prompts for the value
kanad-compute creds set ionq_api_key --value KEY
kanad-compute creds get ibm_api_token            # prints ****<last4>
kanad-compute creds get ibm_api_token --reveal   # prints the full value
kanad-compute creds list                         # which keys are present
kanad-compute creds clear ibm_api_token
```
Recognized keys: `ibm_api_token`, `ibm_crn`, `ionq_api_key`, `bluequbit_api_key`.

### `kanad-compute status` / `configure`
`status` prints connection + system info; `configure` updates config without reinitializing.

```bash
kanad-compute status
kanad-compute configure --max-qubits 25 --gpu
```

### Legacy (deprecated) — HTTP polling worker
The original model ran a local FastAPI server that the cloud polled every 2 s. It still works for
backward compatibility but is **deprecated** in favour of `connect` and will be removed a few minor
releases after the WS gateway is the default.

```bash
kanad-compute start            # starts the local FastAPI server (deprecated)
kanad-compute key              # prints the legacy API key to paste into the web app
```

## Configuration

Stored at `~/.kanad-compute/config.json`:

```json
{
  "node_id": "uuid",
  "kanad_url": "https://kanad.xyz",
  "port": 7440,
  "max_qubits": 20,
  "max_workers": 2,
  "gpu_enabled": false
}
```

Backend credentials live in the **OS keychain** (via `kanad-compute creds`), not in this file.

## Roadmap

The Python node is the shipping artifact today; a Rust runtime (native statevector + VQE, with a
PyO3 shim for advanced solvers) is in progress under `rust/`. See [`todo.md`](todo.md) for the full
architecture migration plan.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Links

- [Kanad Platform](https://kanad.xyz)
- [Kanad Framework](https://github.com/mk0dz/kanad)
- [DeepRealm Labs](https://deeprealm.in)
