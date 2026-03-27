# kanad-compute

Turn your computer into a quantum chemistry compute server for [Kanad](https://kanad.xyz).

Run VQE calculations, molecular simulations, and quantum analyses on your own hardware — then connect to the Kanad platform for visualization, collaboration, and reporting.

## Quick Start

```bash
# Install
pip install kanad-compute

# Initialize (creates config + API key)
kanad-compute init

# Start the server
kanad-compute start
```

Then paste your API key into **Kanad > Profile > Backend Credentials > Kanad Compute**.

## What It Does

kanad-compute runs a local FastAPI server that executes quantum chemistry calculations using the [Kanad framework](https://github.com/mk0dz/kanad). When you select "Kanad Compute" as a backend in the Kanad web app, your calculations run on YOUR machine instead of cloud services.

**Supported solvers:** PhysicsVQE, HardwareVQE, HybridSubspaceVQE, SQD, KrylovSQD, VQE, VarQITE, qEOM, EfficientVQE, ExcitedStates

**Supported backends:** Statevector (local), Qiskit Aer (CPU/GPU), IBM Quantum (with your credentials), IonQ (with your credentials)

## Requirements

- Python 3.11+
- 8GB+ RAM recommended
- [Kanad](https://github.com/mk0dz/kanad) library installed (`pip install kanad` or install from source)

## Installation

### From GitHub (recommended)

```bash
# Clone the repo
git clone https://github.com/mk0dz/kanad-compute.git
cd kanad-compute

# Install in development mode
pip install -e .

# With GPU acceleration
pip install -e ".[gpu]"

# With IBM Quantum hardware
pip install -e ".[ibm]"

# With IonQ
pip install -e ".[ionq]"

# Everything
pip install -e ".[all]"
```

> **Note:** You also need the Kanad framework installed. If you don't have it:
> ```bash
> git clone https://github.com/mk0dz/kanad.git
> cd kanad && pip install -e .
> ```

### From PyPI (coming soon)
```bash
pip install kanad-compute
```

## CLI Commands

### `kanad-compute init`

Initialize configuration. Creates `~/.kanad-compute/config.json` with a unique node ID and API key.

```bash
kanad-compute init --port 7440 --max-qubits 20 --gpu
```

Options:
- `--port` — Server port (default: 7440)
- `--max-qubits` — Maximum qubits to accept (default: 20)
- `--gpu / --no-gpu` — Enable GPU acceleration
- `--ibm-token` — IBM Quantum API token
- `--ionq-key` — IonQ API key

### `kanad-compute start`

Start the compute server.

```bash
kanad-compute start --host 0.0.0.0 --port 7440
```

### `kanad-compute status`

Check server status and system info.

```bash
kanad-compute status
```

### `kanad-compute key`

Display your API key (for pasting into Kanad app).

```bash
kanad-compute key
```

### `kanad-compute configure`

Update configuration without reinitializing.

```bash
kanad-compute configure --ibm-token YOUR_TOKEN --max-qubits 25
```

## Connecting to Kanad

1. Run `kanad-compute start`
2. Copy your API key: `kanad-compute key`
3. Go to [kanad.xyz](https://kanad.xyz) > Profile > Backend Credentials
4. Under "Kanad Compute", paste the API key and set the server URL (`http://localhost:7440`)
5. Click "Test" to verify the connection
6. Select "Kanad Compute" as your backend when running experiments

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check (no auth) |
| GET | `/info` | System info and capabilities |
| POST | `/jobs` | Submit a calculation job |
| GET | `/jobs/{id}` | Get job status and results |
| POST | `/jobs/{id}/cancel` | Cancel a running job |
| GET | `/jobs` | List recent jobs |

All endpoints except `/health` require Bearer token authentication.

## Architecture

```
Your Machine
+-- kanad-compute server (FastAPI, port 7440)
|   +-- /health, /info
|   +-- /jobs (submit, poll, cancel)
|   +-- Thread Pool Executor
|       +-- Kanad Solvers (PhysicsVQE, HardwareVQE, ...)
|       +-- Kanad Backends (statevector, aer, ibm, ionq)
+-- Config: ~/.kanad-compute/config.json

Kanad Web App (kanad.xyz)
+-- Profile > Backend Credentials > Kanad Compute
+-- Schrodinger Lab > Select "Kanad Compute" backend
+-- Jobs proxied to your machine via API key auth
```

## Configuration

Config is stored at `~/.kanad-compute/config.json`:

```json
{
  "node_id": "uuid",
  "api_key": "your-api-key",
  "port": 7440,
  "max_qubits": 20,
  "max_workers": 2,
  "gpu_enabled": false,
  "ibm_api_token": null,
  "ionq_api_key": null
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Links

- [Kanad Platform](https://kanad.xyz)
- [Kanad Framework](https://github.com/mk0dz/kanad)
- [DeepRealm Labs](https://deeprealm.in)
