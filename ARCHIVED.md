# ARCHIVED — kanad-compute

**Status: archived as of 2026-06.**

This repository is no longer the live compute path for [Kanad](https://kanad.xyz).

## What changed

In the 2026-06 pivot, compute was moved **in-process** into the Kanad app's
FastAPI backend (`kanad-app`). Calculations now run inside that backend and
stream progress to the browser over `/ws/jobs/{id}`. The external compute-node
architecture this repo implemented — the WebSocket gateway, RFC 8628 device
authentication, the HTTP polling fallback, the worker processes, and the
compute↔node wire protocol — has been removed from the backend and is no longer
used.

The accompanying Bring-Your-Own-Secrets credentials model means provider secrets
(IBM Quantum / BlueQubit / IonQ) for free users now live on the user's local
device and are injected per request, used transiently, and never persisted by
the backend. Paid tier runs on Kanad-owned server-side credentials. There is no
longer a separate compute node holding credentials.

## Why it's kept

The Python node and the Rust workspace are preserved in this repository's git
history for possible future use as **in-process acceleration** for the backend.
They are NOT the live path and are not maintained against the current backend.

## Where to look instead

All active compute lives in the **kanad-app** FastAPI backend:

- Calculation submission/streaming: `api/routes/calculations.py`
- Live updates over WebSocket: `api/websocket/manager.py`, endpoint `/ws/jobs/{id}`
- Server config and tier credentials: `api/config.py`

Do not build new functionality on top of this repository.
