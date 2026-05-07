"""Wire protocol for kanad-compute ↔ kanad-app WebSocket gateway.

Mirror of `api/protocol.py` on the kanad-app side. Both sides import from
their own copy so the modules stay independently typed.

Versioning: PROTOCOL_VERSION is MAJOR.MINOR. The server rejects MAJOR
mismatches on Hello (close 1003). MINOR additions are forward-compatible —
additive fields with defaults only.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PROTOCOL_VERSION = "1.0"


def _major(v: str) -> str:
    return v.split(".", 1)[0]


def is_compatible(server_version: str) -> bool:
    try:
        return _major(server_version) == _major(PROTOCOL_VERSION)
    except Exception:
        return False


_StrictModel = ConfigDict(extra="forbid")


# ── Event payloads (compute → app) ────────────────────────────────────────────

class LogPayload(BaseModel):
    model_config = _StrictModel
    level: Literal["debug", "info", "warning", "error"] = "info"
    message: str
    detail: Optional[str] = None


class ProgressPayload(BaseModel):
    model_config = _StrictModel
    iteration: int
    total: Optional[int] = None
    energy: Optional[float] = None
    gradient_norm: Optional[float] = None
    message: Optional[str] = None


class PartialResultPayload(BaseModel):
    model_config = _StrictModel
    energy: Optional[float] = None
    hf_energy: Optional[float] = None
    fci_energy: Optional[float] = None
    fields: dict[str, Any] = Field(default_factory=dict)


class FinalResultPayload(BaseModel):
    model_config = _StrictModel
    energy: Optional[float] = None
    hf_energy: Optional[float] = None
    fci_energy: Optional[float] = None
    error_mha: Optional[float] = None
    n_evaluations: Optional[int] = None
    converged: Optional[bool] = None
    convergence_history: Optional[list[dict[str, Any]]] = None
    wall_time_ms: Optional[int] = None
    actual_backend: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ErrorPayload(BaseModel):
    model_config = _StrictModel
    message: str
    traceback: Optional[str] = None
    code: Optional[str] = None


KIND_TO_PAYLOAD: dict[str, type[BaseModel]] = {
    "Log": LogPayload,
    "Progress": ProgressPayload,
    "PartialResult": PartialResultPayload,
    "FinalResult": FinalResultPayload,
    "Error": ErrorPayload,
}


# ── Typed sub-models for ExperimentRequest ────────────────────────────────────

class Atom(BaseModel):
    model_config = _StrictModel
    symbol: str
    position: list[float]

    @field_validator("position")
    @classmethod
    def _three_coords(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("position must be [x, y, z]")
        return v


class MoleculeSpec(BaseModel):
    model_config = _StrictModel
    atoms: list[Atom]
    basis: str = "sto-3g"
    charge: int = 0
    multiplicity: int = 1


class SolverSpec(BaseModel):
    model_config = _StrictModel
    type: str
    ansatz_type: str = "hardware_efficient"
    max_iterations: int = 100
    max_excitations: int = 5
    optimizer: Optional[str] = None
    mapper_type: Optional[str] = None
    convergence_threshold: Optional[float] = None
    n_layers: Optional[int] = None
    shots: Optional[int] = None
    frozen_core: bool = False
    include_singles: bool = True
    include_doubles: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


# ── Top-level messages ────────────────────────────────────────────────────────

class Hello(BaseModel):
    model_config = _StrictModel
    type: Literal["Hello"] = "Hello"
    protocol_version: str = PROTOCOL_VERSION
    node_id: str
    client_version: str = "0.1.0"
    system_info: Optional[dict[str, Any]] = None
    vault: Optional[dict[str, bool]] = None
    last_ack_seq: dict[str, int] = Field(default_factory=dict)


class Registered(BaseModel):
    model_config = _StrictModel
    type: Literal["Registered"] = "Registered"
    protocol_version: str = PROTOCOL_VERSION
    node_id: str
    session_id: str
    server_version: str = "0.1.0"


class ExperimentRequest(BaseModel):
    model_config = _StrictModel
    type: Literal["ExperimentRequest"] = "ExperimentRequest"
    experiment_id: str
    user_id: str
    molecule: MoleculeSpec
    solver: SolverSpec
    backend: str
    backend_credentials: Optional[dict[str, str]] = None
    deadline_ms: int = 600_000


class CancelExperiment(BaseModel):
    model_config = _StrictModel
    type: Literal["CancelExperiment"] = "CancelExperiment"
    experiment_id: str


class ExperimentEvent(BaseModel):
    model_config = _StrictModel
    type: Literal["ExperimentEvent"] = "ExperimentEvent"
    experiment_id: str
    seq: int
    ts_ms: int
    kind: Literal["Log", "Progress", "PartialResult", "FinalResult", "Error"]
    payload: dict[str, Any]

    @model_validator(mode="after")
    def _validate_payload_shape(self) -> "ExperimentEvent":
        model = KIND_TO_PAYLOAD.get(self.kind)
        if model is None:
            raise ValueError(f"unknown event kind: {self.kind!r}")
        model.model_validate(self.payload)
        return self


class Ack(BaseModel):
    model_config = _StrictModel
    type: Literal["Ack"] = "Ack"
    experiment_id: str
    last_seq: int


class Ping(BaseModel):
    model_config = _StrictModel
    type: Literal["Ping"] = "Ping"
    ts_ms: int


class Pong(BaseModel):
    model_config = _StrictModel
    type: Literal["Pong"] = "Pong"
    ts_ms: int


# ── Discriminated unions ──────────────────────────────────────────────────────

ServerMessage = Union[Registered, ExperimentRequest, CancelExperiment, Ack, Ping, Pong]
ClientMessage = Union[Hello, ExperimentEvent, Ping, Pong]


_SERVER_PARSERS: dict[str, type[BaseModel]] = {
    "Registered": Registered,
    "ExperimentRequest": ExperimentRequest,
    "CancelExperiment": CancelExperiment,
    "Ack": Ack,
    "Ping": Ping,
    "Pong": Pong,
}


def parse_server_message(raw: dict) -> ServerMessage:
    cls = _SERVER_PARSERS.get(raw.get("type"))
    if cls is None:
        raise ValueError(f"Unknown server message type: {raw.get('type')!r}")
    return cls.model_validate(raw)  # type: ignore[return-value]
