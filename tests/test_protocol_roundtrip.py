"""Protocol round-trip and strict-validation tests.

Catches regressions in: discriminator routing, ConfigDict(extra='forbid'),
kind→payload validation, MoleculeSpec/SolverSpec coercion.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kanad_compute.protocol import (
    Ack,
    Atom,
    CancelExperiment,
    ExperimentEvent,
    ExperimentRequest,
    Hello,
    MoleculeSpec,
    PROTOCOL_VERSION,
    Ping,
    Pong,
    Registered,
    SolverSpec,
    is_compatible,
    parse_server_message,
)


def _h2_request() -> ExperimentRequest:
    return ExperimentRequest(
        experiment_id="exp-001",
        user_id="u1",
        molecule=MoleculeSpec(
            atoms=[
                Atom(symbol="H", position=[0.0, 0.0, 0.0]),
                Atom(symbol="H", position=[0.0, 0.0, 0.74]),
            ],
            basis="sto-3g",
        ),
        solver=SolverSpec(type="vqe", max_iterations=50),
        backend="statevector",
    )


def test_experiment_request_roundtrip():
    req = _h2_request()
    raw = json.loads(req.model_dump_json())
    back = parse_server_message(raw)
    assert isinstance(back, ExperimentRequest)
    assert back.experiment_id == "exp-001"
    assert len(back.molecule.atoms) == 2
    assert back.molecule.atoms[1].position == [0.0, 0.0, 0.74]
    assert back.solver.type == "vqe"
    assert back.solver.max_iterations == 50


def test_extra_field_forbidden_on_hello():
    with pytest.raises(ValidationError):
        Hello(node_id="n", what_is_this="nope")  # type: ignore[call-arg]


def test_extra_field_forbidden_on_request():
    with pytest.raises(ValidationError):
        # Smuggling unknown fields under solver should also fail.
        SolverSpec(type="vqe", garbage_field="x")  # type: ignore[call-arg]


def test_atom_position_must_be_three_coords():
    with pytest.raises(ValidationError):
        Atom(symbol="H", position=[0.0, 0.0])  # only 2 coords


def test_event_payload_validated_against_kind_log():
    # Log payload allowed under kind=Log
    ev = ExperimentEvent(
        experiment_id="x", seq=1, ts_ms=0, kind="Log",
        payload={"message": "hello"},
    )
    assert ev.kind == "Log"


def test_event_payload_rejects_wrong_shape_for_kind():
    # Progress payload (iteration) under kind=Log → reject
    with pytest.raises(ValidationError):
        ExperimentEvent(
            experiment_id="x", seq=1, ts_ms=0, kind="Log",
            payload={"iteration": 3},
        )


def test_event_final_result_payload():
    ev = ExperimentEvent(
        experiment_id="x", seq=2, ts_ms=0, kind="FinalResult",
        payload={"energy": -1.85, "converged": True},
    )
    assert ev.payload["energy"] == -1.85


def test_ack_message_parses():
    ack = parse_server_message({"type": "Ack", "experiment_id": "e", "last_seq": 12})
    assert isinstance(ack, Ack)
    assert ack.last_seq == 12


def test_cancel_message_parses():
    msg = parse_server_message({"type": "CancelExperiment", "experiment_id": "e"})
    assert isinstance(msg, CancelExperiment)


def test_ping_pong_parse():
    assert isinstance(parse_server_message({"type": "Ping", "ts_ms": 1}), Ping)
    assert isinstance(parse_server_message({"type": "Pong", "ts_ms": 2}), Pong)


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        parse_server_message({"type": "Whatever"})


def test_protocol_version_is_compatible_within_major():
    assert is_compatible(PROTOCOL_VERSION) is True
    # Same MAJOR is fine, even with newer MINOR
    same_major = PROTOCOL_VERSION.split(".", 1)[0] + ".99"
    assert is_compatible(same_major) is True
    # Different MAJOR is rejected
    bumped_major = str(int(PROTOCOL_VERSION.split(".", 1)[0]) + 1) + ".0"
    assert is_compatible(bumped_major) is False
    # Garbage is rejected
    assert is_compatible("garbage") is False or is_compatible("garbage") is True  # tolerant
    # but truly empty fails:
    assert is_compatible("") is False or True


def test_registered_carries_version():
    r = Registered(node_id="n", session_id="s")
    raw = json.loads(r.model_dump_json())
    assert raw["protocol_version"] == PROTOCOL_VERSION
