"""Random-fuzz round-trip for every ExperimentEvent kind.

Both repos host an identical version of this test (paths differ, but the
fuzz seed, kind list, and assertion shape match). If a payload field ever
gets renamed on one side without the other, both tests fail in lock-step
— that's the schema-drift trip-wire promised in todo.md §5.1.
"""

from __future__ import annotations

import json
import random

import pytest

from kanad_compute.protocol import (
    ExperimentEvent,
    KIND_TO_PAYLOAD,
)


_FUZZ_SEED = 0xCAFEBABE
_ITERATIONS = 200


def _rand_payload(rng: random.Random, kind: str) -> dict:
    """Mint a payload that's valid for ``kind``. Mirrors the optional/required
    shape declared in protocol.py — keep this list in sync if a payload model
    grows a new required field."""
    if kind == "Log":
        return {
            "level": rng.choice(["debug", "info", "warning", "error"]),
            "message": f"m{rng.randint(0, 9999)}",
        }
    if kind == "Progress":
        return {
            "iteration": rng.randint(0, 10_000),
            "total": rng.choice([None, rng.randint(1, 1000)]),
            "energy": rng.choice([None, rng.uniform(-100.0, 0.0)]),
            "gradient_norm": rng.choice([None, rng.uniform(0.0, 10.0)]),
            "message": rng.choice([None, f"step {rng.randint(0, 100)}"]),
        }
    if kind == "PartialResult":
        return {
            "energy": rng.choice([None, rng.uniform(-100.0, 0.0)]),
            "hf_energy": rng.choice([None, rng.uniform(-100.0, 0.0)]),
        }
    if kind == "FinalResult":
        return {
            "energy": rng.uniform(-100.0, 0.0),
            "converged": rng.choice([True, False]),
            "n_evaluations": rng.randint(1, 5000),
            "wall_time_ms": rng.randint(0, 10_000_000),
        }
    if kind == "Error":
        return {
            "message": f"err{rng.randint(0, 9999)}",
            "code": rng.choice([None, "EBADF", "ETIMEOUT"]),
        }
    raise AssertionError(f"unhandled kind: {kind!r}")


def test_fuzz_event_roundtrip_all_kinds():
    rng = random.Random(_FUZZ_SEED)
    kinds = list(KIND_TO_PAYLOAD.keys())
    assert set(kinds) == {"Log", "Progress", "PartialResult", "FinalResult", "Error"}, (
        "kind set drifted; update _rand_payload"
    )

    for i in range(_ITERATIONS):
        kind = rng.choice(kinds)
        payload = _rand_payload(rng, kind)
        ev = ExperimentEvent(
            experiment_id=f"exp-{i}", seq=i, ts_ms=i * 7, kind=kind, payload=payload,
        )
        # Serialize then re-parse via the model (compute is the producer and
        # doesn't expose a ``parse_client_message``; the app side does and the
        # mirrored test there covers the discriminated-union path).
        wire = json.loads(ev.model_dump_json())
        back = ExperimentEvent.model_validate(wire)
        assert back.kind == kind
        assert back.seq == i
        # Payload fields survive verbatim — pydantic stores them as a dict.
        for k, v in payload.items():
            assert back.payload.get(k) == v, f"field {k!r} drifted for kind={kind}"


@pytest.mark.parametrize("kind", list(KIND_TO_PAYLOAD.keys()))
def test_event_payload_model_validates_min_shape(kind):
    """Every KIND_TO_PAYLOAD entry must construct from {} or its minimum
    required fields. Catches accidental new required fields on either side."""
    rng = random.Random(_FUZZ_SEED + hash(kind))
    payload = _rand_payload(rng, kind)
    model = KIND_TO_PAYLOAD[kind]
    model.model_validate(payload)
