from __future__ import annotations

import pytest

from mackerel_road.experiments.artifacts import (
    CHECKPOINT_PREFIX,
    RESULT_PREFIX,
    encode_payload,
    extract_payload,
)


def test_extract_payload_round_trip() -> None:
    expected = {"protocol": {"device": "cuda"}, "runs": [{"variant": "viability"}]}
    encoded = encode_payload(expected)

    assert extract_payload(f"before\n{RESULT_PREFIX}{encoded}\nafter\n") == expected


def test_extract_payload_rejects_missing_or_duplicate_results() -> None:
    with pytest.raises(ValueError, match="found 0"):
        extract_payload("no payload")
    encoded = encode_payload({"protocol": {}, "runs": []})
    with pytest.raises(ValueError, match="found 2"):
        extract_payload(f"{RESULT_PREFIX}{encoded}\n{RESULT_PREFIX}{encoded}")


def test_extract_payload_uses_latest_checkpoint_until_final_result() -> None:
    first = {"protocol": {}, "runs": [{"seed": 0}]}
    second = {"protocol": {}, "runs": [{"seed": 0}, {"seed": 1}]}
    final = {"protocol": {}, "runs": [{"seed": 2}]}
    checkpoint_log = (
        f"{CHECKPOINT_PREFIX}{encode_payload(first)}\n{CHECKPOINT_PREFIX}{encode_payload(second)}\n"
    )

    assert extract_payload(checkpoint_log) == second
    assert extract_payload(checkpoint_log + f"{RESULT_PREFIX}{encode_payload(final)}\n") == final
