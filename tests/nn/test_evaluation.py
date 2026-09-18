from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mackerel_road.game.maps import generate_route_map  # noqa: E402
from mackerel_road.nn.evaluation import (  # noqa: E402
    evaluate_neural_model,
    play_neural_match,
)
from mackerel_road.nn.mechanics import EventKind, NeuralMechanics  # noqa: E402


def _tiny_map():
    return generate_route_map(7, widths=(1, 2, 2, 1))


def test_hard_neural_match_is_replayable_and_uses_valid_events() -> None:
    model = NeuralMechanics()
    arguments = {
        "cloud_seed": 11,
        "event_seed": 12,
        "decision_seed": 13,
        "light_skill": 0.8,
        "strong_skill": 0.8,
    }

    first = play_neural_match(model, _tiny_map(), **arguments)
    second = play_neural_match(model, _tiny_map(), **arguments)

    assert first == second
    assert len(first.records) == 3
    assert first.light.finished
    assert first.strong.finished
    for record in first.records:
        for outcome in (record.light_outcome, record.strong_outcome):
            assert outcome is not None
            assert outcome.event in EventKind


def test_neural_diagnostics_are_bounded() -> None:
    diagnostics = evaluate_neural_model(NeuralMechanics(), (_tiny_map(),), epsilons=(0.0, 1.0))

    assert (
        diagnostics.equal_skill_light
        + diagnostics.equal_skill_strong
        + diagnostics.equal_skill_draw
        == 1.0
    )
    assert diagnostics.cgq_points[0] == (0.0, 0.5)
    assert 0.0 <= diagnostics.event_rate <= 1.0
    assert 0.0 <= diagnostics.failure_rate <= 1.0
