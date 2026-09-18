from __future__ import annotations

import json
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from mackerel_road.nn.mechanics import (  # noqa: E402
    NeuralMechanics,
    event_mask_from_terrain,
)
from mackerel_road.nn.training import (  # noqa: E402
    load_serializable_state_dict,
    serializable_state_dict,
)


def test_neural_state_dict_round_trip() -> None:
    source = NeuralMechanics()
    target = NeuralMechanics()
    with torch.no_grad():
        next(source.parameters()).add_(0.25)
    load_serializable_state_dict(target, serializable_state_dict(source))

    features = torch.zeros((2, 12))
    features[:, 0] = 1.0
    features[:, 2:5] = torch.tensor((1.0, 1.0, 0.75))
    features[:, 5:8] = torch.tensor((0.5, 0.5, 0.5))
    features[:, 8] = 1.0
    mask = event_mask_from_terrain(features[:, 8:11].argmax(dim=-1))

    source_result = source(features, mask)
    target_result = target(features, mask)
    assert torch.allclose(source_result.travel_time, target_result.travel_time)
    assert torch.allclose(source_result.event_probability, target_result.event_probability)


def test_training_and_validation_pool_maps_before_objective(monkeypatch) -> None:
    from mackerel_road.experiments.protocol import ABLATIONS
    from mackerel_road.game.maps import generate_route_map
    from mackerel_road.nn import training

    original = training.neural_game_objective
    calls: list[int] = []

    def record_objective(model, maps, spec, **kwargs):
        calls.append(len(maps))
        return original(model, maps, spec, **kwargs)

    def tiny_pool(start, count, horizon):
        return tuple(generate_route_map(start + i, widths=(1, 2, 1)) for i in range(count))

    monkeypatch.setattr(training, "neural_game_objective", record_objective)
    monkeypatch.setattr(training, "_generate_pool", tiny_pool)
    result = training.train_one(
        ABLATIONS[-1],
        0,
        device=torch.device("cpu"),
        steps=1,
        batch_maps=4,
        learning_rate=0.003,
        patience=6,
        validation_interval=1,
        validation_maps_count=8,
        test_maps_count=0,
        horizon=None,
    )
    assert calls == [8, 4, 8]
    assert result["test"] is None
    history = result["history"]
    assert isinstance(history, list)
    assert len(history) == 2


def test_resume_requires_matching_protocol_and_finished_evaluation(tmp_path) -> None:
    from mackerel_road.nn.training import load_completed_runs

    protocol = {"kind": "neural_transition_engine", "test_maps": 100}
    payload: dict[str, Any] = {
        "protocol": protocol,
        "runs": [
            {
                "variant": "+skill_weather",
                "seed": 0,
                "best_validation_loss": 1.0,
                "state_dict": {"weight": [1.0]},
                "history": [{"step": 0}],
                "test": {"equal_skill_light": 0.5},
            }
        ],
    }
    path = tmp_path / "resume.json"
    path.write_text(json.dumps(payload))
    assert len(load_completed_runs(path, protocol)) == 1
    with pytest.raises(ValueError, match="protocol differs"):
        load_completed_runs(path, {**protocol, "test_maps": 10})
    incomplete = json.loads(path.read_text())
    incomplete["runs"][0]["test"] = None
    path.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="completed hard evaluation"):
        load_completed_runs(path, protocol)
