from __future__ import annotations

from typing import Any

import pytest

from mackerel_road.experiments.summarize import (
    select_neural_run,
    validate_neural_payload,
    write_neural_summary,
)


def _payload() -> dict[str, Any]:
    return {
        "protocol": {
            "kind": "neural_transition_engine",
            "cuda_device": "test GPU",
            "train_maps": 64,
            "validation_maps": 4,
            "test_maps": 1,
            "steps": 2,
            "events": ["NONE", "LANDSLIDE"],
        },
        "runs": [
            {
                "variant": "+skill_weather",
                "seed": seed,
                "best_step": 1,
                "best_validation_loss": loss,
                "elapsed_seconds": 1.0,
                "state_dict": {"weight": [[0.1]]},
                "history": [{"step": 0}],
                "test": None,
            }
            for seed, loss in ((0, 0.3), (1, 0.2))
        ],
    }


def test_neural_payload_selection_and_summary(tmp_path) -> None:
    payload = _payload()
    validate_neural_payload(payload)
    assert select_neural_run(payload)["seed"] == 1

    output = tmp_path / "summary.md"
    write_neural_summary(payload, output)
    assert "Teacher engine: `none`" in output.read_text()


def _test_metrics(light: float, profile_end: float) -> dict[str, object]:
    return {
        "equal_skill_light": light,
        "equal_skill_strong": 1.0 - light,
        "equal_skill_draw": 0.0,
        "exact_light": 0.5,
        "exact_strong": 0.5,
        "exact_draw": 0.0,
        "cgq_points": [[0.0, 0.5], [1.0, profile_end]],
        "skill_leverage": 0.5 - profile_end,
        "skill_tolerance": 0.2,
        "failure_rate": 0.0,
        "mean_final_freshness": 0.6,
        "mean_freshness_margin": 0.03,
        "event_rate": 0.2,
        "event_frequencies": {"NONE": 0.8, "LANDSLIDE": 0.2},
        "semantic_penalties": {"semantic_distance": 0.0},
    }


def test_neural_summary_groups_ablations_and_profiles_selected_run(tmp_path) -> None:
    payload = _payload()
    payload["runs"][0]["test"] = _test_metrics(0.4, 0.1)
    payload["runs"][1]["test"] = _test_metrics(0.6, 0.0)
    payload["runs"].append(
        {
            "variant": "viability",
            "seed": 0,
            "best_step": 1,
            "best_validation_loss": 0.1,
            "elapsed_seconds": 1.0,
            "state_dict": {"weight": [[0.1]]},
            "history": [{"step": 0}],
            "test": _test_metrics(0.7, 0.3),
        }
    )

    output = tmp_path / "summary.md"
    write_neural_summary(payload, output)
    summary = output.read_text()

    assert "| +skill_weather | 0.500 ± 0.100" in summary
    assert "| viability | 0.700" in summary
    assert "Epsilon to weaker-player score: `0.00:0.500, 1.00:0.000`" in summary


def test_neural_summary_allows_undefined_skill_tolerance(tmp_path) -> None:
    payload = _payload()
    metrics = _test_metrics(0.5, 0.5)
    metrics["skill_tolerance"] = None
    payload["runs"][1]["test"] = metrics

    output = tmp_path / "summary.md"
    write_neural_summary(payload, output)

    assert "Skill tolerance: `undefined`" in output.read_text()


def test_reporting_payload_can_omit_weights_but_checkpoint_selection_cannot(tmp_path) -> None:
    payload = _payload()
    for run in payload["runs"]:
        del run["state_dict"]
    validate_neural_payload(payload, require_state=False)
    write_neural_summary(payload, tmp_path / "summary.md")
    with pytest.raises(ValueError, match="state dictionary"):
        select_neural_run(payload)
