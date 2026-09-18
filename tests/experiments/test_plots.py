from __future__ import annotations

import json
from pathlib import Path

from mackerel_road.experiments.plots import write_neural_plots


def _test_metrics(light: float, leverage: float) -> dict[str, object]:
    return {
        "equal_skill_light": light,
        "equal_skill_strong": 1.0 - light,
        "skill_leverage": leverage,
        "cgq_points": [
            [0.0, 0.5],
            [0.25, 0.4],
            [0.5, 0.3],
            [0.75, 0.2],
            [1.0, 0.5 - leverage],
        ],
        "event_frequencies": {
            "NONE": 0.8,
            "LANDSLIDE": 0.08,
            "FALLEN_TREE": 0.07,
            "CART_BLOCKAGE": 0.05,
        },
    }


def _run(variant: str, seed: int, light: float, leverage: float) -> dict[str, object]:
    return {
        "variant": variant,
        "seed": seed,
        "best_step": 10,
        "best_validation_loss": 1.0,
        "elapsed_seconds": 1.0,
        "state_dict": {"weight": [[0.1]]},
        "history": [
            {"step": 0, "validation_total": 2.0},
            {"step": 10, "validation_total": 1.0},
        ],
        "test": _test_metrics(light, leverage),
    }


def _payload(runs: list[dict[str, object]]) -> dict[str, object]:
    return {"protocol": {"kind": "neural_transition_engine"}, "runs": runs}


def test_write_neural_plots(tmp_path: Path) -> None:
    ablation = _payload(
        [
            _run("viability", 0, 0.9, 0.2),
            _run("+balance", 0, 0.5, 0.25),
            _run("+specialization", 0, 0.55, 0.3),
            _run("+skill_weather", 0, 0.52, 0.4),
        ]
    )
    final = _payload(
        [
            _run("+skill_weather", 0, 0.52, 0.44),
            _run("+skill_weather", 1, 0.48, 0.42),
        ]
    )
    ablation_path = tmp_path / "ablation.json"
    final_path = tmp_path / "final.json"
    ablation_path.write_text(json.dumps(ablation), encoding="utf-8")
    final_path.write_text(json.dumps(final), encoding="utf-8")

    outputs = write_neural_plots(ablation_path, final_path, tmp_path / "figures")

    assert len(outputs) == 6
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_ablation_does_not_substitute_a_different_final_evaluation(tmp_path, monkeypatch) -> None:
    from mackerel_road.experiments import plots

    ablation = _payload([_run(variant, 0, 0.52, 0.4) for variant in plots.VARIANTS])
    final = _payload([_run("+skill_weather", 0, 0.9, 0.1)])
    original = plots._tests
    calls = []

    def record_tests(payload, variant):
        calls.append((payload is ablation, variant))
        return original(payload, variant)

    monkeypatch.setattr(plots, "_tests", record_tests)
    plots.plot_ablation(ablation, final, tmp_path)
    assert calls == [(True, variant) for variant in plots.VARIANTS]
    calls.clear()
    plots.plot_report_overview(ablation, final, tmp_path)
    assert calls[:4] == [(True, variant) for variant in plots.VARIANTS]
    assert calls[4:] == [(False, "+skill_weather")]
