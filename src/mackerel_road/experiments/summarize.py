"""Validate neural-run payloads and write a compact review summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def validate_neural_payload(payload: dict[str, Any], *, require_state: bool = True) -> None:
    protocol = payload.get("protocol")
    runs = payload.get("runs")
    if not isinstance(protocol, dict) or protocol.get("kind") != "neural_transition_engine":
        raise ValueError("payload is not a neural transition experiment")
    if not isinstance(runs, list) or not runs:
        raise ValueError("payload contains no neural runs")
    seen: set[tuple[str, int]] = set()
    for run in runs:
        key = (str(run.get("variant")), int(run.get("seed", -1)))
        if key in seen:
            raise ValueError(f"duplicate neural run: {key}")
        seen.add(key)
        if not math.isfinite(float(run.get("best_validation_loss", float("nan")))):
            raise ValueError(f"run {key} has an invalid validation loss")
        if require_state and (not isinstance(run.get("state_dict"), dict) or not run["state_dict"]):
            raise ValueError(f"run {key} has no neural state dictionary")
        if not run.get("history"):
            raise ValueError(f"run {key} has no training history")


def select_neural_run(payload: dict[str, Any], *, require_state: bool = True) -> dict[str, Any]:
    validate_neural_payload(payload, require_state=require_state)
    candidates = [run for run in payload["runs"] if run["variant"] == "+skill_weather"]
    if not candidates:
        candidates = payload["runs"]
    return min(candidates, key=lambda run: float(run["best_validation_loss"]))


def _summary(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{mean(values):.3f} ± {pstdev(values):.3f}"


def write_neural_summary(payload: dict[str, Any], output: Path) -> None:
    validate_neural_payload(payload, require_state=False)
    protocol = payload["protocol"]
    runs = payload["runs"]
    selected = select_neural_run(payload, require_state=False)
    evaluated = [run for run in runs if run.get("test")]
    lines = [
        "# Neural Mackerel Road results",
        "",
        "## Run",
        "",
        f"- Device: `{protocol['cuda_device']}`",
        f"- Training maps: `{protocol['train_maps']}`",
        f"- Validation maps: `{protocol['validation_maps']}`",
        f"- Held-out maps per model seed: `{protocol['test_maps']}`",
        "- Held-out map set shared across model seeds: `yes`",
        "- Across-seed summary: `mean ± population SD` (not a map-level CI)",
        f"- Steps: `{protocol['steps']}`",
        f"- Events: `{', '.join(protocol['events'])}`",
        "- Transition targets: `none`",
        "- Teacher engine: `none`",
        "",
        "## Optimization",
        "",
        "| Variant | Seed | Best step | Validation loss | Seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {run['variant']} | {run['seed']} | {run['best_step']} | "
        f"{run['best_validation_loss']:.4f} | {run['elapsed_seconds']:.1f} |"
        for run in runs
    )
    lines.extend(
        [
            "",
            "## Selected run",
            "",
            f"- Variant: `{selected['variant']}`",
            f"- Seed: `{selected['seed']}`",
            f"- Validation loss: `{selected['best_validation_loss']:.4f}`",
        ]
    )
    if evaluated:
        lines.extend(
            [
                "",
                "## Held-out hard games",
                "",
                "| Variant | Light win | Strong win | Failure | Freshness | Events | "
                "Skill leverage |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for variant in dict.fromkeys(str(run["variant"]) for run in evaluated):
            tests = [run["test"] for run in evaluated if run["variant"] == variant]
            columns = (
                _summary([test["equal_skill_light"] for test in tests]),
                _summary([test["equal_skill_strong"] for test in tests]),
                _summary([test["failure_rate"] for test in tests]),
                _summary([test["mean_final_freshness"] for test in tests]),
                _summary([test["event_rate"] for test in tests]),
                _summary([test["skill_leverage"] for test in tests]),
            )
            lines.append(f"| {variant} | " + " | ".join(columns) + " |")

        selected_test = selected.get("test")
        if selected_test:
            tolerance = selected_test["skill_tolerance"]
            tolerance_text = "undefined" if tolerance is None else f"{tolerance:.3f}"
            event_frequencies = ", ".join(
                f"{name}={value:.3f}" for name, value in selected_test["event_frequencies"].items()
            )
            semantic_penalties = ", ".join(
                f"{name.removeprefix('semantic_')}={value:.4f}"
                for name, value in selected_test["semantic_penalties"].items()
            )
            points = ", ".join(
                f"{epsilon:.2f}:{score:.3f}" for epsilon, score in selected_test["cgq_points"]
            )
            lines.extend(
                [
                    "",
                    "## Selected held-out diagnostics",
                    "",
                    f"- Equal-skill light/strong/draw: "
                    f"`{selected_test['equal_skill_light']:.3f}/"
                    f"{selected_test['equal_skill_strong']:.3f}/"
                    f"{selected_test['equal_skill_draw']:.3f}`",
                    f"- Exact-choice light/strong/draw: "
                    f"`{selected_test['exact_light']:.3f}/"
                    f"{selected_test['exact_strong']:.3f}/"
                    f"{selected_test['exact_draw']:.3f}`",
                    f"- Skill tolerance: `{tolerance_text}`",
                    f"- Mean freshness margin: `{selected_test['mean_freshness_margin']:.3f}`",
                    f"- Event frequencies: `{event_frequencies}`",
                    f"- Semantic penalties: `{semantic_penalties}`",
                    "",
                    "## Selected CGQ profile",
                    "",
                    f"- Epsilon to weaker-player score: `{points}`",
                ]
            )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "- [Objective ablation](../report/figures/neural-ablation.png)",
            "- [Held-out skill profile](../report/figures/neural-skill-profile.png)",
            "- [Event frequencies](../report/figures/neural-event-frequencies.png)",
            "- [Validation history](../report/figures/neural-training.png)",
            "- [Compact report overview](../report/figures/neural-report-overview.png)",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--summary", type=Path, default=Path("results/README.md"))
    parser.add_argument("--selected", type=Path, default=Path("results/selected_model.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    selected = select_neural_run(payload)
    args.selected.parent.mkdir(parents=True, exist_ok=True)
    args.selected.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    write_neural_summary(payload, args.summary)
    print(f"wrote {args.summary} and {args.selected}")


if __name__ == "__main__":
    main()
