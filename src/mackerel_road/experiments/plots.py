"""Generate plots for the direct neural-transition experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal, overload

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from mackerel_road.experiments.protocol import (
    PROFILE_EPSILONS,
    TARGET_WIN_PROFILE,
)
from mackerel_road.experiments.summarize import validate_neural_payload

VARIANTS = ("viability", "+balance", "+specialization", "+skill_weather")
VARIANT_LABELS = ("Base", "+Balance", "+Role", "Full")
EVENTS = ("NONE", "LANDSLIDE", "FALLEN_TREE", "CART_BLOCKAGE")
EVENT_LABELS = ("No event", "Landslide", "Fallen tree", "Cart blockage")
LIGHT_COLOR = "#d97757"
STRONG_COLOR = "#3d5a80"
ACCENT_COLOR = "#2563eb"
TARGET_COLOR = "#6b7280"
GRID_COLOR = "#d1d5db"


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_neural_payload(payload, require_state=False)
    return payload


def _tests(payload: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    tests = [
        run["test"]
        for run in payload["runs"]
        if run["variant"] == variant and run.get("test") is not None
    ]
    if not tests:
        raise ValueError(f"payload has no evaluated runs for {variant}")
    return tests


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "font.size": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


@overload
def _finish(
    figure: Any,
    output_dir: Path,
    stem: str,
    *,
    write_pdf: Literal[False] = False,
) -> tuple[Path]: ...


@overload
def _finish(
    figure: Any,
    output_dir: Path,
    stem: str,
    *,
    write_pdf: Literal[True],
) -> tuple[Path, Path]: ...


def _finish(
    figure: Any,
    output_dir: Path,
    stem: str,
    *,
    write_pdf: bool = False,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    figure.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    if not write_pdf:
        return (png,)
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    return png, pdf


def plot_ablation(ablation: dict[str, Any], final: dict[str, Any], output_dir: Path) -> tuple[Path]:
    """Plot equal-skill porter balance and skill leverage by objective stage."""
    light_means: list[float] = []
    light_sds: list[float] = []
    strong_means: list[float] = []
    strong_sds: list[float] = []
    leverage_means: list[float] = []
    leverage_sds: list[float] = []
    for variant in VARIANTS:
        tests = _tests(ablation, variant)
        light_mean, light_sd = _mean_sd([test["equal_skill_light"] for test in tests])
        strong_mean, strong_sd = _mean_sd([test["equal_skill_strong"] for test in tests])
        leverage_mean, leverage_sd = _mean_sd([test["skill_leverage"] for test in tests])
        light_means.append(light_mean)
        light_sds.append(light_sd)
        strong_means.append(strong_mean)
        strong_sds.append(strong_sd)
        leverage_means.append(leverage_mean)
        leverage_sds.append(leverage_sd)

    positions = list(range(len(VARIANTS)))
    width = 0.36
    figure, (balance_axis, leverage_axis) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    balance_axis.bar(
        [position - width / 2 for position in positions],
        light_means,
        width,
        yerr=light_sds,
        capsize=3,
        label="Light",
        color=LIGHT_COLOR,
    )
    balance_axis.bar(
        [position + width / 2 for position in positions],
        strong_means,
        width,
        yerr=strong_sds,
        capsize=3,
        label="Strong",
        color=STRONG_COLOR,
    )
    balance_axis.axhline(0.5, color=TARGET_COLOR, linestyle="--", linewidth=1)
    balance_axis.set_title("Equal-skill porter wins")
    balance_axis.set_ylabel("Win rate")
    balance_axis.set_ylim(0.0, 1.02)
    balance_axis.set_xticks(positions, VARIANT_LABELS)
    balance_axis.legend(frameon=False, ncols=2, loc="upper right")
    balance_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)

    leverage_axis.bar(
        positions,
        leverage_means,
        yerr=leverage_sds,
        capsize=3,
        color=ACCENT_COLOR,
        width=0.62,
    )
    leverage_axis.set_title("Competence sensitivity")
    leverage_axis.set_ylabel("Skill leverage")
    leverage_axis.set_ylim(0.0, 0.52)
    leverage_axis.set_xticks(positions, VARIANT_LABELS)
    leverage_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    figure.suptitle("Neural objective ablation", fontsize=12, fontweight="bold")
    figure.tight_layout()
    return _finish(figure, output_dir, "neural-ablation")


def plot_skill_profile(final: dict[str, Any], output_dir: Path) -> tuple[Path]:
    """Plot the held-out corruption profile against its training target."""
    tests = _tests(final, "+skill_weather")
    means: list[float] = []
    sds: list[float] = []
    for index in range(len(PROFILE_EPSILONS)):
        point_mean, point_sd = _mean_sd([test["cgq_points"][index][1] for test in tests])
        means.append(point_mean)
        sds.append(point_sd)

    lower = [max(0.0, value - spread) for value, spread in zip(means, sds, strict=True)]
    upper = [min(1.0, value + spread) for value, spread in zip(means, sds, strict=True)]
    figure, axis = plt.subplots(figsize=(4.0, 2.8))
    axis.plot(
        PROFILE_EPSILONS,
        TARGET_WIN_PROFILE,
        marker="o",
        linestyle="--",
        color=TARGET_COLOR,
        label="Target",
    )
    axis.plot(
        PROFILE_EPSILONS,
        means,
        marker="o",
        linewidth=2,
        color=ACCENT_COLOR,
        label="Held-out hard play",
    )
    axis.fill_between(PROFILE_EPSILONS, lower, upper, color=ACCENT_COLOR, alpha=0.16)
    axis.set_title("Focal-player score under action corruption")
    axis.set_xlabel(r"Corruption level $\epsilon$")
    axis.set_ylabel("Focal-player score")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(0.0, 0.55)
    axis.set_xticks(PROFILE_EPSILONS)
    axis.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    axis.legend(frameon=False)
    figure.tight_layout()
    return _finish(figure, output_dir, "neural-skill-profile")


def plot_event_frequencies(final: dict[str, Any], output_dir: Path) -> tuple[Path]:
    """Plot frequencies of sampled held-out road events."""
    tests = _tests(final, "+skill_weather")
    summaries = [_mean_sd([test["event_frequencies"][event] for test in tests]) for event in EVENTS]
    means = [summary[0] for summary in summaries]
    sds = [summary[1] for summary in summaries]
    positions = list(range(len(EVENTS)))
    colors = ("#9ca3af", "#8b5e3c", "#2f855a", "#b7791f")

    figure, axis = plt.subplots(figsize=(4.5, 2.8))
    axis.bar(positions, means, yerr=sds, capsize=3, color=colors, width=0.65)
    axis.set_title("Held-out sampled event frequencies")
    axis.set_ylabel("Fraction of traversals")
    axis.set_ylim(0.0, 0.86)
    axis.set_xticks(positions, EVENT_LABELS, rotation=16, ha="right")
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    figure.tight_layout()
    return _finish(figure, output_dir, "neural-event-frequencies")


def plot_training(final: dict[str, Any], output_dir: Path) -> tuple[Path]:
    """Plot validation-objective trajectories for the three full models."""
    runs = [run for run in final["runs"] if run["variant"] == "+skill_weather"]
    if not runs:
        raise ValueError("final payload has no full-objective runs")

    figure, axis = plt.subplots(figsize=(4.5, 2.8))
    for run in runs:
        steps = [record["step"] for record in run["history"]]
        values = [record["validation_total"] for record in run["history"]]
        axis.plot(steps, values, marker="o", markersize=3, label=f"Seed {run['seed']}")
    axis.set_title("Full-objective validation history")
    axis.set_xlabel("Optimization step")
    axis.set_ylabel("Validation objective")
    axis.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    axis.legend(frameon=False, ncols=3)
    figure.tight_layout()
    return _finish(figure, output_dir, "neural-training")


def plot_report_overview(
    ablation: dict[str, Any], final: dict[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    """Plot the three primary results in a compact two-column-report figure."""
    light_means: list[float] = []
    light_sds: list[float] = []
    strong_means: list[float] = []
    strong_sds: list[float] = []
    leverage_means: list[float] = []
    leverage_sds: list[float] = []
    for variant in VARIANTS:
        tests = _tests(ablation, variant)
        light_mean, light_sd = _mean_sd([test["equal_skill_light"] for test in tests])
        strong_mean, strong_sd = _mean_sd([test["equal_skill_strong"] for test in tests])
        leverage_mean, leverage_sd = _mean_sd([test["skill_leverage"] for test in tests])
        light_means.append(light_mean)
        light_sds.append(light_sd)
        strong_means.append(strong_mean)
        strong_sds.append(strong_sd)
        leverage_means.append(leverage_mean)
        leverage_sds.append(leverage_sd)

    final_tests = _tests(final, "+skill_weather")
    profile_means: list[float] = []
    profile_sds: list[float] = []
    for index in range(len(PROFILE_EPSILONS)):
        point_mean, point_sd = _mean_sd([test["cgq_points"][index][1] for test in final_tests])
        profile_means.append(point_mean)
        profile_sds.append(point_sd)
    profile_lower = [
        max(0.0, value - spread) for value, spread in zip(profile_means, profile_sds, strict=True)
    ]
    profile_upper = [
        min(1.0, value + spread) for value, spread in zip(profile_means, profile_sds, strict=True)
    ]

    positions = list(range(len(VARIANTS)))
    width = 0.36
    report_labels = ("Base", "+Bal.", "+Role", "Full")
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 1.5))
    balance_axis, leverage_axis, profile_axis = axes
    balance_axis.bar(
        [position - width / 2 for position in positions],
        light_means,
        width,
        yerr=light_sds,
        capsize=2,
        color=LIGHT_COLOR,
        label="Light",
    )
    balance_axis.bar(
        [position + width / 2 for position in positions],
        strong_means,
        width,
        yerr=strong_sds,
        capsize=2,
        color=STRONG_COLOR,
        label="Strong",
    )
    balance_axis.axhline(0.5, color=TARGET_COLOR, linestyle="--", linewidth=0.8)
    balance_axis.set_title("(a) Equal-skill wins", fontsize=8)
    balance_axis.set_ylabel("Win rate", fontsize=7)
    balance_axis.set_ylim(0.0, 1.02)
    balance_axis.set_xticks(positions, report_labels)
    balance_axis.legend(frameon=False, ncols=2, fontsize=6, loc="upper right")

    leverage_axis.bar(
        positions,
        leverage_means,
        yerr=leverage_sds,
        capsize=2,
        color=ACCENT_COLOR,
        width=0.62,
    )
    leverage_axis.set_title("(b) Competence sensitivity", fontsize=8)
    leverage_axis.set_ylabel("Skill leverage", fontsize=7)
    leverage_axis.set_ylim(0.0, 0.52)
    leverage_axis.set_xticks(positions, report_labels)

    profile_axis.plot(
        PROFILE_EPSILONS,
        TARGET_WIN_PROFILE,
        marker="o",
        markersize=3,
        linestyle="--",
        color=TARGET_COLOR,
        label="Target",
    )
    profile_axis.plot(
        PROFILE_EPSILONS,
        profile_means,
        marker="o",
        markersize=3,
        color=ACCENT_COLOR,
        label="Hard play",
    )
    profile_axis.fill_between(
        PROFILE_EPSILONS,
        profile_lower,
        profile_upper,
        color=ACCENT_COLOR,
        alpha=0.16,
    )
    profile_axis.set_title("(c) Corruption profile", fontsize=8)
    profile_axis.set_xlabel(r"Corruption $\epsilon$", fontsize=7)
    profile_axis.set_ylabel("Focal score", fontsize=7)
    profile_axis.set_xlim(-0.02, 1.02)
    profile_axis.set_ylim(0.0, 0.55)
    profile_axis.set_xticks(PROFILE_EPSILONS)
    profile_axis.legend(frameon=False, fontsize=6)

    for axis in axes:
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.7)
        axis.tick_params(labelsize=6.5)
    figure.tight_layout(pad=0.25, w_pad=0.6)
    return _finish(figure, output_dir, "neural-report-overview", write_pdf=True)


def write_neural_plots(ablation_path: Path, final_path: Path, output_dir: Path) -> tuple[Path, ...]:
    """Load experiment payloads and generate all neural-result plots."""
    _style()
    ablation = _load_payload(ablation_path)
    final = _load_payload(final_path)
    outputs = (
        *plot_ablation(ablation, final, output_dir),
        *plot_skill_profile(final, output_dir),
        *plot_event_frequencies(final, output_dir),
        *plot_training(final, output_dir),
        *plot_report_overview(ablation, final, output_dir),
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ablation", type=Path)
    parser.add_argument("final", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("report/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_neural_plots(args.ablation, args.final, args.output_dir)
    print(f"wrote {len(outputs)} neural plot files to {args.output_dir}")


if __name__ == "__main__":
    main()
