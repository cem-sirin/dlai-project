"""Build submission tables and figures from measured results, without PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from mackerel_road.experiments.plots import (
    ACCENT_COLOR,
    TARGET_COLOR,
    VARIANT_LABELS,
    VARIANTS,
    _finish,
    _style,
    plt,
    write_neural_plots,
)
from mackerel_road.experiments.protocol import PROFILE_EPSILONS, TARGET_WIN_PROFILE
from mackerel_road.experiments.summarize import (
    select_neural_run,
    validate_neural_payload,
    write_neural_summary,
)
from mackerel_road.game.maps import generate_route_map, route_count
from mackerel_road.game.types import Terrain


def _best(run: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in run["history"] if row["step"] == run["best_step"])


def _pm(values: list[float], digits: int = 3, scale: float = 1.0) -> str:
    return f"{scale * mean(values):.{digits}f}\\pm{scale * pstdev(values):.{digits}f}"


def validate_submission(payload: dict[str, Any]) -> None:
    """Reject partial runs or unmatched seed sets before generating the report."""
    validate_neural_payload(payload, require_state=False)
    protocol = payload["protocol"]
    seeds = set(protocol["optimizer_seeds"])
    if seeds != {0, 1, 2}:
        raise ValueError("submission tables require optimizer seeds 0, 1, and 2")
    expected = {(variant, seed) for variant in VARIANTS for seed in seeds}
    actual = {(run["variant"], run["seed"]) for run in payload["runs"]}
    if actual != expected:
        raise ValueError("submission requires all variants on the same optimizer seeds")
    if protocol.get("objective_reduction") != "pooled_maps":
        raise ValueError("submission requires the corrected pooled-map objective")
    if protocol.get("corruption_reference") != "fixed_zero_corruption_policy":
        raise ValueError("submission requires a fixed corruption reference")
    pools = [
        set(
            range(
                protocol[f"{name}_seed_start"],
                protocol[f"{name}_seed_start"] + protocol[f"{name}_maps"],
            )
        )
        for name in ("train", "validation", "test")
    ]
    if any(pools[i] & pools[j] for i in range(len(pools)) for j in range(i)):
        raise ValueError("training, validation, and test seeds must be disjoint")
    for run in payload["runs"]:
        if not run.get("test"):
            raise ValueError("submission requires completed hard evaluations")
        if tuple(point[0] for point in run["test"]["cgq_points"]) != PROFILE_EPSILONS:
            raise ValueError("unexpected corruption grid")
        if abs(_best(run)["validation_total"] - run["best_validation_loss"]) > 1e-6:
            raise ValueError("selected checkpoint does not match its validation record")


def write_tables(payload: dict[str, Any], report: Path) -> None:
    runs = payload["runs"]
    full = [run for run in runs if run["variant"] == VARIANTS[-1]]
    macros = {"SelectedSeed": str(select_neural_run(payload, require_state=False)["seed"])}
    metrics = {
        "LightWins": ("equal_skill_light", 1, 100.0),
        "StrongWins": ("equal_skill_strong", 1, 100.0),
        "ExactLightWins": ("exact_light", 1, 100.0),
        "Freshness": ("mean_final_freshness", 3, 1.0),
        "EventRate": ("event_rate", 1, 100.0),
        "Leverage": ("skill_leverage", 3, 1.0),
        "Tolerance": ("skill_tolerance", 3, 1.0),
    }
    for prefix, variant in zip(("Base", "Balance", "Role", "Full"), VARIANTS, strict=True):
        selected = [run for run in runs if run["variant"] == variant]
        for name, (key, digits, scale) in metrics.items():
            values = [run["test"][key] for run in selected]
            if all(value is not None for value in values):
                macros[prefix + name] = _pm(values, digits, scale)
    macros["FullProfileEnd"] = f"{mean(r['test']['cgq_points'][-1][1] for r in full):.3f}"
    macros["FullWeather"] = _pm([_best(r)["diagnostics"]["weather_effect"] for r in full], 4)
    report.mkdir(parents=True, exist_ok=True)
    (report / "results.tex").write_text(
        "% Generated from results/neural-evaluated.json; do not edit by hand.\n"
        + "\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items())
        + "\n",
        encoding="utf-8",
    )

    rows = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Variant & Light wins & Freshness & Event rate & Leverage & Tolerance\\",
        r"\midrule",
    ]
    for label, variant in zip(VARIANT_LABELS, VARIANTS, strict=True):
        selected = [run for run in runs if run["variant"] == variant]
        cells = []
        for key in (
            "equal_skill_light",
            "mean_final_freshness",
            "event_rate",
            "skill_leverage",
            "skill_tolerance",
        ):
            values = [run["test"][key] for run in selected]
            cells.append("--" if any(v is None for v in values) else "$" + _pm(values) + "$")
        rows.append(label + " & " + " & ".join(cells) + r"\\")
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    (report / "ablation-table.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Variant & Seed & Best step & Validation loss & Light wins & Failures & Seconds\\",
        r"\midrule",
    ]
    for run in runs:
        label = VARIANT_LABELS[VARIANTS.index(run["variant"])]
        rows.append(
            f"{label} & {run['seed']} & {run['best_step']} & "
            f"{run['best_validation_loss']:.4f} & {run['test']['equal_skill_light']:.3f} & "
            f"{run['test']['failure_rate']:.3f} & {run['elapsed_seconds']:.1f}" + r"\\"
        )
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    (report / "seed-table.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Quantity & Seed 0 & Seed 1 & Seed 2\\",
        r"\midrule",
    ]
    for name, key in (
        ("Mean role gap (strong minus light)", "mean_role_difference"),
        ("Mean absolute role gap", "mean_absolute_role_difference"),
        ("Strong-favored map fraction", "strong_favored_fraction"),
        ("Light-favored map fraction", "light_favored_fraction"),
        ("Cloud cost", "weather_effect"),
        ("Difficulty penalty", "semantic_difficulty"),
    ):
        rows.append(
            name
            + " & "
            + " & ".join(
                f"{_best(r)['diagnostics'][key]:.4f}" for r in sorted(full, key=lambda r: r["seed"])
            )
            + r"\\"
        )
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    (report / "diagnostics-table.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_supplement(payload: dict[str, Any], output: Path) -> None:
    _style()
    full = [run for run in payload["runs"] if run["variant"] == VARIANTS[-1]]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for label, values, color, linestyle in (
        ("Design target", [list(TARGET_WIN_PROFILE)], TARGET_COLOR, "--"),
        (
            "Validation surrogate",
            [
                [_best(run)["diagnostics"][f"profile_{e:.2f}"] for e in PROFILE_EPSILONS]
                for run in full
            ],
            "#d97757",
            "-",
        ),
        (
            "Held-out hard play",
            [[point[1] for point in run["test"]["cgq_points"]] for run in full],
            ACCENT_COLOR,
            "-",
        ),
    ):
        center = [mean(row[i] for row in values) for i in range(len(PROFILE_EPSILONS))]
        spread = [pstdev(row[i] for row in values) for i in range(len(PROFILE_EPSILONS))]
        axes[0].plot(
            PROFILE_EPSILONS,
            center,
            label=label,
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=3,
        )
        axes[0].fill_between(
            PROFILE_EPSILONS,
            [x - s for x, s in zip(center, spread, strict=True)],
            [x + s for x, s in zip(center, spread, strict=True)],
            color=color,
            alpha=0.1,
        )
    axes[0].set(
        xlabel=r"Corruption $\epsilon$",
        ylabel="Focal score",
        title="Surrogate and sampled outcomes",
        ylim=(0, 0.55),
    )
    axes[0].legend(frameon=False, fontsize=7)
    components = ("balance", "specialization", "skill_profile", "weather", "regularity")
    for run in full:
        best = _best(run)
        axes[1].plot(
            range(len(components)),
            [best["components"][key] for key in components],
            marker="o",
            markersize=3,
            label=f"Seed {run['seed']}",
        )
    axes[1].set_xticks(range(len(components)), ("Balance", "Role", "Skill", "Weather", "Reg."))
    axes[1].set(
        yscale="log", ylabel="Weighted loss (log scale)", title="Selected validation components"
    )
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    _finish(figure, output, "neural-surrogate-diagnostics", write_pdf=True)

    route_map = generate_route_map(30000)
    position = {
        node: (i, j - (len(layer) - 1) / 2)
        for i, layer in enumerate(route_map.layers)
        for j, node in enumerate(layer)
    }
    colors = {Terrain.MOUNTAIN: "#8b5e3c", Terrain.FOREST: "#2f855a", Terrain.POST_ROAD: "#3d5a80"}
    figure, axis = plt.subplots(figsize=(7.2, 2.2))
    seen: set[Terrain] = set()
    for road in route_map.roads:
        left, right = position[road.source], position[road.destination]
        axis.plot(
            (left[0], right[0]),
            (left[1], right[1]),
            color=colors[road.terrain],
            alpha=0.7,
            linewidth=1.0,
            label=road.terrain.value.replace("_", " ") if road.terrain not in seen else None,
        )
        seen.add(road.terrain)
    axis.scatter(
        [xy[0] for xy in position.values()],
        [xy[1] for xy in position.values()],
        s=35,
        color="#374151",
        zorder=3,
    )
    cloud_node = route_map.layers[1][0]
    axis.scatter(
        *position[cloud_node],
        s=200,
        facecolors="none",
        edgecolors=ACCENT_COLOR,
        linewidths=2,
        zorder=4,
        label="Illustrative cloud position",
    )
    axis.annotate(
        "Obama",
        position[route_map.start],
        xytext=(0, -18),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )
    axis.annotate(
        "Kyoto",
        position[route_map.destination],
        xytext=(0, -18),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )
    axis.set_title(
        f"Generated map (seed 30000): {len(route_map.layers) - 1} decisions, "
        f"{route_count(route_map)} complete routes"
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncols=4, frameon=False, fontsize=7)
    axis.set_axis_off()
    figure.tight_layout()
    _finish(figure, output, "neural-map", write_pdf=True)


def build_submission(input_path: Path, root: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    validate_submission(payload)
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    compact = {
        **payload,
        "runs": [
            {key: value for key, value in run.items() if key != "state_dict"}
            for run in payload["runs"]
        ],
    }
    compact.setdefault(
        "provenance", {"raw_payload_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}
    )
    data_path = results / "neural-evaluated.json"
    data_path.write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    if all(run.get("state_dict") for run in payload["runs"]):
        selected = select_neural_run(payload)
        (results / "selected_model.json").write_text(
            json.dumps(selected, indent=2) + "\n", encoding="utf-8"
        )
    write_neural_summary(compact, results / "README.md")
    write_neural_plots(data_path, data_path, root / "report" / "figures")
    write_tables(compact, root / "report")
    plot_supplement(compact, root / "report" / "figures")
    print(f"Built submission evidence from {len(compact['runs'])} measured runs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    build_submission(args.input, args.root)


if __name__ == "__main__":
    main()
