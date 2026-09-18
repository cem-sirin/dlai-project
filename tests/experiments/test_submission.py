from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from mackerel_road.experiments.plots import VARIANTS
from mackerel_road.experiments.protocol import PROFILE_EPSILONS
from mackerel_road.experiments.submission import build_submission, validate_submission


def _complete_fixture():
    return {
        "protocol": {
            "kind": "neural_transition_engine",
            "objective_reduction": "pooled_maps",
            "corruption_reference": "fixed_zero_corruption_policy",
            "optimizer_seeds": [0, 1, 2],
            "train_seed_start": 0,
            "train_maps": 64,
            "validation_seed_start": 10000,
            "validation_maps": 16,
            "test_seed_start": 30000,
            "test_maps": 100,
        },
        "runs": [
            {
                "variant": variant,
                "seed": seed,
                "best_step": 10,
                "best_validation_loss": 0.25,
                "history": [{"step": 10, "validation_total": 0.25}],
                "test": {"cgq_points": [[epsilon, 0.5] for epsilon in PROFILE_EPSILONS]},
            }
            for variant in VARIANTS
            for seed in range(3)
        ],
    }


def test_submission_accepts_complete_compact_records():
    validate_submission(_complete_fixture())


def test_submission_builds_report_without_docs_directory(tmp_path: Path):
    source = Path(__file__).resolve().parents[2] / "results" / "neural-evaluated.json"
    build_submission(source, tmp_path)
    assert not (tmp_path / "docs").exists()
    assert (tmp_path / "report" / "results.tex").is_file()
    assert (tmp_path / "report" / "figures" / "neural-report-overview.pdf").is_file()
    assert (tmp_path / "report" / "figures" / "neural-surrogate-diagnostics.pdf").is_file()
    summary = (tmp_path / "results" / "README.md").read_text()
    assert "../report/figures/neural-ablation.png" in summary


@pytest.mark.parametrize("failure", ["missing_run", "missing_test", "overlap", "checkpoint"])
def test_submission_rejects_incomplete_or_inconsistent_evidence(failure):
    payload = deepcopy(_complete_fixture())
    if failure == "missing_run":
        payload["runs"].pop()
    elif failure == "missing_test":
        payload["runs"][0]["test"] = None
    elif failure == "overlap":
        payload["protocol"]["test_seed_start"] = 0
    else:
        payload["runs"][0]["best_validation_loss"] = 0.5
    with pytest.raises(ValueError):
        validate_submission(payload)
