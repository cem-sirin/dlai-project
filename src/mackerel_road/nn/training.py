"""Optimize the neural transition engine directly through induced games."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from mackerel_road.experiments.artifacts import CHECKPOINT_PREFIX, RESULT_PREFIX, encode_payload
from mackerel_road.experiments.protocol import (
    ABLATIONS,
    TEST_SEED_START,
    TRAIN_MAPS,
    TRAIN_SEED_START,
    VALIDATION_MAPS,
    VALIDATION_SEED_START,
    ObjectiveSpec,
)
from mackerel_road.experiments.summarize import validate_neural_payload
from mackerel_road.game.maps import generate_route_map
from mackerel_road.nn.evaluation import evaluate_neural_model
from mackerel_road.nn.mechanics import EventKind, NeuralMechanics
from mackerel_road.nn.objective import neural_game_objective


def _float_dict(values: dict[str, Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in values.items()}


def serializable_state_dict(model: NeuralMechanics) -> dict[str, list[float] | list[list[float]]]:
    return {name: value.detach().cpu().tolist() for name, value in model.state_dict().items()}


def load_serializable_state_dict(
    model: NeuralMechanics,
    values: Mapping[str, object],
) -> None:
    current = model.state_dict()
    state = {
        name: torch.as_tensor(values[name], dtype=tensor.dtype, device=tensor.device)
        for name, tensor in current.items()
    }
    model.load_state_dict(state)


def _select_specs(names: str) -> tuple[ObjectiveSpec, ...]:
    requested = tuple(part.strip() for part in names.split(",") if part.strip())
    by_name = {spec.name: spec for spec in ABLATIONS}
    missing = tuple(name for name in requested if name not in by_name)
    if missing:
        raise ValueError(f"unknown variants: {', '.join(missing)}")
    return tuple(by_name[name] for name in requested)


def _generate_pool(seed_start: int, count: int, horizon: int | None) -> tuple:
    return tuple(generate_route_map(seed_start + index, horizon=horizon) for index in range(count))


def train_one(
    spec: ObjectiveSpec,
    seed: int,
    *,
    device: torch.device,
    steps: int,
    batch_maps: int,
    learning_rate: float,
    patience: int,
    validation_interval: int,
    validation_maps_count: int,
    test_maps_count: int,
    horizon: int | None,
    test_seed_start: int = TEST_SEED_START,
) -> dict[str, object]:
    torch.manual_seed(seed)
    sampler = random.Random(seed)
    train_maps = _generate_pool(TRAIN_SEED_START, TRAIN_MAPS, horizon)
    validation_maps = _generate_pool(VALIDATION_SEED_START, validation_maps_count, horizon)
    model = NeuralMechanics().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_state = deepcopy(model.state_dict())
    best_validation = float("inf")
    best_step = 0
    stale_validations = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()

    for step in range(steps + 1):
        if step > 0:
            model.train()
            indices = sampler.sample(range(TRAIN_MAPS), batch_maps)
            optimizer.zero_grad(set_to_none=True)
            # Aggregate map statistics BEFORE the nonlinear balance/profile losses.
            # Averaging singleton losses instead forces balance on every map and
            # prevents complementary role advantages across the map population.
            result = neural_game_objective(
                model,
                tuple(train_maps[index] for index in indices),
                spec,
                checkpoint_plans=True,
            )
            result.total.backward()
            total = result.total.detach()
            train_components = _float_dict(result.components)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        else:
            total = torch.full((), float("nan"), device=device)
            train_components = {}

        if step % validation_interval != 0 and step != steps:
            continue
        model.eval()
        with torch.no_grad():
            validation = neural_game_objective(model, validation_maps, spec)
            validation_total = float(validation.total.cpu())
            validation_components = _float_dict(validation.components)
            validation_diagnostics = _float_dict(validation.diagnostics)
        row: dict[str, object] = {
            "step": step,
            "training_total": None if step == 0 else float(total.cpu()),
            "training_components": train_components,
            "validation_total": validation_total,
            "components": validation_components,
            "diagnostics": validation_diagnostics,
        }
        history.append(row)
        if validation_total < best_validation - 1e-7:
            best_validation = validation_total
            best_step = step
            best_state = deepcopy(model.state_dict())
            stale_validations = 0
        else:
            stale_validations += 1
        print(
            f"variant={spec.name} seed={seed} step={step} "
            f"train={float(total.cpu()):.6f} val={validation_total:.6f}",
            flush=True,
        )
        if stale_validations >= patience:
            break

    model.load_state_dict(best_state)
    test = None
    if test_maps_count:
        test_maps = _generate_pool(test_seed_start, test_maps_count, horizon)
        test = evaluate_neural_model(model, test_maps).to_dict()
    elapsed = time.perf_counter() - started
    return {
        "variant": spec.name,
        "seed": seed,
        "best_step": best_step,
        "best_validation_loss": best_validation,
        "elapsed_seconds": elapsed,
        "model_config": model.config(),
        "state_dict": serializable_state_dict(model),
        "test": test,
        "history": history,
    }


def load_completed_runs(path: Path, protocol: Mapping[str, object]) -> list[dict[str, object]]:
    """Resume completed seed runs only, rejecting any change to the experiment."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_neural_payload(payload)
    if payload["protocol"] != protocol:
        raise ValueError("resume protocol differs from the requested experiment")
    if protocol["test_maps"] and any(run.get("test") is None for run in payload["runs"]):
        raise ValueError("resume input contains a run without completed hard evaluation")
    return payload["runs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="+skill_weather")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-maps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--validation-maps", type=int, default=16)
    parser.add_argument("--test-maps", type=int, default=10)
    parser.add_argument("--test-seed-start", type=int, default=TEST_SEED_START)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path, help="Decoded payload of completed seed runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0 or args.steps <= 0 or args.batch_maps <= 0:
        raise SystemExit("seeds, steps, and batch-maps must be positive")
    if not 1 <= args.validation_maps <= VALIDATION_MAPS:
        raise SystemExit(f"validation-maps must be in [1, {VALIDATION_MAPS}]")
    if args.test_maps < 0:
        raise SystemExit("test-maps must be nonnegative")
    if args.batch_maps > TRAIN_MAPS:
        raise SystemExit(f"batch-maps must not exceed {TRAIN_MAPS}")
    if args.batch_maps < 2:
        raise SystemExit("batch-maps must be at least two for population-level objectives")
    if args.test_seed_start < VALIDATION_SEED_START + VALIDATION_MAPS:
        raise SystemExit("test-seed-start must be beyond the training and validation pools")
    specs = _select_specs(args.variants)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    protocol = {
        "kind": "neural_transition_engine",
        "objective_reduction": "pooled_maps",
        "corruption_reference": "fixed_zero_corruption_policy",
        "train_seed_start": TRAIN_SEED_START,
        "train_maps": TRAIN_MAPS,
        "validation_seed_start": VALIDATION_SEED_START,
        "validation_maps": args.validation_maps,
        "test_seed_start": args.test_seed_start,
        "test_maps": args.test_maps,
        "steps": args.steps,
        "batch_maps": args.batch_maps,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "validation_interval": args.validation_interval,
        "horizon": args.horizon,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "events": [event.name for event in EventKind],
        "ablations": [asdict(spec) for spec in specs],
        "optimizer_seeds": list(range(args.seeds)),
    }
    runs = [] if args.resume is None else load_completed_runs(args.resume, protocol)
    completed = {(run["variant"], run["seed"]) for run in runs}
    expected = {(spec.name, seed) for spec in specs for seed in range(args.seeds)}
    if not completed <= expected:
        raise SystemExit("resume input contains runs outside the requested experiment")
    for spec in specs:
        for seed in range(args.seeds):
            if (spec.name, seed) in completed:
                print(f"Resuming: retain completed variant={spec.name} seed={seed}", flush=True)
                continue
            run = train_one(
                spec,
                seed,
                device=device,
                steps=args.steps,
                batch_maps=args.batch_maps,
                learning_rate=args.learning_rate,
                patience=args.patience,
                validation_interval=args.validation_interval,
                validation_maps_count=args.validation_maps,
                test_maps_count=args.test_maps,
                horizon=args.horizon,
                test_seed_start=args.test_seed_start,
            )
            runs.append(run)
            print(
                CHECKPOINT_PREFIX + encode_payload({"protocol": protocol, "runs": runs}),
                flush=True,
            )

    print(RESULT_PREFIX + encode_payload({"protocol": protocol, "runs": runs}), flush=True)


if __name__ == "__main__":
    main()
