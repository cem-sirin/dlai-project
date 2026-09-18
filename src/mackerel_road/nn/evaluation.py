"""Hard sampled races and held-out diagnostics for a neural mechanism."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise
from random import Random
from statistics import mean

import torch

from mackerel_road.game.maps import CLEAR_SKY, initial_cloud_node, move_cloud
from mackerel_road.game.types import PorterKind, PorterState, Road, RouteMap
from mackerel_road.nn.mechanics import (
    EventKind,
    NeuralMechanics,
    event_mask_from_terrain,
    role_features,
)
from mackerel_road.nn.objective import semantic_penalty
from mackerel_road.nn.planning import (
    CompiledRouteMap,
    compile_route_map,
    plan_from_state,
)


@dataclass(frozen=True)
class NeuralRoadOutcome:
    travel_time: float
    stamina_loss: float
    freshness_loss: float
    event: EventKind
    event_probability: float
    raining: bool


@dataclass(frozen=True)
class NeuralRoundRecord:
    cloud_node: str
    light_road: Road | None
    strong_road: Road | None
    light_outcome: NeuralRoadOutcome | None
    strong_outcome: NeuralRoadOutcome | None
    light_after: PorterState
    strong_after: PorterState
    next_cloud_node: str


@dataclass(frozen=True)
class NeuralMatchResult:
    records: tuple[NeuralRoundRecord, ...]
    light: PorterState
    strong: PorterState
    winner: PorterKind | None


@dataclass(frozen=True)
class NeuralDiagnostics:
    equal_skill_light: float
    equal_skill_strong: float
    equal_skill_draw: float
    exact_light: float
    exact_strong: float
    exact_draw: float
    cgq_points: tuple[tuple[float, float], ...]
    skill_leverage: float
    skill_tolerance: float | None
    failure_rate: float
    mean_final_freshness: float
    mean_freshness_margin: float
    event_rate: float
    event_frequencies: dict[str, float]
    semantic_penalties: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _initial_stamina(kind: PorterKind) -> float:
    return 15.0 if kind is PorterKind.LIGHT else 17.0


def _cloud_index(compiled: CompiledRouteMap, cloud_node: str) -> int:
    return compiled.clear_sky if cloud_node == CLEAR_SKY else compiled.node_index[cloud_node]


def _choose_road(
    model: NeuralMechanics,
    route_map: RouteMap,
    compiled: CompiledRouteMap,
    porter: PorterState,
    cloud_node: str,
    skill: float,
    rng: Random,
    *,
    cargo_load: float,
) -> Road | None:
    if porter.finished:
        return None
    roads = route_map.outgoing(porter.node)
    if rng.random() > skill:
        return rng.choice(roads)
    with torch.no_grad():
        plan = plan_from_state(
            model,
            compiled,
            porter.kind,
            node=compiled.node_index[porter.node],
            cloud=_cloud_index(compiled, cloud_node),
            stamina=porter.stamina / _initial_stamina(porter.kind),
            freshness=porter.freshness,
            elapsed_time=porter.elapsed_time,
            cargo_load=cargo_load,
            hard=True,
        )
    scores = plan.first_action_score[0]
    legal_scores = scores[: len(roads)]
    best = float(legal_scores.max().cpu())
    choices = [
        road
        for road, score in zip(roads, legal_scores, strict=True)
        if abs(float(score.cpu()) - best) <= 1e-9
    ]
    return rng.choice(choices)


def _sample_event(probabilities: torch.Tensor, uniform: float) -> int:
    cumulative = 0.0
    last_allowed = EventKind.NONE
    for index, probability in enumerate(probabilities.detach().cpu().tolist()):
        if probability > 0.0:
            last_allowed = index
        cumulative += float(probability)
        if uniform <= cumulative:
            return index
    return int(last_allowed)


def _apply_neural_transition(
    model: NeuralMechanics,
    route_map: RouteMap,
    compiled: CompiledRouteMap,
    porter: PorterState,
    road: Road | None,
    cloud_node: str,
    event_rng: Random,
    *,
    cargo_load: float,
) -> tuple[PorterState, NeuralRoadOutcome | None]:
    if porter.finished:
        return porter, None
    assert road is not None
    road_index = route_map.roads.index(road)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    road_features = compiled.road_features[road_index].to(dtype=dtype)
    features = torch.cat(
        (
            role_features(porter.kind, device=device).to(dtype=dtype),
            torch.tensor(
                (
                    porter.stamina / _initial_stamina(porter.kind),
                    porter.freshness,
                    cargo_load,
                ),
                dtype=dtype,
                device=device,
            ),
            road_features,
            torch.tensor((float(road.destination == cloud_node),), dtype=dtype, device=device),
        )
    ).unsqueeze(0)
    event_mask = event_mask_from_terrain(compiled.road_terrain[road_index].reshape(1))
    with torch.no_grad():
        transition = model(features, event_mask)
    event_id = _sample_event(transition.event_probability[0], event_rng.random())
    event = EventKind(event_id)
    travel_time = float(transition.travel_time[0, event_id].cpu())
    next_stamina = float(transition.next_stamina[0, 0].cpu()) * _initial_stamina(porter.kind)
    next_freshness = float(transition.next_freshness[0, event_id].cpu())
    failed = next_stamina <= 0.0 or next_freshness <= 0.0
    after = PorterState(
        kind=porter.kind,
        node=road.destination,
        elapsed_time=porter.elapsed_time + travel_time,
        stamina=next_stamina,
        freshness=0.0 if failed else next_freshness,
        finished=failed or road.destination == route_map.destination,
    )
    return after, NeuralRoadOutcome(
        travel_time=travel_time,
        stamina_loss=porter.stamina - next_stamina,
        freshness_loss=porter.freshness - next_freshness,
        event=event,
        event_probability=float(transition.event_probability[0, event_id].cpu()),
        raining=road.destination == cloud_node,
    )


def play_neural_match(
    model: NeuralMechanics,
    route_map: RouteMap,
    *,
    cloud_seed: int,
    event_seed: int,
    decision_seed: int,
    light_skill: float,
    strong_skill: float,
    cargo_load: float = 0.75,
) -> NeuralMatchResult:
    device = next(model.parameters()).device
    compiled = compile_route_map(route_map, device)
    cloud_rng = Random(cloud_seed)
    light_event_rng = Random(event_seed)
    strong_event_rng = Random(event_seed + 1)
    light_decision_rng = Random(decision_seed)
    strong_decision_rng = Random(decision_seed + 1)
    cloud_node = initial_cloud_node(route_map, cloud_rng)
    light = PorterState(PorterKind.LIGHT, route_map.start, 0.0, 15.0, 1.0)
    strong = PorterState(PorterKind.STRONG, route_map.start, 0.0, 17.0, 1.0)
    records: list[NeuralRoundRecord] = []

    model.eval()
    while not (light.finished and strong.finished):
        light_road = _choose_road(
            model,
            route_map,
            compiled,
            light,
            cloud_node,
            light_skill,
            light_decision_rng,
            cargo_load=cargo_load,
        )
        strong_road = _choose_road(
            model,
            route_map,
            compiled,
            strong,
            cloud_node,
            strong_skill,
            strong_decision_rng,
            cargo_load=cargo_load,
        )
        light_after, light_outcome = _apply_neural_transition(
            model,
            route_map,
            compiled,
            light,
            light_road,
            cloud_node,
            light_event_rng,
            cargo_load=cargo_load,
        )
        strong_after, strong_outcome = _apply_neural_transition(
            model,
            route_map,
            compiled,
            strong,
            strong_road,
            cloud_node,
            strong_event_rng,
            cargo_load=cargo_load,
        )
        next_cloud = move_cloud(route_map, cloud_node, cloud_rng)
        records.append(
            NeuralRoundRecord(
                cloud_node=cloud_node,
                light_road=light_road,
                strong_road=strong_road,
                light_outcome=light_outcome,
                strong_outcome=strong_outcome,
                light_after=light_after,
                strong_after=strong_after,
                next_cloud_node=next_cloud,
            )
        )
        light, strong, cloud_node = light_after, strong_after, next_cloud
        if len(records) > len(route_map.layers) + 2:
            raise RuntimeError("neural race exceeded the layered-map round cap")

    difference = light.freshness - strong.freshness
    winner = None
    if abs(difference) > 1e-9:
        winner = PorterKind.LIGHT if difference > 0.0 else PorterKind.STRONG
    return NeuralMatchResult(tuple(records), light, strong, winner)


def _rates(winners: list[PorterKind | None]) -> tuple[float, float, float]:
    counts = Counter(winners)
    total = len(winners)
    return (
        counts[PorterKind.LIGHT] / total,
        counts[PorterKind.STRONG] / total,
        counts[None] / total,
    )


def _focal_score(winner: PorterKind | None, focal: PorterKind) -> float:
    if winner is focal:
        return 1.0
    if winner is None:
        return 0.5
    return 0.0


def evaluate_neural_model(
    model: NeuralMechanics,
    maps: tuple[RouteMap, ...],
    *,
    epsilons: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> NeuralDiagnostics:
    if not maps:
        raise ValueError("at least one held-out map is required")
    equal_winners: list[PorterKind | None] = []
    exact_winners: list[PorterKind | None] = []
    equal_matches: list[NeuralMatchResult] = []
    for index, route_map in enumerate(maps):
        shared = {
            "cloud_seed": 5_000_000 + index,
            "event_seed": 6_000_000 + 2 * index,
            "decision_seed": 7_000_000 + 2 * index,
        }
        equal = play_neural_match(model, route_map, **shared, light_skill=0.8, strong_skill=0.8)
        exact = play_neural_match(model, route_map, **shared, light_skill=1.0, strong_skill=1.0)
        equal_winners.append(equal.winner)
        exact_winners.append(exact.winner)
        equal_matches.append(equal)

    profile = []
    for epsilon in epsilons:
        score = 0.0
        for index, route_map in enumerate(maps):
            shared = {
                "cloud_seed": 8_000_000 + index,
                "event_seed": 9_000_000 + 2 * index,
                "decision_seed": 10_000_000 + 2 * index,
            }
            focal_light = play_neural_match(
                model,
                route_map,
                **shared,
                light_skill=1.0 - epsilon,
                strong_skill=1.0,
            )
            focal_strong = play_neural_match(
                model,
                route_map,
                **shared,
                light_skill=1.0,
                strong_skill=1.0 - epsilon,
            )
            score += _focal_score(focal_light.winner, PorterKind.LIGHT)
            score += _focal_score(focal_strong.winner, PorterKind.STRONG)
        profile.append((epsilon, score / (2 * len(maps))))

    leverage = profile[0][1] - profile[-1][1]
    tolerance = None
    if leverage > 0.0:
        retention = [(epsilon, (score - profile[-1][1]) / leverage) for epsilon, score in profile]
        tolerance = sum(
            (right[0] - left[0]) * (left[1] + right[1]) / 2.0 for left, right in pairwise(retention)
        )

    all_porters = [porter for match in equal_matches for porter in (match.light, match.strong)]
    all_outcomes = [
        outcome
        for match in equal_matches
        for record in match.records
        for outcome in (record.light_outcome, record.strong_outcome)
        if outcome is not None
    ]
    events = Counter(outcome.event.name for outcome in all_outcomes)
    semantic_total, semantic_parts = semantic_penalty(model)
    del semantic_total
    equal_rates = _rates(equal_winners)
    exact_rates = _rates(exact_winners)
    return NeuralDiagnostics(
        equal_skill_light=equal_rates[0],
        equal_skill_strong=equal_rates[1],
        equal_skill_draw=equal_rates[2],
        exact_light=exact_rates[0],
        exact_strong=exact_rates[1],
        exact_draw=exact_rates[2],
        cgq_points=tuple(profile),
        skill_leverage=leverage,
        skill_tolerance=tolerance,
        failure_rate=sum(porter.freshness <= 0.0 or porter.stamina <= 0.0 for porter in all_porters)
        / len(all_porters),
        mean_final_freshness=mean(porter.freshness for porter in all_porters),
        mean_freshness_margin=mean(
            abs(match.light.freshness - match.strong.freshness) for match in equal_matches
        ),
        event_rate=sum(outcome.event is not EventKind.NONE for outcome in all_outcomes)
        / len(all_outcomes),
        event_frequencies={
            event.name: events[event.name] / len(all_outcomes) for event in EventKind
        },
        semantic_penalties={
            name: float(value.detach().cpu()) for name, value in semantic_parts.items()
        },
    )
