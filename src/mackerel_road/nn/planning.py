"""Differentiable full-state planning through the local neural mechanism."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mackerel_road.game.maps import CLEAR_SKY
from mackerel_road.game.types import PorterKind, RouteMap, Terrain
from mackerel_road.nn.mechanics import (
    NeuralMechanics,
    event_mask_from_terrain,
    role_features,
)


@dataclass(frozen=True)
class CompiledRouteMap:
    route_map: RouteMap
    nodes: tuple[str, ...]
    node_index: dict[str, int]
    start: int
    destination: int
    clear_sky: int
    initial_clouds: Tensor
    road_destination: Tensor
    road_terrain: Tensor
    road_features: Tensor
    outgoing_roads: Tensor
    outgoing_mask: Tensor
    cloud_destination: Tensor
    cloud_probability: Tensor
    cloud_mask: Tensor


@dataclass(frozen=True)
class PlannerResult:
    """Expected terminal metrics for every requested corruption level."""

    epsilons: Tensor
    final_score: Tensor
    final_freshness: Tensor
    final_stamina: Tensor
    elapsed_time: Tensor
    event_rate: Tensor
    event_delay_variance: Tensor
    first_action_score: Tensor
    first_action_mask: Tensor


@dataclass(frozen=True)
class _TreeLevel:
    state_count: int
    action_parent: Tensor
    action_slot: Tensor
    action_mask: Tensor
    child_action: Tensor
    child_probability: Tensor


def compile_route_map(route_map: RouteMap, device: torch.device) -> CompiledRouteMap:
    nodes = tuple(node for layer in route_map.layers for node in layer)
    node_index = {node: index for index, node in enumerate(nodes)}
    clear_sky = len(nodes)
    max_actions = max(len(route_map.outgoing(node)) for node in nodes[:-1])
    max_cloud_branches = max(
        1,
        max((len(route_map.outgoing(node)) for node in nodes), default=1),
    )
    road_index = {road.road_id: index for index, road in enumerate(route_map.roads)}

    outgoing_roads = torch.zeros((len(nodes), max_actions), dtype=torch.long, device=device)
    outgoing_mask = torch.zeros((len(nodes), max_actions), dtype=torch.bool, device=device)
    for node in nodes:
        roads = route_map.outgoing(node)
        for slot, road in enumerate(roads):
            outgoing_roads[node_index[node], slot] = road_index[road.road_id]
            outgoing_mask[node_index[node], slot] = True

    terrain_index = {terrain: index for index, terrain in enumerate(Terrain)}
    road_destination = torch.tensor(
        [node_index[road.destination] for road in route_map.roads],
        dtype=torch.long,
        device=device,
    )
    road_terrain = torch.tensor(
        [terrain_index[road.terrain] for road in route_map.roads],
        dtype=torch.long,
        device=device,
    )
    road_features = torch.tensor(
        [
            (
                road.distance / 5.5,
                road.difficulty,
                road.exposure,
                float(road.terrain is Terrain.MOUNTAIN),
                float(road.terrain is Terrain.FOREST),
                float(road.terrain is Terrain.POST_ROAD),
            )
            for road in route_map.roads
        ],
        dtype=torch.float32,
        device=device,
    )

    cloud_destination = torch.full(
        (len(nodes) + 1, max_cloud_branches),
        clear_sky,
        dtype=torch.long,
        device=device,
    )
    cloud_probability = torch.zeros(
        (len(nodes) + 1, max_cloud_branches), dtype=torch.float32, device=device
    )
    cloud_mask = torch.zeros((len(nodes) + 1, max_cloud_branches), dtype=torch.bool, device=device)
    for node in (*nodes, CLEAR_SKY):
        source_index = clear_sky if node == CLEAR_SKY else node_index[node]
        roads = () if node == CLEAR_SKY else route_map.outgoing(node)
        if not roads:
            cloud_destination[source_index, 0] = clear_sky
            cloud_probability[source_index, 0] = 1.0
            cloud_mask[source_index, 0] = True
            continue
        probability = 1.0 / len(roads)
        for slot, road in enumerate(roads):
            cloud_destination[source_index, slot] = (
                clear_sky
                if road.destination == route_map.destination
                else node_index[road.destination]
            )
            cloud_probability[source_index, slot] = probability
            cloud_mask[source_index, slot] = True

    return CompiledRouteMap(
        route_map=route_map,
        nodes=nodes,
        node_index=node_index,
        start=node_index[route_map.start],
        destination=node_index[route_map.destination],
        clear_sky=clear_sky,
        initial_clouds=torch.tensor(
            [node_index[node] for node in route_map.layers[1]],
            dtype=torch.long,
            device=device,
        ),
        road_destination=road_destination,
        road_terrain=road_terrain,
        road_features=road_features,
        outgoing_roads=outgoing_roads,
        outgoing_mask=outgoing_mask,
        cloud_destination=cloud_destination,
        cloud_probability=cloud_probability,
        cloud_mask=cloud_mask,
    )


def _terminal_metrics(
    stamina: Tensor,
    freshness: Tensor,
    elapsed: Tensor,
    *,
    hard: bool,
) -> Tensor:
    if hard:
        score = torch.where((stamina > 0.0) & (freshness > 0.0), freshness, 0.0)
        return torch.stack((score, freshness, stamina, elapsed), dim=-1)
    survival = torch.sigmoid((stamina - 0.025) / 0.01) * torch.sigmoid((freshness - 0.025) / 0.01)
    score = freshness * survival
    return torch.stack((score, freshness, stamina, elapsed), dim=-1)


def plan_from_state(
    model: NeuralMechanics,
    compiled: CompiledRouteMap,
    kind: PorterKind,
    *,
    node: int,
    cloud: int,
    stamina: float | Tensor,
    freshness: float | Tensor,
    elapsed_time: float | Tensor = 0.0,
    cargo_load: float = 0.75,
    epsilons: tuple[float, ...] = (0.0,),
    inverse_temperature: float = 35.0,
    hard: bool = False,
) -> PlannerResult:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    steps = (
        len(compiled.route_map.layers) - 1 - compiled.route_map.layer_index(compiled.nodes[node])
    )
    state_node = torch.tensor((node,), dtype=torch.long, device=device)
    state_cloud = torch.tensor((cloud,), dtype=torch.long, device=device)
    state_stamina = torch.as_tensor(stamina, dtype=dtype, device=device).reshape(1)
    state_freshness = torch.as_tensor(freshness, dtype=dtype, device=device).reshape(1)
    state_elapsed = torch.as_tensor(elapsed_time, dtype=dtype, device=device).reshape(1)
    return _plan_roots(
        model,
        compiled,
        kind,
        state_node=state_node,
        state_cloud=state_cloud,
        state_stamina=state_stamina,
        state_freshness=state_freshness,
        state_elapsed=state_elapsed,
        root_weight=torch.ones(1, dtype=dtype, device=device),
        steps=steps,
        cargo_load=cargo_load,
        epsilons=epsilons,
        inverse_temperature=inverse_temperature,
        hard=hard,
    )


def plan_initial_state(
    model: NeuralMechanics,
    route_map: RouteMap,
    kind: PorterKind,
    *,
    clouds: bool = True,
    cargo_load: float = 0.75,
    epsilons: tuple[float, ...] = (0.0,),
    inverse_temperature: float = 35.0,
    hard: bool = False,
) -> PlannerResult:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    compiled = compile_route_map(route_map, device)
    clouds_tensor = (
        compiled.initial_clouds
        if clouds
        else torch.tensor((compiled.clear_sky,), dtype=torch.long, device=device)
    )
    root_count = len(clouds_tensor)
    return _plan_roots(
        model,
        compiled,
        kind,
        state_node=torch.full((root_count,), compiled.start, dtype=torch.long, device=device),
        state_cloud=clouds_tensor,
        state_stamina=torch.ones(root_count, dtype=dtype, device=device),
        state_freshness=torch.ones(root_count, dtype=dtype, device=device),
        state_elapsed=torch.zeros(root_count, dtype=dtype, device=device),
        root_weight=torch.full((root_count,), 1.0 / root_count, dtype=dtype, device=device),
        steps=len(route_map.layers) - 1,
        cargo_load=cargo_load,
        epsilons=epsilons,
        inverse_temperature=inverse_temperature,
        hard=hard,
    )


def _plan_roots(
    model: NeuralMechanics,
    compiled: CompiledRouteMap,
    kind: PorterKind,
    *,
    state_node: Tensor,
    state_cloud: Tensor,
    state_stamina: Tensor,
    state_freshness: Tensor,
    state_elapsed: Tensor,
    root_weight: Tensor,
    steps: int,
    cargo_load: float,
    epsilons: tuple[float, ...],
    inverse_temperature: float,
    hard: bool,
) -> PlannerResult:
    if not epsilons:
        raise ValueError("at least one epsilon is required")
    if any(not 0.0 <= epsilon <= 1.0 for epsilon in epsilons):
        raise ValueError("epsilons must be in [0, 1]")
    if inverse_temperature <= 0.0:
        raise ValueError("inverse_temperature must be positive")
    device = state_stamina.device
    dtype = state_stamina.dtype
    role = role_features(kind, device=device).to(dtype=dtype)
    levels: list[_TreeLevel] = []
    event_rate_sum = torch.zeros((), dtype=dtype, device=device)
    event_variance_sum = torch.zeros((), dtype=dtype, device=device)
    event_rows = 0

    for _ in range(steps):
        action_mask = compiled.outgoing_mask[state_node]
        valid_actions = action_mask.nonzero(as_tuple=False)
        action_parent = valid_actions[:, 0]
        action_slot = valid_actions[:, 1]
        action_road = compiled.outgoing_roads[state_node][action_parent, action_slot]
        action_cloud = state_cloud[action_parent]
        road_features = compiled.road_features[action_road].to(dtype=dtype)
        raining = (compiled.road_destination[action_road] == action_cloud).to(dtype=dtype)
        features = torch.cat(
            (
                role.expand(len(action_road), -1),
                state_stamina[action_parent, None],
                state_freshness[action_parent, None],
                torch.full((len(action_road), 1), cargo_load, dtype=dtype, device=device),
                road_features,
                raining[:, None],
            ),
            dim=-1,
        )
        event_mask = event_mask_from_terrain(compiled.road_terrain[action_road])
        transition = model(features, event_mask)
        expected_time = transition.expected_travel_time
        expected_freshness = transition.expected_next_freshness
        next_stamina_by_action = transition.next_stamina.squeeze(-1)
        event_rate_sum = event_rate_sum + transition.event_rate.sum()
        event_variance_sum = event_variance_sum + transition.event_delay_variance.sum()
        event_rows += len(action_road)

        cloud_destination = compiled.cloud_destination[action_cloud]
        cloud_probability = compiled.cloud_probability[action_cloud].to(dtype=dtype)
        cloud_mask = compiled.cloud_mask[action_cloud]
        valid_clouds = cloud_mask.nonzero(as_tuple=False)
        child_action = valid_clouds[:, 0]
        cloud_slot = valid_clouds[:, 1]
        child_probability = cloud_probability[child_action, cloud_slot]

        levels.append(
            _TreeLevel(
                state_count=len(state_node),
                action_parent=action_parent,
                action_slot=action_slot,
                action_mask=action_mask,
                child_action=child_action,
                child_probability=child_probability,
            )
        )
        state_node = compiled.road_destination[action_road][child_action]
        state_cloud = cloud_destination[child_action, cloud_slot]
        state_stamina = next_stamina_by_action[child_action]
        state_freshness = expected_freshness[child_action]
        state_elapsed = state_elapsed[action_parent][child_action] + expected_time[child_action]

    # Carry an uncorrupted reference branch even when the caller does not request
    # epsilon=0. All interventions corrupt this SAME policy, rather than planning
    # a new reference policy that anticipates each level of future corruption.
    planning_epsilons = (0.0, *epsilons)
    epsilon_tensor = torch.tensor(planning_epsilons, dtype=dtype, device=device)
    metrics = _terminal_metrics(state_stamina, state_freshness, state_elapsed, hard=hard)
    values = metrics[:, None, :].expand(-1, len(planning_epsilons), -1)
    first_action_score = torch.empty(0, dtype=dtype, device=device)
    first_action_mask = torch.empty(0, dtype=torch.bool, device=device)

    for level_index, level in reversed(tuple(enumerate(levels))):
        action_count = len(level.action_parent)
        action_values = torch.zeros(
            (action_count, len(planning_epsilons), metrics.shape[-1]),
            dtype=dtype,
            device=device,
        )
        action_values.index_add_(
            0,
            level.child_action,
            level.child_probability[:, None, None] * values,
        )
        max_actions = level.action_mask.shape[1]
        padded = torch.zeros(
            (level.state_count, len(planning_epsilons), max_actions, metrics.shape[-1]),
            dtype=dtype,
            device=device,
        )
        padded[level.action_parent, :, level.action_slot, :] = action_values
        scores = padded[..., 0]
        mask = level.action_mask[:, None, :]
        masked_scores = scores.masked_fill(~mask, -torch.inf)
        reference_scores = masked_scores[:, :1, :]
        if hard:
            best = reference_scores.argmax(dim=-1)
            reference = torch.nn.functional.one_hot(best, num_classes=max_actions).to(dtype=dtype)
        else:
            reference = torch.softmax(inverse_temperature * reference_scores, dim=-1)
        uniform = level.action_mask.to(dtype=dtype)
        uniform = uniform / uniform.sum(dim=-1, keepdim=True)
        policy = (1.0 - epsilon_tensor[None, :, None]) * reference + epsilon_tensor[
            None, :, None
        ] * uniform[:, None, :]
        values = (policy[..., None] * padded).sum(dim=2)
        if level_index == 0:
            first_action_score = (root_weight[:, None, None] * scores).sum(dim=0)
            first_action_mask = level.action_mask[0]

    root_values = (root_weight[:, None, None] * values).sum(dim=0)
    denominator = max(event_rows, 1)
    return PlannerResult(
        epsilons=epsilon_tensor[1:],
        final_score=root_values[1:, 0],
        final_freshness=root_values[1:, 1],
        final_stamina=root_values[1:, 2],
        elapsed_time=root_values[1:, 3],
        event_rate=event_rate_sum / denominator,
        event_delay_variance=event_variance_sum / denominator,
        first_action_score=first_action_score[1:],
        first_action_mask=first_action_mask,
    )
