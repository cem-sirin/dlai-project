"""Game-level objectives for the neural transition mechanism."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from mackerel_road.experiments.protocol import (
    PROFILE_EPSILONS,
    TARGET_WIN_PROFILE,
    ObjectiveSpec,
)
from mackerel_road.game.types import PorterKind, RouteMap
from mackerel_road.nn.mechanics import NeuralMechanics, event_mask_from_terrain
from mackerel_road.nn.planning import PlannerResult, plan_initial_state


@dataclass(frozen=True)
class NeuralObjectiveResult:
    total: Tensor
    components: dict[str, Tensor]
    diagnostics: dict[str, Tensor]


def _plan_maps(
    model: NeuralMechanics,
    maps: tuple[RouteMap, ...],
    kind: PorterKind,
    *,
    epsilons: tuple[float, ...],
    clouds: bool,
    checkpoint_plans: bool = False,
) -> tuple[PlannerResult, ...]:
    if checkpoint_plans:
        # Retain each plan's outputs, not all planning-tree activations at once.
        # Non-reentrant checkpointing supports the dataclass and model parameters.
        return tuple(
            checkpoint(
                plan_initial_state,
                model,
                route_map,
                kind,
                epsilons=epsilons,
                clouds=clouds,
                use_reentrant=False,
            )
            for route_map in maps
        )
    return tuple(
        plan_initial_state(model, route_map, kind, epsilons=epsilons, clouds=clouds)
        for route_map in maps
    )


def _stack(results: tuple[PlannerResult, ...], name: str) -> Tensor:
    return torch.stack([getattr(result, name) for result in results])


def _semantic_probe_features(model: NeuralMechanics, samples: int = 96) -> Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    index = torch.arange(samples, device=device, dtype=dtype)
    terrain = (torch.arange(samples, device=device) % 3).long()
    features = torch.zeros((samples, 12), dtype=dtype, device=device)
    features[:, 0] = (torch.arange(samples, device=device) % 2 == 0).to(dtype)
    features[:, 1] = 1.0 - features[:, 0]
    features[:, 2] = 0.35 + 0.60 * ((index * 0.37) % 1.0)
    features[:, 3] = 0.25 + 0.70 * ((index * 0.61) % 1.0)
    features[:, 4] = 0.50 + 0.45 * ((index * 0.23) % 1.0)
    features[:, 5] = 0.20 + 0.70 * ((index * 0.43) % 1.0)
    features[:, 6] = 0.10 + 0.80 * ((index * 0.29) % 1.0)
    features[:, 7] = 0.10 + 0.80 * ((index * 0.53) % 1.0)
    features[torch.arange(samples, device=device), 8 + terrain] = 1.0
    features[:, 11] = (torch.arange(samples, device=device) % 3 == 0).to(dtype)
    return features


def semantic_penalty(model: NeuralMechanics) -> tuple[Tensor, dict[str, Tensor]]:
    features = _semantic_probe_features(model)
    terrain = features[:, 8:11].argmax(dim=-1)
    mask = event_mask_from_terrain(terrain)
    base = model(features, mask)

    longer = features.clone()
    longer[:, 5] = (longer[:, 5] + 0.12).clamp_max(1.0)
    longer_result = model(longer, mask)
    distance = (
        torch.relu(base.expected_travel_time - longer_result.expected_travel_time).mean()
        + torch.relu(base.stamina_loss - longer_result.stamina_loss).mean()
    )

    harder = features.clone()
    harder[:, 6] = (harder[:, 6] + 0.12).clamp_max(1.0)
    harder_result = model(harder, mask)
    difficulty = (
        torch.relu(base.road_time - harder_result.road_time).mean()
        + torch.relu(base.stamina_loss - harder_result.stamina_loss).mean()
    )

    clear = features.clone()
    clear[:, 11] = 0.0
    rainy = features.clone()
    rainy[:, 11] = 1.0
    clear_result = model(clear, mask)
    rainy_result = model(rainy, mask)
    rain = (
        torch.relu(clear_result.expected_travel_time - rainy_result.expected_travel_time).mean()
        + torch.relu(clear_result.spoilage_rate - rainy_result.spoilage_rate).mean()
    )

    hard_light = features.clone()
    hard_light[:, 0:2] = torch.tensor((1.0, 0.0), dtype=features.dtype, device=features.device)
    hard_light[:, 6] = 0.85
    hard_strong = hard_light.clone()
    hard_strong[:, 0:2] = torch.tensor((0.0, 1.0), dtype=features.dtype, device=features.device)
    hard_light_result = model(hard_light, mask)
    hard_strong_result = model(hard_strong, mask)
    strong_role = (
        torch.relu(hard_strong_result.road_time - hard_light_result.road_time).mean()
        + torch.relu(hard_strong_result.stamina_loss - hard_light_result.stamina_loss).mean()
    )

    easy_light = hard_light.clone()
    easy_light[:, 6] = 0.15
    easy_strong = hard_strong.clone()
    easy_strong[:, 6] = 0.15
    light_role = torch.relu(
        model(easy_light, mask).road_time - model(easy_strong, mask).road_time
    ).mean()

    components: dict[str, Tensor] = {
        "semantic_distance": distance,
        "semantic_difficulty": difficulty,
        "semantic_rain": rain,
        "semantic_strong_role": strong_role,
        "semantic_light_role": light_role,
    }
    return torch.stack(tuple(components.values())).sum(), components


def neural_game_objective(
    model: NeuralMechanics,
    maps: tuple[RouteMap, ...],
    spec: ObjectiveSpec,
    *,
    checkpoint_plans: bool = False,
) -> NeuralObjectiveResult:
    if not maps:
        raise ValueError("the objective needs at least one map")
    epsilons = PROFILE_EPSILONS if spec.skill_profile else (0.0,)
    light = _plan_maps(
        model,
        maps,
        PorterKind.LIGHT,
        epsilons=epsilons,
        clouds=True,
        checkpoint_plans=checkpoint_plans,
    )
    strong = _plan_maps(
        model,
        maps,
        PorterKind.STRONG,
        epsilons=epsilons,
        clouds=True,
        checkpoint_plans=checkpoint_plans,
    )
    light_score = _stack(light, "final_score")
    strong_score = _stack(strong, "final_score")
    light_freshness = _stack(light, "final_freshness")
    strong_freshness = _stack(strong, "final_freshness")
    light_stamina = _stack(light, "final_stamina")
    strong_stamina = _stack(strong, "final_stamina")
    exact_score = torch.cat((light_score[:, 0], strong_score[:, 0]))
    exact_freshness_loss = 1.0 - exact_score

    low_loss = torch.relu(0.28 - exact_freshness_loss).square().mean() / 0.01
    high_loss = torch.relu(exact_freshness_loss - 0.78).square().mean() / 0.01
    stamina_loss = torch.cat((1.0 - light_stamina[:, 0], 1.0 - strong_stamina[:, 0]))
    exhaustion = torch.relu(stamina_loss - 0.90).square().mean() / 0.01
    soft_failure = (
        torch.sigmoid((0.025 - torch.cat((light_freshness[:, 0], strong_freshness[:, 0]))) / 0.01)
        + torch.sigmoid((stamina_loss - 0.975) / 0.01)
    ).mean()
    viability = low_loss + high_loss + exhaustion + 0.10 * soft_failure

    role_difference = (1.0 - light_score[:, 0]) - (1.0 - strong_score[:, 0])
    balance = (role_difference.mean() / 0.025).square()
    smooth_scale = 0.01
    strong_advantage = (
        smooth_scale * torch.nn.functional.softplus(role_difference / smooth_scale)
    ).mean()
    light_advantage = (
        smooth_scale * torch.nn.functional.softplus(-role_difference / smooth_scale)
    ).mean()
    specialization = ((strong_advantage - 0.020) / 0.015).square() + (
        (light_advantage - 0.020) / 0.015
    ).square()

    luck_features = _semantic_probe_features(model)
    luck_terrain = luck_features[:, 8:11].argmax(dim=-1)
    luck_transition = model(luck_features, event_mask_from_terrain(luck_terrain))
    event_rate = luck_transition.event_rate.mean()
    event_variance = luck_transition.event_delay_variance.mean()
    luck = (
        torch.relu(0.12 - event_rate).square() / 0.01
        + torch.relu(event_rate - 0.35).square() / 0.01
        + torch.relu(0.04 - event_variance).square() / 0.01
        + torch.relu(event_variance - 0.80).square() / 0.25
    )

    semantic, semantic_components = semantic_penalty(model)
    parameter_regularity = torch.stack(
        [parameter.square().mean() for parameter in model.parameters()]
    ).mean()
    regularity = 0.25 * semantic + 1e-5 * parameter_regularity

    components = {
        "viability": viability,
        "balance": balance,
        "specialization": specialization,
        "luck": luck,
        "regularity": regularity,
    }
    diagnostics = {
        "mean_final_score": exact_score.mean(),
        "mean_final_stamina": torch.cat((light_stamina[:, 0], strong_stamina[:, 0])).mean(),
        "mean_role_difference": role_difference.mean(),
        "mean_absolute_role_difference": role_difference.abs().mean(),
        "strong_favored_fraction": (role_difference > 0.0).to(exact_score.dtype).mean(),
        "light_favored_fraction": (role_difference < 0.0).to(exact_score.dtype).mean(),
        "strong_context_advantage": strong_advantage,
        "light_context_advantage": light_advantage,
        "event_rate": event_rate,
        "event_delay_variance": event_variance,
        **semantic_components,
    }

    if spec.skill_profile:
        profile = []
        for index, epsilon in enumerate(PROFILE_EPSILONS):
            light_focal = torch.sigmoid((light_score[:, index] - strong_score[:, 0]) / 0.025)
            strong_focal = torch.sigmoid((strong_score[:, index] - light_score[:, 0]) / 0.025)
            profile.append(torch.cat((light_focal, strong_focal)).mean())
            diagnostics[f"profile_{epsilon:.2f}"] = profile[-1]
        profile_tensor = torch.stack(profile)
        target = torch.tensor(
            TARGET_WIN_PROFILE,
            dtype=profile_tensor.dtype,
            device=profile_tensor.device,
        )
        components["skill_profile"] = ((profile_tensor - target) / 0.10).square().mean()

    if spec.weather:
        light_clear = _stack(
            _plan_maps(
                model,
                maps,
                PorterKind.LIGHT,
                epsilons=(0.0,),
                clouds=False,
                checkpoint_plans=checkpoint_plans,
            ),
            "final_score",
        )[:, 0]
        strong_clear = _stack(
            _plan_maps(
                model,
                maps,
                PorterKind.STRONG,
                epsilons=(0.0,),
                clouds=False,
                checkpoint_plans=checkpoint_plans,
            ),
            "final_score",
        )[:, 0]
        weather_effect = torch.cat(
            (light_clear - light_score[:, 0], strong_clear - strong_score[:, 0])
        ).mean()
        components["weather"] = ((weather_effect - 0.025) / 0.015).square()
        diagnostics["weather_effect"] = weather_effect

    total = components["viability"] + components["luck"] + components["regularity"]
    if spec.balance:
        total = total + components["balance"]
    if spec.specialization:
        total = total + components["specialization"]
    if spec.skill_profile:
        total = total + components["skill_profile"]
    if spec.weather:
        total = total + components["weather"]
    return NeuralObjectiveResult(total=total, components=components, diagnostics=diagnostics)
