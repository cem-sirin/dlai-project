"""Local, stateful neural transition mechanics for Mackerel Road."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mackerel_road.game.types import PorterKind, Terrain

N_FEATURES = 12
ROAD_DISTANCE_MAX = 5.5
TIME_MIN = 0.20
TIME_MAX = 6.00
STAMINA_LOSS_MAX = 0.16
SPOILAGE_RATE_MAX = 0.060
ROLE_ADVANTAGE_MAX = 0.55
EVENT_DELAY_MAX = 3.00


class EventKind(IntEnum):
    NONE = 0
    LANDSLIDE = 1
    FALLEN_TREE = 2
    CART_BLOCKAGE = 3


N_EVENTS = len(EventKind) - 1

EVENT_FOR_TERRAIN = {
    Terrain.MOUNTAIN: EventKind.LANDSLIDE,
    Terrain.FOREST: EventKind.FALLEN_TREE,
    Terrain.POST_ROAD: EventKind.CART_BLOCKAGE,
}


def _logit(fraction: float) -> float:
    return math.log(fraction / (1.0 - fraction))


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass(frozen=True)
class NeuralTransitionBatch:
    """All categorical event branches for a batch of local transitions."""

    event_probability: Tensor
    travel_time: Tensor
    next_stamina: Tensor
    next_freshness: Tensor
    road_time: Tensor
    stamina_loss: Tensor
    spoilage_rate: Tensor
    event_delay: Tensor

    @property
    def expected_travel_time(self) -> Tensor:
        return (self.event_probability * self.travel_time).sum(dim=-1)

    @property
    def expected_next_freshness(self) -> Tensor:
        return (self.event_probability * self.next_freshness).sum(dim=-1)

    @property
    def event_rate(self) -> Tensor:
        return 1.0 - self.event_probability[:, EventKind.NONE]

    @property
    def event_delay_variance(self) -> Tensor:
        mean = (self.event_probability * self.event_delay).sum(dim=-1)
        second = (self.event_probability * self.event_delay.square()).sum(dim=-1)
        return (second - mean.square()).clamp_min(0.0)


class NeuralMechanics(nn.Module):
    """A map-blind local transition model with structurally limited role effects."""

    def __init__(
        self,
        *,
        base_hidden: int = 64,
        role_hidden: int = 16,
        event_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.base_hidden = base_hidden
        self.role_hidden = role_hidden
        self.event_hidden = event_hidden
        self.base_net = _mlp(10, base_hidden, 3)
        self.role_net = _mlp(1, role_hidden, 2)
        self.event_net = _mlp(7, event_hidden, N_EVENTS + 1)
        self.event_delay_net = _mlp(7, event_hidden, N_EVENTS)
        self._initialize_outputs()

    def _initialize_outputs(self) -> None:
        base_output = self.base_net[-1]
        role_output = self.role_net[-1]
        event_output = self.event_net[-1]
        delay_output = self.event_delay_net[-1]
        assert isinstance(base_output, nn.Linear)
        assert isinstance(role_output, nn.Linear)
        assert isinstance(event_output, nn.Linear)
        assert isinstance(delay_output, nn.Linear)

        for output in (base_output, role_output, event_output, delay_output):
            nn.init.zeros_(output.weight)

        with torch.no_grad():
            base_output.bias.copy_(
                torch.tensor(
                    (
                        _logit((2.4 - TIME_MIN) / (TIME_MAX - TIME_MIN)),
                        _logit(0.065 / STAMINA_LOSS_MAX),
                        _logit(0.018 / SPOILAGE_RATE_MAX),
                    )
                )
            )
            role_output.bias.fill_(_logit(0.20 / ROLE_ADVANTAGE_MAX))
            event_output.bias.zero_()
            event_output.bias[EventKind.NONE] = _logit(0.80)
            delay_output.bias.fill_(_logit(0.80 / EVENT_DELAY_MAX))

    def config(self) -> dict[str, int]:
        return {
            "base_hidden": self.base_hidden,
            "role_hidden": self.role_hidden,
            "event_hidden": self.event_hidden,
        }

    def forward(self, features: Tensor, event_mask: Tensor) -> NeuralTransitionBatch:
        if features.ndim != 2 or features.shape[1] != N_FEATURES:
            raise ValueError(f"features must have shape [N, {N_FEATURES}]")
        expected_mask_shape = (features.shape[0], N_EVENTS + 1)
        if event_mask.shape != expected_mask_shape:
            raise ValueError(f"event_mask must have shape {expected_mask_shape}")
        if event_mask.dtype != torch.bool:
            raise ValueError("event_mask must be boolean")
        if not bool(event_mask[:, EventKind.NONE].all()):
            raise ValueError("NO_EVENT must be allowed for every transition")

        role = features[:, 0:2]
        x_base = features[:, 2:12]
        x_event = features[:, 5:12]
        difficulty = features[:, 6:7]

        raw_base = self.base_net(x_base)
        base_time = TIME_MIN + (TIME_MAX - TIME_MIN) * torch.sigmoid(raw_base[:, 0:1])
        base_stamina = STAMINA_LOSS_MAX * torch.sigmoid(raw_base[:, 1:2])
        spoilage_rate = SPOILAGE_RATE_MAX * torch.sigmoid(raw_base[:, 2:3])

        role_advantage = ROLE_ADVANTAGE_MAX * torch.sigmoid(self.role_net(difficulty))
        time_advantage = role_advantage[:, 0:1]
        strong_stamina_advantage = role_advantage[:, 1:2]
        is_light = role[:, 0:1]
        is_strong = role[:, 1:2]
        speed_gate = is_strong * difficulty + is_light * (1.0 - difficulty)
        travel_time = base_time * torch.exp(-time_advantage * speed_gate)
        stamina_loss = base_stamina * torch.exp(-strong_stamina_advantage * is_strong * difficulty)

        event_logits = self.event_net(x_event).masked_fill(~event_mask, -torch.inf)
        event_probability = event_logits.softmax(dim=-1)
        real_event_delay = EVENT_DELAY_MAX * torch.sigmoid(self.event_delay_net(x_event))
        event_delay = F.pad(real_event_delay, (1, 0))
        event_travel_time = travel_time + event_delay
        next_stamina = (features[:, 2:3] - stamina_loss).clamp(0.0, 1.0)
        next_freshness = features[:, 3:4] * torch.exp(-spoilage_rate * event_travel_time)

        return NeuralTransitionBatch(
            event_probability=event_probability,
            travel_time=event_travel_time,
            next_stamina=next_stamina,
            next_freshness=next_freshness,
            road_time=travel_time.squeeze(-1),
            stamina_loss=stamina_loss.squeeze(-1),
            spoilage_rate=spoilage_rate.squeeze(-1),
            event_delay=event_delay,
        )


def role_features(kind: PorterKind, *, device: torch.device) -> Tensor:
    if kind is PorterKind.LIGHT:
        return torch.tensor((1.0, 0.0), device=device)
    return torch.tensor((0.0, 1.0), device=device)


def event_mask_from_terrain(terrain_ids: Tensor) -> Tensor:
    """Return NO_EVENT plus one terrain-compatible event per transition."""
    if terrain_ids.ndim != 1:
        raise ValueError("terrain_ids must have shape [N]")
    if terrain_ids.numel() and not bool(((terrain_ids >= 0) & (terrain_ids < 3)).all()):
        raise ValueError("terrain_ids must contain values in [0, 2]")
    mask = torch.zeros(
        (terrain_ids.shape[0], N_EVENTS + 1),
        dtype=torch.bool,
        device=terrain_ids.device,
    )
    mask[:, EventKind.NONE] = True
    terrain_to_event = torch.tensor(
        (
            EventKind.LANDSLIDE,
            EventKind.FALLEN_TREE,
            EventKind.CART_BLOCKAGE,
        ),
        dtype=torch.long,
        device=terrain_ids.device,
    )
    event_ids = terrain_to_event[terrain_ids]
    mask.scatter_(1, event_ids.unsqueeze(1), True)
    return mask
