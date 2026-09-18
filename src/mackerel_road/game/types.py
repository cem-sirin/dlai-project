"""Domain objects for the two-porter Mackerel Road race."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Terrain(StrEnum):
    MOUNTAIN = "mountain"
    FOREST = "forest"
    POST_ROAD = "post_road"


class PorterKind(StrEnum):
    LIGHT = "light"
    STRONG = "strong"


@dataclass(frozen=True)
class Road:
    road_id: str
    source: str
    destination: str
    terrain: Terrain
    distance: float
    difficulty: float
    exposure: float

    def __post_init__(self) -> None:
        if self.source == self.destination:
            raise ValueError("a road must connect two different nodes")
        if self.distance <= 0.0:
            raise ValueError("road distance must be positive")
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError("road difficulty must be in [0, 1]")
        if not 0.0 <= self.exposure <= 1.0:
            raise ValueError("road exposure must be in [0, 1]")


@dataclass(frozen=True)
class RouteMap:
    layers: tuple[tuple[str, ...], ...]
    roads: tuple[Road, ...]

    def __post_init__(self) -> None:
        if len(self.layers) < 3:
            raise ValueError("a route map needs start, intermediate, and destination layers")
        if len(self.layers[0]) != 1 or len(self.layers[-1]) != 1:
            raise ValueError("the first and last layers must contain one node")
        nodes = tuple(node for layer in self.layers for node in layer)
        if len(nodes) != len(set(nodes)):
            raise ValueError("node identifiers must be unique")
        layer_of = {node: index for index, layer in enumerate(self.layers) for node in layer}
        for road in self.roads:
            if road.source not in layer_of or road.destination not in layer_of:
                raise ValueError(f"road {road.road_id!r} refers to an unknown node")
            if layer_of[road.destination] != layer_of[road.source] + 1:
                raise ValueError("roads must advance exactly one layer")
        for layer in self.layers[:-1]:
            for node in layer:
                if not self.outgoing(node):
                    raise ValueError(f"nonterminal node {node!r} has no outgoing road")
        for layer in self.layers[1:]:
            for node in layer:
                if not self.incoming(node):
                    raise ValueError(f"node {node!r} is unreachable")

    @property
    def start(self) -> str:
        return self.layers[0][0]

    @property
    def destination(self) -> str:
        return self.layers[-1][0]

    def outgoing(self, node: str) -> tuple[Road, ...]:
        return tuple(road for road in self.roads if road.source == node)

    def incoming(self, node: str) -> tuple[Road, ...]:
        return tuple(road for road in self.roads if road.destination == node)

    def layer_index(self, node: str) -> int:
        for index, layer in enumerate(self.layers):
            if node in layer:
                return index
        raise ValueError(f"unknown node: {node}")


@dataclass(frozen=True)
class PorterState:
    kind: PorterKind
    node: str
    elapsed_time: float
    stamina: float
    freshness: float
    finished: bool = False


@dataclass(frozen=True)
class RaceState:
    round_index: int
    cloud_node: str
    light: PorterState
    strong: PorterState


@dataclass(frozen=True)
class MechanicParameters:
    """Positive and bounded parameters preserve declared mechanic semantics."""

    base_time_per_distance: float = 0.70
    difficulty_time_scale: float = 0.60
    load_time_scale: float = 0.18
    rain_time_scale: float = 0.45
    base_stamina_per_distance: float = 0.12
    difficulty_stamina_scale: float = 0.50
    rain_stamina_scale: float = 0.12
    freshness_decay_per_time: float = 0.025
    rain_freshness_scale: float = 0.012
    strong_hard_advantage: float = 0.35
    light_easy_speed_advantage: float = 0.12
    congestion_time: float = 0.25
    light_initial_stamina: float = 15.0
    strong_initial_stamina: float = 17.0

    def __post_init__(self) -> None:
        positive = (
            self.base_time_per_distance,
            self.difficulty_time_scale,
            self.load_time_scale,
            self.rain_time_scale,
            self.base_stamina_per_distance,
            self.difficulty_stamina_scale,
            self.rain_stamina_scale,
            self.freshness_decay_per_time,
            self.rain_freshness_scale,
            self.light_initial_stamina,
            self.strong_initial_stamina,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("positive mechanic parameters must be greater than zero")
        for name, value in (
            ("strong_hard_advantage", self.strong_hard_advantage),
            ("light_easy_speed_advantage", self.light_easy_speed_advantage),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.congestion_time < 0.0:
            raise ValueError("congestion_time must be nonnegative")


@dataclass(frozen=True)
class RoadOutcome:
    travel_time: float
    stamina_loss: float
    freshness_loss: float
    raining: bool
    congested: bool


@dataclass(frozen=True)
class RoundRecord:
    before: RaceState
    light_road: Road | None
    strong_road: Road | None
    light_outcome: RoadOutcome | None
    strong_outcome: RoadOutcome | None
    after: RaceState
