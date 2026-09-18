"""Seeded generation of moderate layered route maps and moving rain clouds."""

from __future__ import annotations

from itertools import pairwise
from random import Random

from mackerel_road.game.types import Road, RouteMap, Terrain

CLEAR_SKY = "clear_sky"
MIN_HORIZON = 7
MAX_HORIZON = 10
MIN_WIDTH = 3
MAX_WIDTH = 5


def _node_layers(widths: tuple[int, ...]) -> tuple[tuple[str, ...], ...]:
    if len(widths) < 3 or widths[0] != 1 or widths[-1] != 1:
        raise ValueError("widths must begin and end with one node")
    if any(width <= 0 for width in widths):
        raise ValueError("layer widths must be positive")
    layers: list[tuple[str, ...]] = [("obama",)]
    for layer_index, width in enumerate(widths[1:-1], start=1):
        layers.append(tuple(f"l{layer_index}n{node_index}" for node_index in range(width)))
    layers.append(("kyoto",))
    return tuple(layers)


def _sample_widths(rng: Random, horizon: int) -> tuple[int, ...]:
    if not MIN_HORIZON <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be in [{MIN_HORIZON}, {MAX_HORIZON}]")
    internal_widths = [MIN_WIDTH]
    internal_widths.extend(rng.randint(MIN_WIDTH, MAX_WIDTH) for _ in range(horizon - 3))
    internal_widths.append(MIN_WIDTH)
    return (1, *internal_widths, 1)


def _structured_assignments(
    sources: tuple[str, ...],
    destinations: tuple[str, ...],
    out_degree: int,
    rng: Random,
) -> dict[str, set[str]]:
    """Cover the next layer, then add bounded links with a low-density bias."""
    assignments: dict[str, set[str]] = {source: set() for source in sources}
    if len(destinations) == 1:
        for source in sources:
            assignments[source].add(destinations[0])
        return assignments

    shuffled_sources = list(sources)
    shuffled_destinations = list(destinations)
    rng.shuffle(shuffled_sources)
    rng.shuffle(shuffled_destinations)
    for index, destination in enumerate(shuffled_destinations):
        assignments[shuffled_sources[index % len(shuffled_sources)]].add(destination)

    target_degree = min(out_degree, len(destinations))
    for source in shuffled_sources:
        while len(assignments[source]) < target_degree:
            incoming_counts = {
                destination: sum(destination in targets for targets in assignments.values())
                for destination in destinations
                if destination not in assignments[source]
            }
            least_incoming = min(incoming_counts.values())
            candidates = [
                destination
                for destination, count in incoming_counts.items()
                if count == least_incoming
            ]
            assignments[source].add(rng.choice(candidates))
    return assignments


def _road_features(terrain: Terrain, rng: Random) -> tuple[float, float, float]:
    if terrain is Terrain.MOUNTAIN:
        return rng.uniform(1.8, 3.8), rng.uniform(0.65, 0.95), rng.uniform(0.55, 0.95)
    if terrain is Terrain.FOREST:
        return rng.uniform(2.4, 4.5), rng.uniform(0.30, 0.70), rng.uniform(0.15, 0.55)
    return rng.uniform(3.2, 5.5), rng.uniform(0.05, 0.35), rng.uniform(0.25, 0.65)


def generate_route_map(
    seed: int,
    *,
    horizon: int | None = None,
    widths: tuple[int, ...] | None = None,
    out_degree: int = 2,
) -> RouteMap:
    """Generate a sparse structured DAG whose source-target distance is 7--10."""
    if out_degree <= 0:
        raise ValueError("out_degree must be positive")
    if horizon is not None and widths is not None:
        raise ValueError("provide horizon or widths, not both")
    rng = Random(seed)
    if widths is None:
        sampled_horizon = rng.randint(MIN_HORIZON, MAX_HORIZON) if horizon is None else horizon
        widths = _sample_widths(rng, sampled_horizon)
    layers = _node_layers(widths)
    roads: list[Road] = []

    for layer_index, (sources, destinations) in enumerate(pairwise(layers)):
        assignments = _structured_assignments(sources, destinations, out_degree, rng)
        edge_pairs = [
            (source, destination)
            for source in sources
            for destination in sorted(assignments[source])
        ]
        terrain_bag = [tuple(Terrain)[index % len(Terrain)] for index in range(len(edge_pairs))]
        rng.shuffle(terrain_bag)

        for (source, destination), terrain in zip(edge_pairs, terrain_bag, strict=True):
            distance, difficulty, exposure = _road_features(terrain, rng)
            roads.append(
                Road(
                    road_id=f"r{layer_index}_{source}_{destination}",
                    source=source,
                    destination=destination,
                    terrain=terrain,
                    distance=round(distance, 3),
                    difficulty=round(difficulty, 3),
                    exposure=round(exposure, 3),
                )
            )

    return RouteMap(layers=layers, roads=tuple(roads))


def shortest_path_length(route_map: RouteMap) -> int:
    """Return the unweighted source-target distance without assuming layer depth."""
    frontier = {route_map.start}
    distance = 0
    while frontier:
        if route_map.destination in frontier:
            return distance
        frontier = {road.destination for node in frontier for road in route_map.outgoing(node)}
        distance += 1
    raise ValueError("destination is unreachable")


def route_count(route_map: RouteMap) -> int:
    """Count complete source-target routes in the DAG."""
    counts = {route_map.destination: 1}
    for layer in reversed(route_map.layers[:-1]):
        for node in layer:
            counts[node] = sum(counts[road.destination] for road in route_map.outgoing(node))
    return counts[route_map.start]


def initial_cloud_node(route_map: RouteMap, rng: Random) -> str:
    return rng.choice(route_map.layers[1])


def move_cloud(route_map: RouteMap, cloud_node: str, rng: Random) -> str:
    if cloud_node == CLEAR_SKY:
        return CLEAR_SKY
    roads = route_map.outgoing(cloud_node)
    if not roads:
        return CLEAR_SKY
    destination = rng.choice(roads).destination
    return CLEAR_SKY if destination == route_map.destination else destination
