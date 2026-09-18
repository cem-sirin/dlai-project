from __future__ import annotations

from random import Random

from mackerel_road.game.maps import (
    CLEAR_SKY,
    MAX_HORIZON,
    MAX_WIDTH,
    MIN_HORIZON,
    MIN_WIDTH,
    generate_route_map,
    initial_cloud_node,
    move_cloud,
    route_count,
    shortest_path_length,
)
from mackerel_road.game.types import Terrain


def test_generated_map_is_replayable_and_mixed() -> None:
    first = generate_route_map(7)
    second = generate_route_map(7)

    assert first == second
    assert first.start == "obama"
    assert first.destination == "kyoto"
    assert MIN_HORIZON <= shortest_path_length(first) <= MAX_HORIZON
    assert len(first.layers) == shortest_path_length(first) + 1
    assert all(MIN_WIDTH <= len(layer) <= MAX_WIDTH for layer in first.layers[1:-1])
    assert route_count(first) >= 32
    assert {road.terrain for road in first.roads} == set(Terrain)
    assert all(first.outgoing(node) for layer in first.layers[:-1] for node in layer)
    assert all(first.incoming(node) for layer in first.layers[1:] for node in layer)
    assert all(any(len(first.incoming(node)) > 1 for node in layer) for layer in first.layers[2:-1])
    for layer in first.layers[:-1]:
        terrains = [road.terrain for node in layer for road in first.outgoing(node)]
        counts = [terrains.count(terrain) for terrain in Terrain]
        assert max(counts) - min(counts) <= 1


def test_cloud_advances_one_layer_and_clears_before_destination() -> None:
    route_map = generate_route_map(7)
    rng = Random(11)
    cloud = initial_cloud_node(route_map, rng)

    assert cloud in route_map.layers[1]
    for expected_layer in range(2, len(route_map.layers) - 1):
        cloud = move_cloud(route_map, cloud, rng)
        assert cloud in route_map.layers[expected_layer]
    cloud = move_cloud(route_map, cloud, rng)
    assert cloud == CLEAR_SKY
    assert move_cloud(route_map, cloud, rng) == CLEAR_SKY


def test_explicit_small_fixture_remains_available() -> None:
    route_map = generate_route_map(7, widths=(1, 2, 2, 1))

    assert shortest_path_length(route_map) == 3
