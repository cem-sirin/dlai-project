from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mackerel_road.nn.mechanics import (  # noqa: E402
    N_EVENTS,
    EventKind,
    NeuralMechanics,
    event_mask_from_terrain,
)


def _features(*, light: bool, difficulty: float) -> torch.Tensor:
    row = torch.zeros(12)
    row[0 if light else 1] = 1.0
    row[2:5] = torch.tensor((1.0, 1.0, 0.75))
    row[5:8] = torch.tensor((0.6, difficulty, 0.7))
    row[8] = 1.0
    row[11] = 1.0
    return row


def test_neural_mechanics_shapes_masks_and_role_semantics() -> None:
    model = NeuralMechanics()
    hard = torch.stack(
        (_features(light=True, difficulty=0.85), _features(light=False, difficulty=0.85))
    )
    terrain = hard[:, 8:11].argmax(dim=-1)
    mask = event_mask_from_terrain(terrain)
    result = model(hard, mask)

    assert result.event_probability.shape == (2, N_EVENTS + 1)
    assert result.travel_time.shape == (2, N_EVENTS + 1)
    assert result.next_stamina.shape == (2, 1)
    assert result.next_freshness.shape == (2, N_EVENTS + 1)
    assert torch.allclose(result.event_probability.sum(dim=-1), torch.ones(2))
    assert result.event_probability[:, EventKind.FALLEN_TREE].eq(0.0).all()
    assert result.event_probability[:, EventKind.CART_BLOCKAGE].eq(0.0).all()
    assert torch.allclose(result.event_probability[0], result.event_probability[1])
    assert result.road_time[1] < result.road_time[0]
    assert result.stamina_loss[1] < result.stamina_loss[0]
    assert (
        result.next_freshness[:, EventKind.LANDSLIDE]
        .lt(result.next_freshness[:, EventKind.NONE])
        .all()
    )

    easy = torch.stack(
        (_features(light=True, difficulty=0.15), _features(light=False, difficulty=0.15))
    )
    easy_result = model(easy, event_mask_from_terrain(easy[:, 8:11].argmax(dim=-1)))
    assert easy_result.road_time[0] < easy_result.road_time[1]


def test_neural_event_heads_are_role_blind() -> None:
    model = NeuralMechanics()
    features = torch.stack(
        (_features(light=True, difficulty=0.6), _features(light=False, difficulty=0.6))
    )
    mask = event_mask_from_terrain(features[:, 8:11].argmax(dim=-1))
    result = model(features, mask)

    assert torch.allclose(result.event_probability[0], result.event_probability[1])
    assert torch.allclose(result.event_delay[0], result.event_delay[1])
