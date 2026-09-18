from __future__ import annotations

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from mackerel_road.experiments.protocol import ABLATIONS  # noqa: E402
from mackerel_road.game.maps import generate_route_map  # noqa: E402
from mackerel_road.game.types import PorterKind  # noqa: E402
from mackerel_road.nn.mechanics import NeuralMechanics  # noqa: E402
from mackerel_road.nn.objective import neural_game_objective  # noqa: E402
from mackerel_road.nn.planning import plan_initial_state  # noqa: E402


def _tiny_map(seed: int = 7):
    return generate_route_map(seed, widths=(1, 2, 2, 1))


def test_stateful_planner_returns_profile_and_gradients() -> None:
    model = NeuralMechanics()
    result = plan_initial_state(
        model,
        _tiny_map(),
        PorterKind.LIGHT,
        epsilons=(0.0, 0.5, 1.0),
    )

    assert result.final_score.shape == (3,)
    assert result.first_action_score.shape[0] == 3
    assert result.first_action_mask.any()
    assert torch.isfinite(result.final_score).all()
    assert result.final_score.min() >= 0.0
    assert result.final_score.max() <= 1.0

    result.final_score.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_neural_game_objective_backpropagates_without_transition_targets() -> None:
    model = NeuralMechanics()
    result = neural_game_objective(model, (_tiny_map(),), ABLATIONS[-1])

    assert torch.isfinite(result.total)
    assert "skill_profile" in result.components
    assert "weather" in result.components
    assert "event_rate" in result.diagnostics

    result.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_corruption_uses_one_reference_policy_for_all_epsilons(monkeypatch) -> None:
    original_softmax = torch.softmax
    reference_widths: list[int] = []

    def record_softmax(values, *args, **kwargs):
        reference_widths.append(values.shape[1])
        return original_softmax(values, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", record_softmax)
    model = NeuralMechanics()
    joint = plan_initial_state(model, _tiny_map(), PorterKind.LIGHT, epsilons=(0.0, 0.25, 0.5, 1.0))
    single = plan_initial_state(model, _tiny_map(), PorterKind.LIGHT, epsilons=(0.5,))
    assert reference_widths and set(reference_widths) == {1}
    assert joint.epsilons.tolist() == [0.0, 0.25, 0.5, 1.0]
    assert single.epsilons.tolist() == [0.5]
    assert torch.allclose(joint.final_score[2], single.final_score[0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_activation_checkpointing_preserves_pooled_loss_and_gradients(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not installed on this test host")
    direct = NeuralMechanics().to(device)
    with torch.no_grad():
        for head in (direct.base_net, direct.role_net, direct.event_net, direct.event_delay_net):
            output = head[-1]
            assert isinstance(output, torch.nn.Linear)
            output.weight.fill_(0.05)
    checkpointed = deepcopy(direct)
    maps = (_tiny_map(7), _tiny_map(8))
    expected = neural_game_objective(direct, maps, ABLATIONS[-1])
    actual = neural_game_objective(checkpointed, maps, ABLATIONS[-1], checkpoint_plans=True)
    torch.testing.assert_close(actual.total, expected.total)
    expected.total.backward()
    actual.total.backward()
    for left, right in zip(direct.parameters(), checkpointed.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad)
