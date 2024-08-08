from beartype.typing import Tuple

import pytest
import torch
import torch.nn as nn
from nlrl import NeuralLogicRuleLayer


@pytest.fixture
def input_size() -> int:
    return 10


@pytest.fixture
def layer(input_size: int):
    output_size = 5
    layer = NeuralLogicRuleLayer(input_size, output_size)
    return layer


def test_initialization(input_size: int):
    output_size = 5
    layer = NeuralLogicRuleLayer(input_size, output_size)

    assert layer.input_size == input_size
    assert layer.output_size == output_size
    assert isinstance(layer.GN, nn.Parameter)
    assert layer.GN.requires_grad
    assert isinstance(layer.GS, nn.Parameter)
    assert layer.GS.requires_grad
    assert isinstance(layer.GR, nn.Parameter)
    assert layer.GR.requires_grad

    assert layer.GN.shape == (input_size,)
    assert layer.GS.shape == (output_size,)
    assert layer.GR.shape == (input_size, output_size)


# parameterise by batch dimensions
@pytest.mark.parametrize(
    "x", [torch.rand(10), torch.rand(3, 10), torch.rand(5, 10), torch.rand(3, 2, 10)]
)
def test_negation(layer: NeuralLogicRuleLayer, x: torch.Tensor):
    result = layer.negation(x)

    assert result.shape == x.shape
    assert torch.all(result >= 0) and torch.all(
        result <= 1
    ), "Negation output should be in range [0, 1]"


@pytest.mark.parametrize(
    "x", [torch.rand(10), torch.rand(3, 10), torch.rand(5, 10), torch.rand(3, 2, 10)]
)
def test_conjunction(layer: NeuralLogicRuleLayer, x: torch.Tensor):
    result = layer.conjunction(x)

    assert result.shape == (*x.shape[:-1], layer.output_size)
    assert torch.all(result >= 0) and torch.all(
        result <= 1
    ), "Conjunction output should be in range [0, 1]"


@pytest.mark.parametrize(
    "x", [torch.rand(10), torch.rand(3, 10), torch.rand(5, 10), torch.rand(3, 2, 10)]
)
def test_disjunction(layer: NeuralLogicRuleLayer, x: torch.Tensor):
    result = layer.disjunction(x)

    assert result.shape == (*x.shape[:-1], layer.output_size)
    assert torch.all(result >= 0) and torch.all(
        result <= 1
    ), "Disjunction output should be in range [0, 1]"


@pytest.mark.parametrize(
    "x_1, x_2",
    [
        (torch.rand(5), torch.rand(5)),
        (torch.rand(3, 5), torch.rand(3, 5)),
        (torch.rand(5, 5), torch.rand(5, 5)),
        (torch.rand(3, 2, 5), torch.rand(3, 2, 5)),
    ],
)
def test_selection(layer: NeuralLogicRuleLayer, x_1: torch.Tensor, x_2: torch.Tensor):
    result = layer.selection(x_1, x_2)

    assert result.shape == x_1.shape
    assert torch.all(result >= 0) and torch.all(
        result <= 1
    ), "Selection output should be in range [0, 1]"


@pytest.mark.parametrize(
    "x", [torch.rand(10), torch.rand(3, 10), torch.rand(5, 10), torch.rand(3, 2, 10)]
)
def test_forward(layer: NeuralLogicRuleLayer, x: torch.Tensor):
    result = layer(x)

    assert result.shape == (*x.shape[:-1], layer.output_size)
    assert torch.all(result >= 0) and torch.all(
        result <= 1
    ), "Forward output should be in range [0, 1]"


if __name__ == "__main__":
    pytest.main()
