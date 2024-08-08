from typing import Literal
from jaxtyping import Float
from beartype import beartype

import torch
import torch.nn as nn
from torch import Tensor


@beartype
class NeuralLogicRuleLayer(nn.Module):
    """
    Neural logic rule layer modelling a Boolean algebra.

    Operations for conjunction, disjunction, negation, and selection are defined
    algebraically in terms of inputs in the unit interval [0, 1].

    There are two to three learnable parameters for each operation:
    - GN: negation parameter
    - GR: rule parameter
    - GS: selection parameter (only used if not nnf)

    Parameters
    ----------
    input_size : int
        Number of input features.
    output_size : int
        Number of output features.
    nnf : bool, optional
        If True, the layer uses negation normal form (NNF) to eliminate the need
        for selection. Default is False.

    References
    ----------
    - Reimann, J., Schwung, A., Ding, S. X. (2022). Neural logic rule layers.
    Information Sciences v596 pp.185-201. https://doi.org/10.1016/j.ins.2022.03.021
    """

    def __init__(self, input_size: int, output_size: int, nnf: bool = False):
        super(NeuralLogicRuleLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nnf = nnf

        self.GN = nn.Parameter(torch.empty(input_size))
        nn.init.uniform_(self.GN, -0.5, 0.5)

        self.GR = nn.Parameter(torch.empty(input_size, output_size))
        nn.init.uniform_(self.GR, -0.5, 0.5)

        if not nnf:
            self.GS = nn.Parameter(torch.empty(output_size))
            nn.init.uniform_(self.GS, -0.5, 0.5)

    def negation(self, x: Float[Tensor, "... in"]) -> Float[Tensor, "... in"]:
        GN = torch.sigmoid(self.GN)
        return (1 - GN) * x + GN * (1 - x)

    def conjunction(self, x: Float[Tensor, "... in"]) -> Float[Tensor, "... out"]:
        GR = torch.sigmoid(self.GR)
        x = torch.log(x.clamp(min=1e-10))
        return torch.exp(torch.matmul(x, GR))

    def disjunction(self, x: Float[Tensor, "... in"]) -> Float[Tensor, "... out"]:
        GR = torch.sigmoid(self.GR)
        x = torch.log(x.clamp(min=1e-10))
        return 1 - torch.exp(torch.matmul(x, GR))

    def selection(
        self, x_1: Float[Tensor, "... out"], x_2: Float[Tensor, "... out"]
    ) -> Float[Tensor, "... out"]:
        assert not self.nnf, "Selection is not defined for NNF"
        GS = torch.sigmoid(self.GS)
        return (1 - GS) * x_1 + GS * x_2

    def forward(self, x: Float[Tensor, "... in"]) -> Float[Tensor, "... out"]:
        x_neg = self.negation(x)
        x_and = self.conjunction(x_neg)
        if self.nnf:
            return x_and

        x_or = self.disjunction(x_neg)
        return self.selection(x_and, x_or)

    def __call__(self, x: Float[Tensor, "... in"]) -> Float[Tensor, "... out"]:
        # for type hints
        return self.forward(x)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            + f"(input_size={self.input_size}, output_size={self.output_size}, nnf={self.nnf})"
        )


if __name__ == "__main__":
    input_size = 10
    output_size = 5
    x = torch.rand(3, input_size)
    layer = NeuralLogicRuleLayer(input_size, output_size, nnf=False)
    print(layer(x))
