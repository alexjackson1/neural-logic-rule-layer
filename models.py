from beartype import beartype
from jaxtyping import Float

import torch
from torch import nn
from torch import Tensor

from nlrl import NeuralLogicRuleLayer


@beartype
class NeuralLogicNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        out_size: int,
        layers: int,
        hidden_dim: int,
        nnf: bool = False,
    ):
        super(NeuralLogicNetwork, self).__init__()
        self.num_layers = layers
        self.hidden_dim = hidden_dim
        self.nnf = nnf

        if layers == 1:
            self.nlrl1 = NeuralLogicRuleLayer(input_size, out_size, nnf=nnf)
            self.proj = nn.Identity()
        else:
            self.nlrl1 = NeuralLogicRuleLayer(input_size, hidden_dim, nnf=nnf)
            self.proj = NeuralLogicRuleLayer(hidden_dim, out_size, nnf=nnf)

        for i in range(2, layers):
            setattr(
                self, f"nlrl{i}", NeuralLogicRuleLayer(hidden_dim, hidden_dim, nnf=nnf)
            )

    def forward(self, x: Float[Tensor, "... in"]) -> Float[Tensor, "... out"]:
        x = self.nlrl1(x)
        for i in range(2, self.num_layers):
            x = getattr(self, f"nlrl{i}")(x)
        x = self.proj(x)
        return x


class FullyConnectedNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        out_size: int,
        layers: int,
        hidden_dim: int,
        dropout: float = 0.2,
    ):
        super(FullyConnectedNetwork, self).__init__()
        self.num_layers = layers
        self.hidden_dim = hidden_dim

        if layers == 1:
            self.fc1 = nn.Linear(input_size, out_size)
            self.proj = nn.Identity()
        else:
            self.fc1 = nn.Linear(input_size, hidden_dim)
            self.proj = nn.Linear(hidden_dim, out_size)

        for i in range(2, layers):
            setattr(self, f"fc{i}", nn.Linear(hidden_dim, hidden_dim))

        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(layers - 1)])

    def forward(self, x: Float[Tensor, "... in"]) -> Float[Tensor, "... out"]:
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropouts[0](x)
        for i in range(2, self.num_layers):
            x = getattr(self, f"fc{i}")(x)
            x = torch.relu(x)
            x = self.dropouts[i - 1](x)
        x = self.proj(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    input_size = 2
    output_size = 9
    layers = 1
    hidden_dim = 5

    model = NeuralLogicNetwork(input_size, output_size, layers, hidden_dim)
    x = torch.rand(1, input_size)
    y = model(x)

    print(model.nlrl1)
    print(model.proj)
    print(y)
