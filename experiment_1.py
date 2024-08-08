from typing import Literal
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Float

from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import Module

import numpy as np
from numpy import random

from models import FullyConnectedNetwork


@beartype
def generate_samples(rng: random.Generator, n: int) -> Float[Tensor, "n 2"]:
    """Generate n pairs of samples."""
    r = rng.random((n, 2))
    return torch.tensor(r, dtype=torch.float32)


@beartype
def augment_samples(
    rng: random.Generator, samples: Float[Tensor, "n 2"], amplitude: float
) -> Float[Tensor, "n 2"]:
    """Add Gaussian noise to the samples."""
    noise = rng.normal(0, amplitude, samples.shape)
    return samples + noise


@beartype
def l_and(x: Float[Tensor, "..."], y: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Algebraic conjunction (x ∧ y)."""
    return x * y


@beartype
def l_or(x: Float[Tensor, "..."], y: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Algebraic disjunction (x ∨ y)."""
    return x + y - x * y


@beartype
def l_not(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Algebraic negation (≠g x)."""
    return 1 - x


@beartype
def l_mi(x: Float[Tensor, "..."], y: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Algebraic material implication (x → y)."""
    return l_or(l_not(x), y)


@beartype
def l_xor(x: Float[Tensor, "..."], y: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Algebraic exclusive or (x ⊕ y)."""
    return l_and(l_or(x, y), l_not(l_and(x, y)))


@beartype
def l_eq(x: Float[Tensor, "..."], y: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Algebraic equality (x ≣ y)."""
    return l_not(l_xor(x, y))


all_fns = [
    ("x ∧ y", l_and),
    ("x ∨ y", l_or),
    ("≠g x", lambda x, _: l_not(x)),
    ("≠g y", lambda _, y: l_not(y)),
    ("x ⊕ y", l_xor),
    ("(x + y)/2", lambda x, y: (x + y) / 2),
    ("x·≠g y", lambda x, y: x * l_not(y)),
    ("x", lambda x, _: x),
    ("y", lambda _, y: y),
    ("0.7", lambda x, _: torch.full(x.shape, 0.7)),
]


@beartype
def compute_fns(samples: Float[Tensor, "n 2"]) -> Float[Tensor, "n 10"]:
    """Compute the results of all functions on the pair of samples."""
    results = []
    for _, fn in all_fns:
        results.append(fn(samples[:, 0], samples[:, 1]))
    return torch.stack(results, dim=1)


@beartype
def make_dataset(samples: Tuple[int, int], noise: float, seed: int = 42) -> Tuple[
    Tuple[Float[Tensor, "n 2"], Float[Tensor, "n 10"]],
    Tuple[Float[Tensor, "m 2"], Float[Tensor, "m 10"]],
]:
    """Generate a dataset of samples and their corresponding targets."""
    rng = random.default_rng(seed)

    x_pre_train = generate_samples(rng, samples[0]).float()
    x_train = augment_samples(rng, x_pre_train, noise).float()
    x_test = generate_samples(rng, samples[1]).float()

    y_train = compute_fns(x_pre_train)
    y_test = compute_fns(x_test)

    return (x_train, y_train), (x_test, y_test)


@beartype
def accuracy(
    y_1: Float[Tensor, "... 1"], y_2: Float[Tensor, "... 1"], delta: float = 0.005
) -> Float[Tensor, ""]:
    return torch.mean((torch.abs(y_1 - y_2) < delta).float())


def evaluate(
    model: Module,
    x: Float[Tensor, "... in"],
    y: Float[Tensor, "... 1"],
    criterion: Module,
) -> Tuple[Float[Tensor, ""], Float[Tensor, ""]]:
    model.eval()
    with torch.no_grad():
        test_output = model(x)
        test_loss = criterion(test_output, y)
        test_accuracy = accuracy(y, test_output)
        return test_loss, test_accuracy


@beartype
def init_model(
    arch: Literal["AN", "AO", "FC"], layers: int, hidden_dim: int, dropout: float = 0.2
) -> Module:
    if arch == "FC":
        return FullyConnectedNetwork(2, 1, layers, hidden_dim, dropout)
    return NeuralLogicNetwork(2, 1, layers, hidden_dim, arch == "AN")


@beartype
def train_model(
    model: Module,
    data: Tuple[
        Tuple[Float[Tensor, "n 2"], Float[Tensor, "n 1"]],
        Tuple[Float[Tensor, "m 2"], Float[Tensor, "m 1"]],
    ],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_div: int,
    lr_step: int,
    test_every: int,
    desc: str = "Training",
):
    (x_train, y_train), (x_test, y_test) = data

    x_train, y_train = x_train, y_train
    x_test, y_test = x_test, y_test

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()  # Assuming the use of MSE loss

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    pb = tqdm(
        range(num_epochs * x_train.shape[0]),
        desc=desc,
        unit="batch",
        unit_scale=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    for epoch in range(num_epochs):
        pb.set_description(f"{desc} [{epoch+1}/{num_epochs}]")
        model.train()

        epoch_loss, epoch_acc = 0, 0
        for batch in range(0, len(x_train), batch_size):
            # Batch the data
            x_batch = x_train[batch : batch + batch_size]
            y_batch = y_train[batch : batch + batch_size]

            # Compute forwards and backwards pass
            optimiser.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimiser.step()

            # Update loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += accuracy(y_batch, output).item()

            if (batch // batch_size) % test_every == 0:
                test_loss, test_acc = evaluate(
                    model, x_test, y_test[:, fn].unsqueeze(-1), criterion
                )
                test_losses.append(test_loss.item())
                test_accuracies.append(test_acc.item())
                pb.set_postfix(
                    {
                        "train_loss": epoch_loss / (batch + 1),
                        "test_loss": test_loss.item(),
                        "train_acc": epoch_acc / (batch + 1) * 100,
                        "test_acc": test_acc.item() * 100,
                    }
                )

            pb.update(batch_size)

        if (epoch + 1) % lr_step == 0:
            for param_group in optimiser.param_groups:
                param_group["lr"] /= lr_div

        train_losses.append(epoch_loss / (len(x_train) / batch_size))
        train_accuracies.append(epoch_acc / (len(x_train) / batch_size))
        pb.set_postfix(
            {
                "train_loss": train_losses[-1],
                "test_loss": test_losses[-1],
                "train_acc": train_accuracies[-1],
                "test_acc": test_accuracies[-1],
            }
        )

    return train_losses, test_losses, train_accuracies, test_accuracies


if __name__ == "__main__":
    import itertools
    from models import FullyConnectedNetwork, NeuralLogicNetwork

    F = len(all_fns)  # number of functions
    FN_NAMES = [fn[0] for fn in all_fns]  # function names

    A = ["AN", "AO", "FC"]  # architectures
    L = {"AN": [2, 3], "AO": [2, 3], "FC": [2, 3, 4]}  # number of layers
    H = {"AN": [5, 25, 50], "AO": [5, 25, 50], "FC": [5, 25, 50, 100]}  # hidden units
    D = 0.2  # dropout

    N = 10  # number of runs
    E = 100  # number of epochs
    B = 20  # batch size

    LR = {"AN": 0.1, "AO": 0.1, "FC": 0.001}  # learning rate
    LR_DIV = 10  # divide learning rate by this
    LR_STEP = 20  # this number of epochs

    TEST_EVERY = 500  # test every this number of batches

    d_train, d_test = make_dataset((10_000, 1_000), 0.0025)

    for arch in A:
        for l, h, fn, i in itertools.product(L[arch], H[arch], range(F), range(N)):
            fn_name = FN_NAMES[fn]
            desc = f"{arch}-{l}-{h} ({fn_name}, {i+1}/{N})"

            train_data = d_train[0], d_train[1][:, fn].unsqueeze(-1)
            test_data = d_test[0], d_test[1][:, fn].unsqueeze(-1)

            model = init_model(arch, l, h, D)
            data = train_data, test_data
            train_losses, test_losses, train_accuracies, test_accuracies = train_model(
                model, data, E, B, LR[arch], LR_DIV, LR_STEP, TEST_EVERY, desc
            )
