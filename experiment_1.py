from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Float

import os
import argparse

from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import Module

from models import FullyConnectedNetwork, NeuralLogicNetwork
from config import ALL_FNS


@beartype
def generate_samples(rng: torch.Generator, n: int) -> Float[Tensor, "n 2"]:
    """Generate n pairs of samples."""
    return torch.rand(n, 2, generator=rng)


@beartype
def augment_samples(
    rng: torch.Generator, samples: Float[Tensor, "n 2"], amplitude: float
) -> Float[Tensor, "n 2"]:
    """Add Gaussian noise to the samples."""
    noise = amplitude * torch.randn(samples.shape, generator=rng)
    return samples + noise


@beartype
def compute_fns(samples: Float[Tensor, "n 2"]) -> Float[Tensor, "n 10"]:
    """Compute the results of all functions on the pair of samples."""
    results = []
    for _, fn in ALL_FNS:
        results.append(fn(samples[:, 0], samples[:, 1]))
    return torch.stack(results, dim=1)


@beartype
def make_dataset(samples: Tuple[int, int], noise: float, seed: int = 42) -> Tuple[
    Tuple[Float[Tensor, "n 2"], Float[Tensor, "n 10"]],
    Tuple[Float[Tensor, "m 2"], Float[Tensor, "m 10"]],
]:
    """Generate a dataset of samples and their corresponding targets."""
    rng = torch.Generator()
    rng.manual_seed(seed)

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
    criterion = torch.nn.BCELoss()

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
            output = torch.clamp(output, 0, 1)
            loss = criterion(output, y_batch)
            loss.backward()
            optimiser.step()

            # Update loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += accuracy(y_batch, output).item()

            if (batch // batch_size) % test_every == 0:
                test_loss, test_acc = evaluate(model, x_test, y_test, criterion)
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
                "train_acc": train_accuracies[-1] * 100,
                "test_acc": test_accuracies[-1] * 100,
            }
        )

    return train_losses, test_losses, train_accuracies, test_accuracies


class Arguments(argparse.Namespace):
    id: str
    arch: str
    layers: int
    hidden: int
    epochs: int
    batch_size: int
    lr: float
    lr_div: int
    lr_step: int
    test_interval: int
    data_seed: int
    train_seed: int
    train_size: int
    test_size: int
    train_noise: float
    function: int
    dropout: float


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="")
    parser.add_argument("--arch", type=str, default="AO")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_div", type=int, default=10)
    parser.add_argument("--lr_step", type=int, default=20)
    parser.add_argument("--test_interval", type=int, default=500)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--train_seed", type=int, default=10)
    parser.add_argument("--train_size", type=int, default=10_000)
    parser.add_argument("--test_size", type=int, default=1_000)
    parser.add_argument("--train_noise", type=float, default=0.0025)
    parser.add_argument("--function", type=int, default=0, choices=range(len(ALL_FNS)))
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args(namespace=Arguments())

    # Extract the function name
    FN_NAMES = [fn[0] for fn in ALL_FNS]
    fn_name = FN_NAMES[args.function]
    desc = f"{args.arch}-{args.layers}-{args.hidden} ({fn_name})"

    # Generate the dataset
    d_train, d_test = make_dataset(
        (args.train_size, args.test_size), args.train_noise, args.data_seed
    )

    # Extract the data
    train_data = d_train[0], d_train[1][:, args.function].unsqueeze(-1)
    test_data = d_test[0], d_test[1][:, args.function].unsqueeze(-1)

    # Create the model
    if args.arch == "FC":
        model = FullyConnectedNetwork(2, 1, args.layers, args.hidden, args.dropout)
    else:
        model = NeuralLogicNetwork(2, 1, args.layers, args.hidden, args.arch == "AN")

    # Set the random seed
    torch.manual_seed(args.train_seed)

    # Train the model
    data = train_data, test_data
    train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        model,
        data,
        args.epochs,
        args.batch_size,
        args.lr,
        args.lr_div,
        args.lr_step,
        args.test_interval,
        desc,
    )

    # Save the results
    results = {
        "fn_name": fn_name,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
    }

    results_dir = f"results/{args.arch}-{args.layers}-{args.hidden}"
    os.makedirs(results_dir, exist_ok=True)

    results_file = f"{results_dir}/{args.function}-{args.data_seed}-{args.train_seed}"
    if args.id:
        results_file += f"-{args.id}"
    results_file += ".pt"

    torch.save(results, results_file)

    print(f"Results saved to {results_file}")
