from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Float

import os
import argparse

from tqdm import tqdm
import pandas as pd

import torch
from torch import Tensor
from torch.nn import Module

from models import FullyConnectedNetwork, NeuralLogicNetwork
from config import ALL_FNS


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
    desc: str = "Training",
) -> pd.DataFrame:
    (x_train, y_train), (x_test, y_test) = data

    x_train, y_train = x_train, y_train
    x_test, y_test = x_test, y_test

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

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

            pb.update(batch_size)

        if (epoch + 1) % lr_step == 0:
            for param_group in optimiser.param_groups:
                param_group["lr"] /= lr_div

        # Evaluate the model
        train_losses.append(epoch_loss / (len(x_train) / batch_size))
        train_accuracies.append(epoch_acc / (len(x_train) / batch_size))

        test_loss, test_acc = evaluate(model, x_test, y_test, criterion)
        test_losses.append(test_loss.item())
        test_accuracies.append(test_acc.item())

        pb.set_postfix(
            {
                "train_loss": train_losses[-1],
                "test_loss": test_losses[-1],
                "train_acc": train_accuracies[-1] * 100,
                "test_acc": test_accuracies[-1] * 100,
            }
        )

    df = pd.DataFrame(
        {
            "epoch": range(1, epoch + 2),
            "train_loss": train_losses,
            "test_loss": test_losses,
            "train_acc": train_accuracies,
            "test_acc": test_accuracies,
        },
    )

    return df


class ExpOneArgs(argparse.Namespace):
    id: str
    arch: str
    layers: int
    hidden: int
    epochs: int
    batch_size: int
    lr: float
    lr_div: int
    lr_step: int
    seed: int
    function: int
    dropout: float
    device: str
    save_model: bool
    out_dir: str


def run(args: ExpOneArgs):
    torch.manual_seed(args.seed)

    # Extract the function name
    FN_NAMES = [fn[0] for fn in ALL_FNS]
    fn_name = FN_NAMES[args.function]
    desc = f"{args.arch}-{args.layers}-{args.hidden} ({fn_name})"

    # Load the data
    dataset = torch.load("dataset.pt", weights_only=False)
    d_train, d_test = dataset["train"], dataset["test"]

    # Extract the data
    train_data = d_train[0].cuda(), d_train[1][fn_name].cuda()
    test_data = d_test[0].cuda(), d_test[1][fn_name].cuda()

    # Create the model
    if args.arch == "FC":
        model = FullyConnectedNetwork(2, 1, args.layers, args.hidden, args.dropout)
    else:
        model = NeuralLogicNetwork(2, 1, args.layers, args.hidden, args.arch == "AN")

    model.cuda()

    # Train the model
    data = train_data, test_data
    hp = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lr_div": args.lr_div,
        "lr_step": args.lr_step,
    }
    result = train_model(model, data, **hp, desc=desc)

    # Save the results

    results_dir = f"{args.out_dir}/{args.arch}-{args.layers}-{args.hidden}"
    os.makedirs(results_dir, exist_ok=True)

    results_file = f"{results_dir}/{args.function}-{args.seed}"
    if args.id:
        results_file += f"-{args.id}"
    results_file += ".csv"

    result.to_csv(results_file, index=False)

    if args.save_model:
        model_file = results_file.replace(".csv", ".pt")
        torch.save(model.state_dict(), model_file)


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
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--function", type=int, default=0, choices=range(len(ALL_FNS)))
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--out_dir", type=str, default="results")

    args = parser.parse_args(namespace=ExpOneArgs())

    run(args)
