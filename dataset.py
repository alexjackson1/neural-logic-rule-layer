from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Float

import torch
from torch import Tensor

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


if __name__ == "__main__":
    train_data, test_data = make_dataset((10_000, 1_000), 0.0025)

    compile = lambda y: {f: t for (f, _), t in zip(ALL_FNS, torch.split(y, 1, dim=1))}
    train_ys, test_ys = compile(train_data[1]), compile(test_data[1])

    dataset = {
        "train": (train_data[0], train_ys),
        "test": (test_data[0], test_ys),
        "functions": [fn_name for fn_name, _ in ALL_FNS],
    }

    torch.save(dataset, "dataset.pt")
    print(f"Dataset saved to dataset.pt")
