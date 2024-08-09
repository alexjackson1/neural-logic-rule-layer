import os

import torch
from config import ALL_FNS
from experiment_1 import Arguments


def slurm_task_id() -> int:
    return int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))


def slurm_job_id() -> int:
    return int(os.getenv("SLURM_JOB_ID", "0"))


def seed() -> int:
    return slurm_job_id()


# Possible architectures and hyperparameters
ARCH_NAMES = ["AN", "AO", "FC"]
NUM_LAYERS = {"AN": [2, 3], "AO": [2, 3], "FC": [2, 3, 4]}
HIDDEN_UNITS = {"AN": [5, 25, 50], "AO": [5, 25, 50], "FC": [5, 25, 50, 100]}

# Dataset hyperparameters
NUM_FNS = len(ALL_FNS)
FN_NAMES = [fn[0] for fn in ALL_FNS]

# Number of seeds
NUM_RUNS = 10


FC_DROPOUT_RATE = 0.2

NUM_EPOCHS = 100
BATCH_SIZE = 20
LEARNING_RATE = {"AN": 0.1, "AO": 0.1, "FC": 0.001}

LR_DIVISOR = 10
LR_EPOCH_STEP = 20

TEST_BATCH_INTERVAL = 500

DATASET_SEED = 10
TRAIN_SIZE = 10_000
TEST_SIZE = 1_000
TRAIN_NOISE = 0.0025


def to_arguments(job_id: int, task_id: int) -> Arguments:
    # Calculate the total number of runs
    arch_counts = {a: len(NUM_LAYERS[a]) * len(HIDDEN_UNITS[a]) for a in ARCH_NAMES}
    total_runs = sum(arch_counts.values()) * NUM_FNS * NUM_RUNS
    print(total_runs)

    if task_id >= total_runs:
        raise ValueError("Task ID out of bounds")

    # Generate a random permutation of the task IDs
    generator = torch.Generator()
    generator.manual_seed(job_id)
    task_ids = torch.randperm(total_runs, generator=generator).tolist()

    # Map the SLURM task ID to a random task ID
    random_task_id = task_ids[task_id]

    # Find the architecture and hyperparameters
    architecture = None
    for a in ARCH_NAMES:
        if random_task_id < arch_counts[a] * NUM_FNS * NUM_RUNS:
            architecture = a
            break
        random_task_id -= arch_counts[a] * NUM_FNS * NUM_RUNS

    if architecture is None:
        raise ValueError("Invalid task ID")

    num_layers, num_hidden_units = None, None
    for l_ in NUM_LAYERS[architecture]:
        for h_ in HIDDEN_UNITS[architecture]:
            if random_task_id < NUM_FNS * NUM_RUNS:
                num_layers, num_hidden_units = l_, h_
                break
            random_task_id -= NUM_FNS * NUM_RUNS

    if num_layers is None or num_hidden_units is None:
        raise ValueError("Invalid task ID")

    fn_idx = random_task_id // NUM_RUNS
    run_idx = random_task_id % NUM_RUNS

    return Arguments(
        id=f"{run_idx}",
        arch=architecture,
        layers=num_layers,
        hidden=num_hidden_units,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE[architecture],
        lr_div=LR_DIVISOR,
        lr_step=LR_EPOCH_STEP,
        test_interval=TEST_BATCH_INTERVAL,
        data_seed=42,
        train_seed=task_id,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        train_noise=TRAIN_NOISE,
        function=fn_idx,
        dropout=FC_DROPOUT_RATE,
    )


if __name__ == "__main__":
    args = to_arguments(slurm_job_id(), slurm_task_id())
    print(args)
    print(args.arch, args.layers, args.hidden, args.function)
    print(args.train_seed)
    print(args.train_size)
    print(args.test_size)
    print(args.train_noise)
    print(args.dropout)
    print(args.epochs)
    print(args.batch_size)
    print(args.lr)
    print(args.lr_div)
    print(args.lr_step)
    print(args.test_interval)
    print(args.data_seed)
    print(args.id)
