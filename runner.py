import os
from beartype.typing import NamedTuple

from config import ALL_FNS
from experiment_1 import ExpOneArgs, run


def slurm_task_id() -> int:
    return int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))


def slurm_job_id() -> int:
    return int(os.getenv("SLURM_JOB_ID", "0"))


# Possible architectures and hyperparameters
ARCH_NAMES = ["AN", "AO", "FC"]
NUM_LAYERS = {"AN": [2, 3], "AO": [2, 3], "FC": [2, 3, 4]}
HIDDEN_UNITS = {"AN": [5, 25, 50], "AO": [5, 25, 50], "FC": [5, 25, 50, 100]}

# Dataset hyperparameters
NUM_FNS = len(ALL_FNS)
FN_NAMES = [fn[0] for fn in ALL_FNS]

# Number of seeds
NUM_RUNS = 10

NUM_EPOCHS = 100
BATCH_SIZE = 20
LEARNING_RATE = {"AN": 0.1, "AO": 0.1, "FC": 0.001}

FC_DROPOUT_RATE = 0.2

LR_DIVISOR = 10
LR_EPOCH_STEP = 20


class ParsedParams(NamedTuple):
    run_idx: int
    fn_idx: int
    architecture: str
    num_layers: int
    num_hidden_units: int


def to_params(task_id: int) -> ParsedParams:
    # Calculate the total number of runs
    arch_counts = {a: len(NUM_LAYERS[a]) * len(HIDDEN_UNITS[a]) for a in ARCH_NAMES}
    total_networks = sum(arch_counts.values())
    total_runs = total_networks * NUM_FNS * NUM_RUNS

    if task_id >= total_runs:
        raise ValueError("Task ID out of bounds")

    # Find the architecture and hyperparameters
    counter = task_id

    run_idx = None
    for i in range(NUM_RUNS):
        if counter < total_networks * NUM_FNS:
            run_idx = i
            break
        counter -= total_networks * NUM_FNS

    if run_idx is None:
        raise ValueError("Invalid task ID")

    fn_idx = None
    for i in range(NUM_FNS):
        if counter < total_networks:
            fn_idx = i
            break
        counter -= total_networks

    if fn_idx is None:
        raise ValueError("Invalid task ID")

    architecture = None
    for a in ARCH_NAMES:
        if counter < arch_counts[a]:
            architecture = a
            break
        counter -= arch_counts[a]

    if architecture is None:
        raise ValueError("Invalid task ID")

    num_layers, num_hidden_units = None, None
    for l_ in NUM_LAYERS[architecture]:
        for h_ in HIDDEN_UNITS[architecture]:
            if counter == 0:
                num_layers, num_hidden_units = l_, h_
                break

            counter -= 1

        if num_layers is not None:
            break

    if num_layers is None or num_hidden_units is None:
        raise ValueError("Invalid task ID")

    return ParsedParams(run_idx, fn_idx, architecture, num_layers, num_hidden_units)


def to_arguments(seed: int, params: ParsedParams) -> ExpOneArgs:
    return ExpOneArgs(
        id=f"{params.run_idx}",
        arch=params.architecture,
        layers=params.num_layers,
        hidden=params.num_hidden_units,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE[params.architecture],
        lr_div=LR_DIVISOR,
        lr_step=LR_EPOCH_STEP,
        seed=seed,
        function=params.fn_idx,
        dropout=FC_DROPOUT_RATE,
    )


if __name__ == "__main__":
    job_id = slurm_job_id()
    task_id = slurm_task_id()

    params = to_params(task_id)
    args = to_arguments(job_id, params)

    run(args)
