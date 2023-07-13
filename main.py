import submitit
import sys
import os
from utils.process import processDataset
from utils.experiments import *
from utils.parser import get_args

os.environ["WANDB__SERVICE_WAIT"] = "300"
# sys.path.append('.')

# Ensure that all operations are deterministic on GPU (if used) for reproducibility


def main():
    args = get_args()
    executor = submitit.AutoExecutor(
        folder="LOGFOLDER", slurm_max_num_timeout=30)
    executor.update_parameters(
        mem_gb=32,
        slurm_gpus_per_task=1,
        tasks_per_node=1,  # one task per GPU
        slurm_cpus_per_gpu=1,
        nodes=1,
        timeout_min=120,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition="prod",
        slurm_signal_delay_s=120,
        slurm_array_parallelism=15)
    experiments = []
    experiments = experiments+[args]
    executor.map_array(processDataset, [experiments[0]])


if __name__ == '__main__':
    main()
