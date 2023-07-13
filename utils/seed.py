import numpy as np
import random
import torch


def init_seed(args):
    # Set the seed for Torch CUDA operations
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Set the seed for Torch CPU operations
    torch.manual_seed(args.seed)
    # Set the seed for NumPy operations
    np.random.seed(args.seed)
    # Set the seed for random number generation
    random.seed(args.seed)


def seed_worker(worker_id):
    # Generate a seed for the worker based on the initial seed
    worker_seed = torch.initial_seed() % 2**32
    # Set the seed for NumPy operations in the worker
    np.random.seed(worker_seed)
    # Set the seed for random number generation in the worker
    random.seed(worker_seed)
