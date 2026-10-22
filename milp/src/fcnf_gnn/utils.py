"""Shared utilities: seed, git information and GPU memory limit (copied from tsp_gnn.utils)."""

import random
import subprocess

import numpy as np
import torch


def set_seed(seed):
    """Set the seed of every random generator (random, NumPy, PyTorch CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_info():
    """Current commit and whether there are uncommitted changes (to know which code produced the results)."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    return commit, dirty


def limit_gpu_memory(device, max_gib=5.5):
    """Limit the PyTorch allocator to `max_gib` GiB on this (shared) GPU.

    The CUDA context still takes a few hundred MB outside this limit:
    with 5.5 GiB the process stays below 6 GiB in total. Returns the fraction used.
    """
    # set_per_process_memory_fraction requires an index (cuda:0), not just "cuda"
    index = device.index if device.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    fraction = min(1.0, max_gib * 2**30 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    return fraction
