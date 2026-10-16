"""Utilitários partilhados: seed, informação do git e limite de memória da GPU."""

import random
import subprocess

import numpy as np
import torch


def set_seed(seed):
    """Fixa a seed de todos os geradores aleatórios (random, NumPy, PyTorch CPU e GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_info():
    """Commit atual e se há alterações por gravar (para saber que código gerou os resultados)."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    return commit, dirty


def limit_gpu_memory(device, max_gib=5.5):
    """Limita o alocador do PyTorch a `max_gib` GiB nesta GPU (partilhada).

    O contexto do CUDA ocupa ainda algumas centenas de MB fora deste limite:
    com 5,5 GiB o processo fica abaixo de 6 GiB no total. Devolve a fração usada.
    """
    # set_per_process_memory_fraction exige um índice (cuda:0), não só "cuda"
    index = device.index if device.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    fraction = min(1.0, max_gib * 2**30 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    return fraction
