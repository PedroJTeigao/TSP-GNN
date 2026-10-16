"""Geração de instâncias de TSP e rótulos com o LKH (via elkai).

Uma instância é um array (n, 2) de coordenadas em [0,1]².
Um tour é uma permutação de 0..n-1 que começa na cidade 0; a aresta de volta
(última cidade -> primeira) está implícita.
"""

import elkai
import numpy as np

# O LKH só aceita distâncias inteiras: multiplicamos por este fator antes de
# arredondar. Com 1e6, o erro de arredondamento é ~1e-6 por aresta (desprezável).
SCALE = 1_000_000


def generate_instances(num, n, seed):
    """Gera `num` instâncias com `n` cidades uniformes em [0,1]². Forma: (num, n, 2)."""
    rng = np.random.default_rng(seed)  # mesma seed -> mesmos dados
    return rng.random((num, n, 2))


def distance_matrix(coords):
    """Matriz (n, n) das distâncias euclidianas entre todas as cidades."""
    diff = coords[:, None, :] - coords[None, :, :]  # (n, n, 2): diferença entre cada par
    return np.sqrt((diff ** 2).sum(axis=-1))


def solve_lkh(coords, runs=10):
    """Resolve uma instância com o LKH. Devolve o tour como array (n,) de inteiros."""
    dist_int = np.rint(distance_matrix(coords) * SCALE).astype(np.int64)
    tour = elkai.solve_int_matrix(dist_int.tolist(), runs=runs)
    tour = np.asarray(tour, dtype=np.int64)
    # Garantir que o tour começa na cidade 0 (rodar não muda o ciclo).
    start = int(np.where(tour == 0)[0][0])
    return np.roll(tour, -start)


def tour_length(coords, tour):
    """Comprimento real do tour (distâncias em float), incluindo a aresta de volta."""
    ordered = coords[tour]                    # cidades pela ordem do tour
    nxt = np.roll(ordered, -1, axis=0)        # a seguinte de cada uma (a última liga à primeira)
    return float(np.sqrt(((ordered - nxt) ** 2).sum(axis=-1)).sum())


def is_valid_tour(tour, n):
    """True se o tour visita cada uma das n cidades exatamente uma vez."""
    return len(tour) == n and np.array_equal(np.sort(tour), np.arange(n))


def save_dataset(path, coords, tours, lengths, **meta):
    """Grava coordenadas, tours e comprimentos num .npz comprimido (+ metadados, ex. n, seed)."""
    np.savez_compressed(path, coords=coords, tours=tours, lengths=lengths, **meta)


def load_dataset(path):
    """Lê um .npz gravado por save_dataset e devolve um dicionário de arrays."""
    with np.load(path) as f:
        return {key: f[key] for key in f.files}
