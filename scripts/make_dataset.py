"""Gera um conjunto de instâncias de TSP com tours do LKH e grava-o num .npz.

Exemplo:
    python scripts/make_dataset.py --n 20 --num 100 --seed 1234
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from tsp_gnn.data import generate_instances, is_valid_tour, save_dataset, solve_lkh, tour_length

MAX_PROCS = 8  # servidor partilhado: no máximo 8 processos


def main():
    parser = argparse.ArgumentParser(description="Gera instâncias de TSP + tours do LKH.")
    parser.add_argument("--n", type=int, required=True, help="número de cidades por instância")
    parser.add_argument("--num", type=int, required=True, help="número de instâncias")
    parser.add_argument("--seed", type=int, default=1234, help="seed do gerador aleatório")
    parser.add_argument("--out", default=None, help="ficheiro .npz (por omissão: data/tsp{n}_seed{seed}.npz)")
    args = parser.parse_args()
    out = args.out or f"data/tsp{args.n}_seed{args.seed}.npz"

    coords = generate_instances(args.num, args.n, args.seed)

    # Cada instância é independente: resolvemos várias em paralelo.
    procs = min(MAX_PROCS, os.cpu_count())
    start = time.time()
    with Pool(procs) as pool:
        tours = pool.map(solve_lkh, coords, chunksize=max(1, args.num // (procs * 4)))
    elapsed = time.time() - start

    tours = np.stack(tours)
    assert all(is_valid_tour(t, args.n) for t in tours), "o LKH devolveu um tour inválido"
    lengths = np.array([tour_length(c, t) for c, t in zip(coords, tours)])

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_dataset(out, coords, tours, lengths, n=args.n, seed=args.seed)

    sem = lengths.std(ddof=1) / np.sqrt(len(lengths))  # erro-padrão da média
    print(f"instâncias: {args.num}  |  cidades: {args.n}  |  processos: {procs}")
    print(f"comprimento médio: {lengths.mean():.4f} ± {sem:.4f} (erro-padrão)  |  desvio-padrão: {lengths.std(ddof=1):.4f}")
    print(f"mín / máx: {lengths.min():.4f} / {lengths.max():.4f}")
    print(f"tempo LKH: {elapsed:.1f} s  ({1000 * elapsed / args.num:.1f} ms por instância)")
    print(f"gravado em: {out}")


if __name__ == "__main__":
    main()
