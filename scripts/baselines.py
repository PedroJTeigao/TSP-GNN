"""Mede o gap face ao LKH das heurísticas clássicas (guloso e Christofides, com e sem 2-opt).

Grava o resumo em results/baselines.json e os gaps por instância em
results/baselines_per_instance.npz.
Exemplo:
    python scripts/baselines.py --data data/tsp20_test.npz data/tsp50_test.npz
"""

import argparse
import datetime
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from tsp_gnn.baselines import christofides_tour, greedy_distance_tour
from tsp_gnn.data import load_dataset, tour_length
from tsp_gnn.decode import optimality_gap, two_opt, verify_tour
from tsp_gnn.utils import git_info

MAX_PROCS = 8  # servidor partilhado: no máximo 8 processos
METHODS = ["greedy", "greedy+2opt", "christofides", "christofides+2opt"]


def run_instance(args):
    """Corre os 4 métodos numa instância. Devolve {método: (gap %, tempo em ms)}."""
    coords, opt_len = args
    n = len(coords)
    out = {}
    for name, heuristic in [("greedy", greedy_distance_tour), ("christofides", christofides_tour)]:
        start = time.perf_counter()
        tour = heuristic(coords)
        t_build = time.perf_counter() - start
        out[name] = (optimality_gap(tour_length(coords, tour), opt_len), 1000 * t_build)

        start = time.perf_counter()
        improved = verify_tour(two_opt(coords, tour), n)
        t_2opt = time.perf_counter() - start
        # o tempo do "+2opt" inclui construir o tour inicial
        out[name + "+2opt"] = (optimality_gap(tour_length(coords, improved), opt_len),
                               1000 * (t_build + t_2opt))
    return out


def main():
    parser = argparse.ArgumentParser(description="Baselines sem aprendizagem: guloso e Christofides.")
    parser.add_argument("--data", nargs="+", default=["data/tsp20_test.npz", "data/tsp50_test.npz"])
    parser.add_argument("--out", default="results/baselines.json")
    args = parser.parse_args()

    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty,
                        "date": datetime.date.today().isoformat()}}
    per_instance = {}

    procs = min(MAX_PROCS, os.cpu_count())
    for path in args.data:
        name = os.path.splitext(os.path.basename(path))[0]   # ex.: tsp20_test
        d = load_dataset(path)
        with Pool(procs) as pool:
            results = pool.map(run_instance, zip(d["coords"], d["lengths"]), chunksize=8)

        entry = {"n": int(d["coords"].shape[1]), "num": len(results),
                 "lkh_mean_len": float(d["lengths"].mean())}
        for m in METHODS:
            gaps = np.array([r[m][0] for r in results])
            ms = np.array([r[m][1] for r in results])
            lengths = d["lengths"] * (1 + gaps / 100)        # comprimento do tour do método
            entry[m] = {"mean_len": float(lengths.mean()), "gap_mean": float(gaps.mean()),
                        "gap_std": float(gaps.std(ddof=1)), "gap_max": float(gaps.max()),
                        "ms_per_inst": float(ms.mean())}
            per_instance[f"{name}__{m}"] = gaps
        summary[name] = entry

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    npz_path = os.path.splitext(args.out)[0] + "_per_instance.npz"
    np.savez_compressed(npz_path, **per_instance)

    # Tabela: uma linha por método, colunas por conjunto de dados.
    names = [k for k in summary if k != "meta"]
    print(f"{'método':<18}" + "".join(f"{n + ' (méd / máx / ms)':>34}" for n in names))
    for m in METHODS:
        cells = "".join(f"{summary[n][m]['gap_mean']:>12.2f}% /{summary[n][m]['gap_max']:>7.2f}% /"
                        f"{summary[n][m]['ms_per_inst']:>8.1f}  " for n in names)
        print(f"{m:<18}{cells}")
    print(f"\ncommit {commit}{' (com alterações por gravar)' if dirty else ''}"
          f"  |  gravado em: {args.out} e {npz_path}")


if __name__ == "__main__":
    main()
