"""Phase 2, experiment 1: does edge-length normalization ("knn_mean") fix TSP-20 -> TSP-50/100 generalization?

Reads (nothing is written outside --out):
    results/baselines.json (+ _per_instance.npz)                  baselines on TSP-20 and TSP-50 (phase 1)
    results/phase2_exp1/baselines_tsp100.json                     baselines on TSP-100
    results/runs/{model}_seed{s}/eval*                            phase 1 models ("none") on TSP-20 and TSP-50
    results/phase2_exp1/phase1_tsp100/{model}_seed{s}/eval*       phase 1 models ("none") on TSP-100
    results/runs/{model}_knnmean_seed{s}/eval*                    new models ("knn_mean") on TSP-20/50/100
and writes results/phase2_exp1_table.md with:
    Table 1  mean gap ± std across seeds, "none" vs. "knn_mean", next to the baselines
    Table 2  paired per-instance comparison knn_mean − none (Wilcoxon, max p over seeds, Bonferroni)
Example:
    python scripts/phase2_exp1_table.py
"""

import argparse
import datetime
import importlib.util
import json
import os

import numpy as np

# scripts/ is not a package: load summarize.py directly to reuse the phase 1 statistics.
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summarize.py")
_spec = importlib.util.spec_from_file_location("summarize", _path)
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)
mean_std, paired_compare = summarize.mean_std, summarize.paired_compare
MODELS, BASELINES, ALPHA, TIE_TOL = summarize.MODELS, summarize.BASELINES, summarize.ALPHA, summarize.TIE_TOL

DATASETS = {"tsp20_test": "TSP-20", "tsp50_test": "TSP-50", "tsp100_test": "TSP-100"}
METHODS = {"greedy": "greedy", "greedy+2opt": "+ 2-opt"}
NORMS = ("none", "knn_mean")
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- loading

def load_eval(folder):
    """(eval.json, per-instance gaps) of one evaluation folder."""
    with open(os.path.join(folder, "eval.json")) as f:
        ev = json.load(f)
    with np.load(os.path.join(folder, "eval_per_instance.npz")) as f:
        return ev, {k: f[k] for k in f.files}


def load_run(folders):
    """Merge the evaluations of one run that live in several folders (e.g. TSP-20/50 + TSP-100).

    Returns {"meta": ..., "gaps": {dataset: {method: mean gap}}, "per_instance": {...}}.
    Fails if some dataset is missing: the table must not silently mix incomplete runs.
    """
    run = {"meta": None, "gaps": {}, "per_instance": {}}
    for folder in folders:
        ev, per_instance = load_eval(folder)
        run["meta"] = run["meta"] or ev["meta"]
        run["per_instance"].update(per_instance)
        for d in DATASETS:
            if d in ev:
                run["gaps"][d] = {m: ev[d][m]["gap_mean"] for m in METHODS}
    missing = [d for d in DATASETS if d not in run["gaps"]]
    if missing:
        raise FileNotFoundError(f"{folders}: no results for {missing}")
    return run


def load_all(results):
    """{(model, norm): [run per seed]} for both variants."""
    runs = {}
    for m in MODELS:
        runs[(m, "none")] = [load_run([os.path.join(results, "runs", f"{m}_seed{s}"),
                                       os.path.join(results, "phase2_exp1", "phase1_tsp100", f"{m}_seed{s}")])
                             for s in SEEDS]
        runs[(m, "knn_mean")] = [load_run([os.path.join(results, "runs", f"{m}_knnmean_seed{s}")])
                                 for s in SEEDS]
    return runs


def load_baselines(results):
    """Baselines {dataset: entry} and per-instance gaps, from the phase 1 file + the TSP-100 file."""
    summary, per_instance = {}, {}
    for path in (os.path.join(results, "baselines.json"),
                 os.path.join(results, "phase2_exp1", "baselines_tsp100.json")):
        with open(path) as f:
            summary.update({k: v for k, v in json.load(f).items() if k != "meta"})
        with np.load(os.path.splitext(path)[0] + "_per_instance.npz") as f:
            per_instance.update({k: f[k] for k in f.files})
    return summary, per_instance


def check_consistency(runs):
    """Warn if "none" and "knn_mean" differ in anything other than edge_norm, seed and pos_weight."""
    ignore = ("seed", "pos_weight", "edge_norm")
    for m in MODELS:
        configs = [{k: v for k, v in r["meta"]["config"].items() if k not in ignore}
                   for norm in NORMS for r in runs[(m, norm)]]
        if any(c != configs[0] for c in configs):
            print(f"WARNING: the runs of {m} have different configurations: {configs}")
        for norm in NORMS:
            found = {r["meta"]["config"].get("edge_norm", "none") for r in runs[(m, norm)]}
            if found != {norm}:
                raise ValueError(f"{m} {norm}: the checkpoints say edge_norm={found}")


# ---------------------------------------------------------------- tables

def table_gaps(baselines, runs):
    """Table 1: mean gap ± std across seeds, for every model and normalization, next to the baselines."""
    cols = [(d, m) for d in DATASETS for m in METHODS]
    lines = ["| Method | edge_norm | " + " | ".join(f"{DATASETS[d]} {METHODS[m]}" for d, m in cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for key, name in BASELINES.items():
        cells = [f"{baselines[d][key if m == 'greedy' else key + '+2opt']['gap_mean']:.2f}" for d, m in cols]
        lines.append(f"| {name} | — | " + " | ".join(cells) + " |")
    for key, name in MODELS.items():
        for norm in NORMS:
            cells = [mean_std([r["gaps"][d][m] for r in runs[(key, norm)]]) for d, m in cols]
            lines.append(f"| {name} | {norm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare_variants(runs_none, runs_knn, dataset, method, threshold):
    """Paired comparison knn_mean − none: seed s of one against seed s of the other, then aggregated.

    Same rule as phase 1: significant only if max p < threshold and the median has the same sign in every seed.
    """
    key = f"{dataset}__{method}"
    res = [paired_compare(k["per_instance"][key], n["per_instance"][key]) for n, k in zip(runs_none, runs_knn)]
    avg = {k: float(np.mean([x[k] for x in res])) for k in res[0]}
    p_max = max(x["p"] for x in res)
    same_sign = len({np.sign(x["median_diff"]) for x in res}) == 1
    direction = avg["median_diff"] if avg["median_diff"] != 0 else avg["mean_diff"]
    if p_max < threshold and same_sign:
        verdict = "**knn_mean better**" if direction < 0 else "**knn_mean worse**"
    else:
        verdict = "n.s."
    return {**avg, "p_max": p_max, "verdict": verdict}


def table_paired(runs, threshold):
    """Table 2: one row per model × dataset × method."""
    lines = ["| Model | Dataset | Method | knn_mean better | Equal | knn_mean worse | Mean diff. (pp) "
             "| Median diff. (pp) | max p (seeds) | Conclusion |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, name in MODELS.items():
        for d in DATASETS:
            for m in METHODS:
                c = compare_variants(runs[(key, "none")], runs[(key, "knn_mean")], d, m, threshold)
                lines.append(f"| {name} | {DATASETS[d]} | {METHODS[m]} | {c['better']:.1f}% | {c['equal']:.1f}% "
                             f"| {c['worse']:.1f}% | {c['mean_diff']:+.2f} | {c['median_diff']:+.2f} "
                             f"| {c['p_max']:.1e} | {c['verdict']} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Phase 2, experiment 1: none vs. knn_mean.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/phase2_exp1_table.md")
    args = parser.parse_args()

    baselines, _ = load_baselines(args.results)
    runs = load_all(args.results)
    check_consistency(runs)
    n_tests = len(MODELS) * len(DATASETS) * len(METHODS)
    threshold = ALPHA / n_tests  # Bonferroni correction over all the tests of Table 2

    commits = sorted({r["meta"]["train_git_commit"] for rs in runs.values() for r in rs})
    report = "\n".join([
        "# Phase 2, experiment 1 — edge-length normalization",
        "",
        f"Generated by `scripts/phase2_exp1_table.py` on {datetime.date.today().isoformat()}. "
        f"Training commits: {', '.join(commits)}.",
        "",
        "All models are trained on TSP-20 only (12000 instances, 20 epochs, H=64, L=6, k=10, seeds 0, 1, 2). "
        "`none` = raw edge lengths (phase 1 models); `knn_mean` = each edge length divided by the mean edge "
        "length of the instance's k-NN graph. TSP-50 and TSP-100 measure generalization.",
        "",
        "## Table 1 — mean gap to LKH (%)",
        "",
        "Models: mean ± std **across seeds**. Baselines: deterministic.",
        "",
        table_gaps(baselines, runs),
        "",
        "## Table 2 — paired per-instance comparison: knn_mean vs. none",
        "",
        f"For each seed s, the gap of each instance with `knn_mean` (seed s) is compared with the gap with `none` "
        f"(seed s); the table shows the means across the 3 seed pairs. *Equal* = difference ≤ {TIE_TOL:g} pp. "
        f"Diff. = knn_mean − none (negative = normalization helps). Two-sided Wilcoxon test (ties dropped); "
        f"*max p* is the largest p-value across seeds. Significant only if max p < {threshold:.2e} "
        f"(Bonferroni: {ALPHA}/{n_tests} tests) and the median has the same sign in every seed.",
        "",
        table_paired(runs, threshold),
        "",
    ])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
