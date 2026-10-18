"""Phase 2, batch 2 report: every configuration against phase 1, on TSP-20, TSP-50 and TSP-100.

Configurations (all models trained with 12000 instances, 20 epochs, H=64, L=6, seeds 0, 1, 2):
    phase1           phase 1 models (k=10, BatchNorm, raw distances)
    knn_mean         experiment 1: distances divided by the instance's mean k-NN edge length
    kfrac            experiment 2: phase 1 models evaluated on graphs with k = 0.5 n
    knn_mean_kfrac   experiment 2: knn_mean models evaluated on graphs with k = 0.5 n
    bn_batch         experiment 2b: phase 1 models, BatchNorm with the statistics of each test batch
    layernorm        experiment 3: models trained with LayerNorm (k=10)
    layernorm_kfrac  experiment 3: LayerNorm models evaluated on graphs with k = 0.5 n
    train50          control: phase 1 configuration trained on TSP-50 instead of TSP-20
Missing results (e.g. a failed step) are reported as "missing" instead of stopping the report.

Writes results/phase2_batch2_report.md (Table 1: gap ± std across seeds; Table 2: paired
Wilcoxon tests against phase 1 with Bonferroni; Summary; Decisions).
Example:
    python scripts/phase2_batch2_report.py
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
SEEDS = (0, 1, 2)
B2 = "phase2_batch2"  # folder (inside results/) for the evaluation-only outputs of this batch

# name -> (description, function(model, seed) -> folders relative to results/)
CONFIGS = {
    "phase1": ("Phase 1 (k=10, BatchNorm, raw distances)",
               lambda m, s: [f"runs/{m}_seed{s}", f"phase2_exp1/phase1_tsp100/{m}_seed{s}"]),
    "knn_mean": ("Exp. 1: edge_norm knn_mean", lambda m, s: [f"runs/{m}_knnmean_seed{s}"]),
    "kfrac": ("Exp. 2: phase 1 models, test graphs k = 0.5 n", lambda m, s: [f"{B2}/kfrac05/{m}_seed{s}"]),
    "knn_mean_kfrac": ("Exp. 2: knn_mean models, test graphs k = 0.5 n",
                       lambda m, s: [f"{B2}/kfrac05/{m}_knnmean_seed{s}"]),
    "bn_batch": ("Exp. 2b: phase 1 models, BatchNorm batch statistics", lambda m, s: [f"{B2}/bnbatch/{m}_seed{s}"]),
    "layernorm": ("Exp. 3: LayerNorm (k=10)", lambda m, s: [f"runs/{m}_layernorm_seed{s}"]),
    "layernorm_kfrac": ("Exp. 3: LayerNorm, test graphs k = 0.5 n",
                        lambda m, s: [f"{B2}/kfrac05/{m}_layernorm_seed{s}"]),
    "train50": ("Control: trained on TSP-50", lambda m, s: [f"runs/{m}_train50_seed{s}"]),
}
CHANGES = [c for c in CONFIGS if c != "phase1"]
DECISIONS = [
    "`k_frac`: k = min(n − 1, round(k_frac × n)); for n = 20, 50, 100 and k_frac = 0.5 this is exactly 10, 25, 50. "
    "Without `--k-frac` the fixed k = 10 of phase 1 is used (bit-for-bit identical results, checked with "
    "`array_equal` on copies of the phase 1 and experiment 1 models).",
    "Experiment 2 evaluates only on TSP-50 and TSP-100, as requested: on TSP-20, k = 0.5 × 20 = 10 is the phase 1 "
    "graph (a unit test checks the graphs are identical), so those cells are left as \"—\". For the LayerNorm models "
    "(experiment 3) the k = 0.5 n evaluation also includes TSP-20, as requested there; it must equal the k = 10 value.",
    "`knn_mean` combined with k = 0.5 n divides by the mean edge length of the *larger* k-NN graph, i.e. the "
    "definition of knn_mean applied literally to the test graph (not the k = 10 mean seen in training).",
    "Evaluations with k = 0.5 n use 64 graphs per forward pass instead of 256 (peak GPU memory 0.7 GB instead of "
    "2.8 GB on the shared GPU). In eval mode the result does not depend on the batch size.",
    "`--bn-batch-stats`: the BatchNorms of a *copy* of the model drop the stored running statistics, so each layer "
    "normalizes with the mean/variance of the current test batch: 256 graphs of the same size (the last batch has "
    "232 graphs on TSP-20 and 244 on TSP-50/100). This makes the prediction for one instance depend on the other "
    "instances of its batch (transductive); per-instance statistics (batch of 1 graph) were not tried. Weights and "
    "files are not modified.",
    "LayerNorm (experiment 3) replaces every BatchNorm of the encoder (the per-layer norms of GCN/GATv2 and both "
    "norms of each Gated GCN layer) with `nn.LayerNorm(64)` (PyTorch defaults); nothing else changes "
    "(raw distances, k = 10, same seeds, epochs, learning rate and model selection on TSP-20 validation).",
    "Control: new data `tsp50_train` (12000 instances, seed 6) and `tsp50_val` (500, seed 7), LKH with the phase 1 "
    "settings. Model selection uses `tsp50_val` (in-distribution, as phase 1 used `tsp20_val`). Same epochs, batch "
    "size (64 graphs, so 2.5× more edges per batch) and learning rate; pos_weight is recomputed from the TSP-50 "
    "data. Evaluated on TSP-50 and TSP-100 only, as requested.",
    "Phase 1 TSP-100 results are the ones computed in experiment 1 (`results/phase2_exp1/phase1_tsp100/`); "
    "they were not recomputed.",
    "Paired comparisons pair seed s of a configuration with seed s of phase 1. For kfrac and bn_batch this is the "
    "same trained network; for retrained configurations the seed pairing is arbitrary and the instances are what is "
    "paired. The Bonferroni correction is applied over *all* tests in Table 2 (stricter than per experiment).",
    "Conservative reading of the rules: nothing from phase 1 or experiment 1 was rewritten (new folders only); a "
    "failed step is logged and the steps that depend on it are skipped; `sqrt_n` was not trained; the report is "
    "copied to `docs/` but not committed.",
]


# ---------------------------------------------------------------- loading

def load_folder(path):
    """(eval.json, per-instance gaps) of one evaluation folder, or None if it does not exist."""
    if not os.path.exists(os.path.join(path, "eval.json")):
        return None
    with open(os.path.join(path, "eval.json")) as f:
        ev = json.load(f)
    with np.load(os.path.join(path, "eval_per_instance.npz")) as f:
        return ev, {k: f[k] for k in f.files}


def load_config(results, folders_of):
    """{model: {dataset: {"gaps": {method: [per seed]}, "per_instance": [per seed]}}}.

    A dataset is kept for a model only if all seeds have it (no averages over incomplete runs).
    """
    out = {}
    for m in MODELS:
        per_seed = []
        for s in SEEDS:
            ev_all, pi_all = {}, {}
            for folder in folders_of(m, s):
                loaded = load_folder(os.path.join(results, folder))
                if loaded:
                    ev, pi = loaded
                    ev_all.update({d: ev[d] for d in DATASETS if d in ev})
                    pi_all.update(pi)
            per_seed.append((ev_all, pi_all))
        out[m] = {}
        for d in DATASETS:
            if all(d in ev for ev, _ in per_seed):
                out[m][d] = {"gaps": {meth: [ev[d][meth]["gap_mean"] for ev, _ in per_seed] for meth in METHODS},
                             "per_instance": [pi for _, pi in per_seed]}
    return out


def load_baselines(results):
    summary = {}
    for path in (os.path.join(results, "baselines.json"),
                 os.path.join(results, "phase2_exp1", "baselines_tsp100.json")):
        if os.path.exists(path):
            with open(path) as f:
                summary.update({k: v for k, v in json.load(f).items() if k != "meta"})
    return summary


# ---------------------------------------------------------------- tables

def gap_cell(cfg, m, d, meth, expected):
    if d in cfg[m]:
        return mean_std(cfg[m][d]["gaps"][meth])
    return "missing" if d in expected else "—"


def expected_datasets(name):
    """Datasets each configuration is supposed to have (others are "—", not "missing")."""
    return {"kfrac": ["tsp50_test", "tsp100_test"], "knn_mean_kfrac": ["tsp50_test", "tsp100_test"],
            "train50": ["tsp50_test", "tsp100_test"]}.get(name, list(DATASETS))


def table_gaps(baselines, configs):
    cols = [(d, meth) for d in DATASETS for meth in METHODS]
    lines = ["| Model | Configuration | " + " | ".join(f"{DATASETS[d]} {METHODS[x]}" for d, x in cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for key, name in BASELINES.items():
        cells = [f"{baselines[d][key if x == 'greedy' else key + '+2opt']['gap_mean']:.2f}" if d in baselines
                 else "missing" for d, x in cols]
        lines.append(f"| {name} | baseline | " + " | ".join(cells) + " |")
    for m, mname in MODELS.items():
        for c, cfg in configs.items():
            cells = [gap_cell(cfg, m, d, x, expected_datasets(c)) for d, x in cols]
            lines.append(f"| {mname} | {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare(cfg_a, cfg_ref, m, d, meth, threshold):
    """Paired comparison a − reference (seed s vs. seed s), aggregated as in phase 1."""
    key = f"{d}__{meth}"
    res = [paired_compare(a[key], r[key])
           for a, r in zip(cfg_a[m][d]["per_instance"], cfg_ref[m][d]["per_instance"])]
    avg = {k: float(np.mean([x[k] for x in res])) for k in res[0]}
    p_max = max(x["p"] for x in res)
    same_sign = len({np.sign(x["median_diff"]) for x in res}) == 1
    direction = avg["median_diff"] if avg["median_diff"] != 0 else avg["mean_diff"]
    significant = p_max < threshold and same_sign
    verdict = ("better" if direction < 0 else "worse") if significant else "n.s."
    return {**avg, "p_max": p_max, "verdict": verdict}


def all_comparisons(configs):
    """List of (config, model, dataset, method) where both the change and phase 1 have the results."""
    ref = configs["phase1"]
    return [(c, m, d, meth) for c in CHANGES for m in MODELS for d in DATASETS for meth in METHODS
            if d in configs[c][m] and d in ref[m]]


def run_comparisons(configs, threshold):
    return {t: compare(configs[t[0]], configs["phase1"], t[1], t[2], t[3], threshold)
            for t in all_comparisons(configs)}


def table_paired(comparisons):
    lines = ["| Configuration | Model | Dataset | Method | Better | Equal | Worse | Mean diff. (pp) "
             "| Median diff. (pp) | max p (seeds) | Conclusion |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (c, m, d, meth), r in comparisons.items():
        verdict = f"**{r['verdict']}**" if r["verdict"] != "n.s." else "n.s."
        lines.append(f"| {c} | {MODELS[m]} | {DATASETS[d]} | {METHODS[meth]} | {r['better']:.1f}% "
                     f"| {r['equal']:.1f}% | {r['worse']:.1f}% | {r['mean_diff']:+.2f} | {r['median_diff']:+.2f} "
                     f"| {r['p_max']:.1e} | {verdict} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- summary

def mean_gap(cfg, m, d, meth):
    return float(np.mean(cfg[m][d]["gaps"][meth])) if d in cfg[m] else None


def summary_points(configs, comparisons, baselines):
    """Data-driven summary (at most 10 points): one per change, then the control."""
    points = []
    for c in CHANGES:
        if c == "train50":
            continue
        rows = {t: r for t, r in comparisons.items() if t[0] == c}
        if not rows:
            points.append(f"**{c}**: no results (see the log).")
            continue
        parts = []
        for d in DATASETS:
            sub = [r for t, r in rows.items() if t[2] == d]
            if sub:
                better = sum(r["verdict"] == "better" for r in sub)
                worse = sum(r["verdict"] == "worse" for r in sub)
                mean_diff = np.mean([r["mean_diff"] for r in sub])
                parts.append(f"{DATASETS[d]}: {better} better / {worse} worse of {len(sub)} "
                             f"(mean diff. {mean_diff:+.2f} pp)")
        n_better = sum(r["verdict"] == "better" for r in rows.values())
        n_worse = sum(r["verdict"] == "worse" for r in rows.values())
        word = ("helped" if n_better > 0 and n_worse == 0 else "hurt" if n_worse > 0 and n_better == 0
                else "mixed" if n_better and n_worse else "no significant effect")
        points.append(f"**{c}** ({CONFIGS[c][0]}) — {word}. Significant cells (3 models × 2 methods) vs. phase 1: "
                      + "; ".join(parts) + ".")

    # Control: is the TSP-50 failure a generalization problem or a capacity problem?
    ref, ctrl = configs["phase1"], configs["train50"]
    base50 = baselines.get("tsp50_test", {}).get("greedy", {}).get("gap_mean")
    base20 = baselines.get("tsp20_test", {}).get("greedy", {}).get("gap_mean")
    lines = []
    for m, mname in MODELS.items():
        g50_ctrl, g50_ref = mean_gap(ctrl, m, "tsp50_test", "greedy"), mean_gap(ref, m, "tsp50_test", "greedy")
        g20_ref = mean_gap(ref, m, "tsp20_test", "greedy")
        if None in (g50_ctrl, g50_ref, g20_ref, base50, base20):
            continue
        lines.append((mname, g50_ctrl, g50_ref, g20_ref,
                      100 * (1 - g50_ctrl / base50), 100 * (1 - g20_ref / base20)))
    if not lines:
        points.append("**Control (train on TSP-50)**: no results (see the log).")
    else:
        detail = "; ".join(f"{n}: {a:.2f}% (phase 1: {b:.2f}%)" for n, a, b, *_ in lines)
        points.append(f"**Control** — greedy gap on TSP-50 when trained on TSP-50: {detail}.")
        rel = "; ".join(f"{n}: {r50:.0f}% on TSP-50 (trained on 50) vs. {r20:.0f}% on TSP-20 (trained on 20)"
                        for n, *_, r50, r20 in lines)
        points.append(f"**Control** — reduction of the classical-greedy gap (no 2-opt): {rel}.")
        close = sum(r50 >= 0.75 * r20 for *_, r50, r20 in lines)
        if close == len(lines):
            verdict = ("Trained in-distribution, every model recovers most (≥ 75%) of its TSP-20 advantage over "
                       "classical greedy on TSP-50, so the phase 1 failure is mainly a **generalization** problem "
                       "(train/test size mismatch), not a lack of capacity at n = 50.")
        elif close == 0:
            verdict = ("Even trained on TSP-50, no model recovers its TSP-20 advantage over classical greedy, which "
                       "points to a **capacity/training** limit at n = 50 (H=64, L=6, 20 epochs), not only to "
                       "generalization.")
        else:
            verdict = (f"{close} of {len(lines)} models recover ≥ 75% of their TSP-20 advantage when trained on "
                       f"TSP-50: the failure is a generalization problem for those, capacity for the others.")
        points.append(f"**Control — interpretation**: {verdict}")
    return points[:10]


# ---------------------------------------------------------------- main

def build_report(results):
    baselines = load_baselines(results)
    configs = {c: load_config(results, folders) for c, (_, folders) in CONFIGS.items()}
    n_tests = len(all_comparisons(configs))
    threshold = ALPHA / max(n_tests, 1)
    comparisons = run_comparisons(configs, threshold)
    missing = [f"{c} / {MODELS[m]} / {DATASETS[d]}" for c in CONFIGS for m in MODELS
               for d in expected_datasets(c) if d not in configs[c][m]]

    legend = "\n".join(f"- `{c}` — {desc}" for c, (desc, _) in CONFIGS.items())
    return "\n".join([
        "# Phase 2, batch 2 — graph size, normalization layer and the TSP-50 control",
        "",
        f"Generated by `scripts/phase2_batch2_report.py` on {datetime.date.today().isoformat()}.",
        "All models: H=64, L=6, 12000 training instances, 20 epochs, seeds 0, 1, 2. "
        "Unless stated otherwise they are trained on TSP-20; TSP-50 and TSP-100 then measure generalization.",
        "",
        "Configurations:",
        "",
        legend,
        "",
        "## Summary",
        "",
        "\n".join(f"{i}. {p}" for i, p in enumerate(summary_points(configs, comparisons, baselines), 1)),
        "",
        "## Table 1 — mean gap to LKH (%)",
        "",
        "Models: mean ± std **across the 3 seeds**. Baselines: deterministic. \"—\" = not part of this "
        "configuration; \"missing\" = expected but not found (failed step).",
        "",
        table_gaps(baselines, configs),
        "",
        "## Table 2 — paired per-instance comparisons against phase 1",
        "",
        f"For each seed s, the gap of each instance under the configuration is compared with phase 1 (seed s); "
        f"the table shows means across the 3 seed pairs. *Equal* = difference ≤ {TIE_TOL:g} pp. "
        f"Diff. = configuration − phase 1 (negative = the change helps). Two-sided Wilcoxon test (ties dropped); "
        f"*max p* is the largest p-value across seeds. Significant only if max p < {threshold:.2e} "
        f"(Bonferroni: {ALPHA}/{n_tests} tests) and the median has the same sign in every seed.",
        "",
        table_paired(comparisons),
        "",
        "## Decisions",
        "",
        "Choices made without asking (the most conservative option each time):",
        "",
        "\n".join(f"{i}. {d}" for i, d in enumerate(DECISIONS, 1)),
        "",
        "## Missing results",
        "",
        ("\n".join(f"- {x}" for x in missing) if missing else "None: every configuration has all 3 seeds."),
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description="Phase 2, batch 2 report.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/phase2_batch2_report.md")
    args = parser.parse_args()
    report = build_report(args.results)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
