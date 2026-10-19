"""Phase 2, step 5 report: gap as a function of the number of training instances (TSP-20 training).

Training sizes (phase 1 configuration otherwise: BatchNorm, k=10, raw distances, H=64, L=6, 20 epochs):
    12k    phase 1 models (data/tsp20_train.npz, seeds 0, 1, 2)
    25k    first 25000 instances of data/tsp20_train100k.npz (seed 0)
    50k    first 50000 instances of data/tsp20_train100k.npz (seed 0)
    100k   all of data/tsp20_train100k.npz (seeds 0, 1, 2)
Each model is evaluated on test graphs with k = 10 (as in training) and with k = 0.5 n.
Missing results (e.g. a failed step) are reported as "missing" instead of stopping the report.

Writes the report (Table 1: seed 0 at every size; Table 2: mean ± std across seeds at 12k and 100k;
Table 3: paired Wilcoxon tests 100k vs. 12k with Bonferroni; selected epochs; Decisions) and the figure
figs/phase2_datasize_gap.{png,pdf} next to it.
Example:
    python scripts/phase2_datasize_report.py
"""

import argparse
import datetime
import importlib.util
import os

import numpy as np
from matplotlib.lines import Line2D

# scripts/ is not a package: load the phase 1 and batch 2 report scripts directly to reuse their helpers.
_here = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarize = _load("summarize")
batch2 = _load("phase2_batch2_report")
plt = summarize.plt
MODELS, ALPHA, TIE_TOL = summarize.MODELS, summarize.ALPHA, summarize.TIE_TOL
MODEL_COLORS, INK, MUTED = summarize.MODEL_COLORS, summarize.INK, summarize.MUTED
DATASETS, METHODS = batch2.DATASETS, batch2.METHODS
load_folder, load_baselines, mean_std = batch2.load_folder, batch2.load_baselines, summarize.mean_std

SIZES = {12000: "12k", 25000: "25k", 50000: "50k", 100000: "100k"}
SEEDS = {12000: (0, 1, 2), 25000: (0,), 50000: (0,), 100000: (0, 1, 2)}
GRAPHS = {"k10": "k = 10", "kfrac": "k = 0.5 n"}
OUT = "phase2_datasize"  # folder (inside results/) for the k = 0.5 n evaluations of this step


def run_name(m, n, s):
    return f"{m}_data{SIZES[n]}_seed{s}"


def folders(m, n, s, graph):
    """Evaluation folders (relative to results/) of one trained model on one kind of test graph."""
    if n == 12000:
        k10 = [f"runs/{m}_seed{s}", f"phase2_exp1/phase1_tsp100/{m}_seed{s}"]
        # On TSP-20, k = 0.5 × 20 = 10 is the phase 1 graph: the k = 10 folder supplies that column.
        return k10 if graph == "k10" else [k10[0], f"phase2_batch2/kfrac05/{m}_seed{s}"]
    r = run_name(m, n, s)
    return [f"runs/{r}"] if graph == "k10" else [f"{OUT}/kfrac05/{r}"]


DECISIONS = [
    "`data/tsp20_train100k.npz`: 100000 TSP-20 instances, seed 8, LKH with the phase 1 settings. The 25k and 50k "
    "sets are its first 25000 and 50000 instances (`--num-train`), so 25k ⊂ 50k ⊂ 100k. The 12k point is phase 1, "
    "trained on a *different* file (`tsp20_train.npz`, seed 1): going from 12k to 25k also changes the instances, "
    "not only their number.",
    "Validation (`tsp20_val`, used for model selection) and test sets are the phase 1 ones; nothing else changes "
    "(BatchNorm, k = 10, raw distances, batch size 64, lr 1e-3 with cosine schedule over 20 epochs).",
    "Run names are `<model>_data<size>_seed<s>` (not `train50k`, to avoid confusion with the TSP-50 control "
    "`<model>_train50_seed<s>`).",
    "k = 0.5 n evaluations include TSP-20 for the new models (k = 10 there, so they must equal the k = 10 values; "
    "a check). For the phase 1 models the TSP-20 k = 0.5 n column reuses the k = 10 evaluation, which is the same "
    "graph; their TSP-50/100 k = 0.5 n values are those of batch 2 (`results/phase2_batch2/kfrac05/`), not "
    "recomputed. k = 0.5 n evaluations use 64 graphs per forward pass (as in batch 2).",
    "Table 1 and the figure use **seed 0** at every size, because 25k and 50k only have seed 0; Table 2 gives "
    "mean ± std across the 3 seeds where they exist (12k and 100k). Statistical tests (Table 3) compare only 100k "
    "with 12k, pairing seed s with seed s, as in batch 2; single-seed tests at 25k/50k were not run because they "
    "cannot separate the effect of the data from seed-to-seed variation.",
    "The Bonferroni correction is applied over all tests of Table 3 (both kinds of test graph together).",
    "Conservative reading of the rules: nothing from phase 1, experiment 1 or batch 2 was rewritten (new folders "
    "only); a failed step is logged and the steps that depend on it are skipped; the report is copied to `docs/` "
    "but not committed.",
]


# ---------------------------------------------------------------- loading

def load_all(results):
    """{(graph, model, size, seed): {"eval": {dataset: {method: gap}}, "per_instance": {...}, "epoch": int}}."""
    out = {}
    for graph in GRAPHS:
        for m in MODELS:
            for n in SIZES:
                for s in SEEDS[n]:
                    ev_all, pi_all, epoch = {}, {}, None
                    for folder in folders(m, n, s, graph):
                        loaded = load_folder(os.path.join(results, folder))
                        if loaded:
                            ev, pi = loaded
                            ev_all.update({d: {x: ev[d][x]["gap_mean"] for x in METHODS} for d in DATASETS if d in ev})
                            pi_all.update(pi)
                            epoch = epoch or ev["meta"].get("best_epoch")
                    out[graph, m, n, s] = {"eval": ev_all, "per_instance": pi_all, "epoch": epoch}
    return out


def gap(data, graph, m, n, s, d, meth):
    ev = data[graph, m, n, s]["eval"]
    return ev[d][meth] if d in ev else None


# ---------------------------------------------------------------- tables

def fmt(x):
    return "missing" if x is None else f"{x:.2f}"


def table_seed0(data, baselines, graph):
    cols = [(graph, d, x) for d in DATASETS for x in METHODS]
    lines = ["| Model | Train | " + " | ".join(f"{DATASETS[d]} {METHODS[x]}" for _, d, x in cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for key, name in summarize.BASELINES.items():
        cells = [f"{baselines[d][key if x == 'greedy' else key + '+2opt']['gap_mean']:.2f}" if d in baselines
                 else "missing" for _, d, x in cols]
        lines.append(f"| {name} | — | " + " | ".join(cells) + " |")
    for m, mname in MODELS.items():
        for n, size in SIZES.items():
            lines.append(f"| {mname} | {size} | " + " | ".join(fmt(gap(data, g, m, n, 0, d, x)) for g, d, x in cols) + " |")
    return "\n".join(lines)


def table_seeds(data, graph):
    cols = [(graph, d, x) for d in DATASETS for x in METHODS]
    lines = ["| Model | Train | " + " | ".join(f"{DATASETS[d]} {METHODS[x]}" for _, d, x in cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for m, mname in MODELS.items():
        for n in (12000, 100000):
            cells = []
            for g, d, x in cols:
                vals = [gap(data, g, m, n, s, d, x) for s in SEEDS[n]]
                cells.append("missing" if None in vals else mean_std(vals))
            lines.append(f"| {mname} | {SIZES[n]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_epochs(data):
    lines = ["| Model | " + " | ".join(f"{SIZES[n]} seed {s}" for n in SIZES for s in SEEDS[n]) + " |",
             "|" + "---|" * (1 + sum(len(v) for v in SEEDS.values()))]
    for m, mname in MODELS.items():
        cells = [str(data["k10", m, n, s]["epoch"] or "missing") for n in SIZES for s in SEEDS[n]]
        lines.append(f"| {mname} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def all_tests(data):
    """(graph, model, dataset, method) where every seed has both the 100k and the 12k per-instance gaps."""
    tests = []
    for g in GRAPHS:
        for m in MODELS:
            for d in DATASETS:
                for x in METHODS:
                    key = f"{d}__{x}"
                    if all(key in data[g, m, n, s]["per_instance"] for n in (12000, 100000) for s in SEEDS[n]):
                        tests.append((g, m, d, x))
    return tests


def compare(data, g, m, d, x, threshold):
    key = f"{d}__{x}"
    res = [summarize.paired_compare(data[g, m, 100000, s]["per_instance"][key],
                                    data[g, m, 12000, s]["per_instance"][key]) for s in SEEDS[100000]]
    avg = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    p_max = max(r["p"] for r in res)
    same_sign = len({np.sign(r["median_diff"]) for r in res}) == 1
    direction = avg["median_diff"] if avg["median_diff"] != 0 else avg["mean_diff"]
    verdict = ("better" if direction < 0 else "worse") if p_max < threshold and same_sign else "n.s."
    return {**avg, "p_max": p_max, "verdict": verdict}


def table_paired(comparisons):
    lines = ["| Test graph | Model | Dataset | Method | Better | Equal | Worse | Mean diff. (pp) "
             "| Median diff. (pp) | max p (seeds) | Conclusion |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (g, m, d, x), r in comparisons.items():
        verdict = f"**{r['verdict']}**" if r["verdict"] != "n.s." else "n.s."
        lines.append(f"| {GRAPHS[g]} | {MODELS[m]} | {DATASETS[d]} | {METHODS[x]} | {r['better']:.1f}% "
                     f"| {r['equal']:.1f}% | {r['worse']:.1f}% | {r['mean_diff']:+.2f} | {r['median_diff']:+.2f} "
                     f"| {r['p_max']:.1e} | {verdict} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- figure

def plot(data, baselines, figs_dir):
    """Gap vs. training instances (seed 0, log x axis): TSP-20 / TSP-50 × greedy / + 2-opt.

    Solid lines: test graphs with k = 10; dashed: k = 0.5 n (TSP-50 only: on TSP-20 they are the same graph).
    Vertical bars at 12k and 100k: min–max across the 3 seeds. Grey line: classical greedy (+ 2-opt).
    """
    sizes = list(SIZES)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for row, d in enumerate(("tsp20_test", "tsp50_test")):
        for col, x in enumerate(METHODS):
            ax = axes[row, col]
            for m, mname in MODELS.items():
                for g, style in (("k10", "-"), ("kfrac", "--")):
                    if g == "kfrac" and d == "tsp20_test":
                        continue
                    pts = [(n, gap(data, g, m, n, 0, d, x)) for n in sizes]
                    pts = [(n, v) for n, v in pts if v is not None]
                    if pts:
                        ax.plot(*zip(*pts), style, marker="o", color=MODEL_COLORS[m], linewidth=1.8, markersize=4)
                    for n in (12000, 100000):
                        vals = [gap(data, g, m, n, s, d, x) for s in SEEDS[n]]
                        if None not in vals:
                            ax.vlines(n, min(vals), max(vals), color=MODEL_COLORS[m], linewidth=1, alpha=0.6)
            if d in baselines:
                key = "greedy" if x == "greedy" else "greedy+2opt"
                ax.axhline(baselines[d][key]["gap_mean"], color=MUTED, linewidth=1, linestyle=":")
            summarize.style_axes(ax)
            ax.set_xscale("log")
            ax.set_xticks(sizes, [SIZES[n] for n in sizes])
            ax.minorticks_off()
            ax.set_title(f"{DATASETS[d]}, {METHODS[x]}", color=INK, fontsize=10, loc="left")
            if col == 0:
                ax.set_ylabel("Test gap to LKH (%)", color=INK)
            if row == 1:
                ax.set_xlabel("Training instances (TSP-20)", color=INK)
    fig.suptitle("Gap vs. number of training instances (seed 0; bars = min–max over 3 seeds)", color=INK,
                 fontsize=11, x=0.02, ha="left")
    # One legend for the whole figure: colour = model, line style = test graph.
    handles = [Line2D([], [], color=MODEL_COLORS[m], linewidth=1.8, marker="o", markersize=4, label=name)
               for m, name in MODELS.items()]
    handles += [Line2D([], [], color=MUTED, linewidth=1.8, linestyle=st, label=f"test graph {GRAPHS[g]}")
                for g, st in (("k10", "-"), ("kfrac", "--"))]
    handles.append(Line2D([], [], color=MUTED, linewidth=1, linestyle=":", label="classical greedy (same method)"))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    os.makedirs(figs_dir, exist_ok=True)
    summarize.save(fig, figs_dir, "phase2_datasize_gap")


# ---------------------------------------------------------------- summary

def summary_points(data, comparisons):
    points = []
    for m, mname in MODELS.items():
        parts = []
        for d in ("tsp20_test", "tsp50_test"):
            series = [gap(data, "k10", m, n, 0, d, "greedy") for n in SIZES]
            if None not in (series[0], series[-1]):
                parts.append(f"{DATASETS[d]} " + " → ".join(fmt(v) for v in series))
        points.append(f"**{mname}**, greedy gap (seed 0, k = 10), 12k → 25k → 50k → 100k: "
                      + ("; ".join(parts) if parts else "no results") + ".")
    if comparisons:
        for g in GRAPHS:
            rows = {t: r for t, r in comparisons.items() if t[0] == g}
            parts = []
            for d in DATASETS:
                sub = [r for t, r in rows.items() if t[2] == d]
                if sub:
                    parts.append(f"{DATASETS[d]}: {sum(r['verdict'] == 'better' for r in sub)} better / "
                                 f"{sum(r['verdict'] == 'worse' for r in sub)} worse of {len(sub)} "
                                 f"(mean diff. {np.mean([r['mean_diff'] for r in sub]):+.2f} pp)")
            if parts:
                points.append(f"**100k vs. 12k, test graphs {GRAPHS[g]}** (3 seeds, Table 3): " + "; ".join(parts) + ".")
    else:
        points.append("**100k vs. 12k**: no paired tests (100k runs missing, see the log).")
    return points


NOTE = ("**More data also means more training.** All runs use 20 epochs with batch size 64, so the number of "
        "gradient steps grows with the data: 12k → 3 760 steps, 25k → 7 820, 50k → 15 640, 100k → 31 260. "
        "\"More data\" and \"more training\" are therefore not fully separated here: an improvement can come "
        "from either (a fixed-step control, e.g. 12k for ≈ 166 epochs, would be needed to separate them).")


def build_report(results):
    data = load_all(results)
    baselines = load_baselines(results)
    tests = all_tests(data)
    threshold = ALPHA / max(len(tests), 1)
    comparisons = {t: compare(data, *t, threshold) for t in tests}
    missing = [f"{run_name(m, n, s) if n != 12000 else f'{m}_seed{s} (phase 1)'} / {GRAPHS[g]} / {DATASETS[d]}"
               for g in GRAPHS for m in MODELS for n in SIZES for s in SEEDS[n] for d in DATASETS
               if d not in data[g, m, n, s]["eval"]]
    return "\n".join([
        "# Phase 2, step 5 — more training data",
        "",
        f"Generated by `scripts/phase2_datasize_report.py` on {datetime.date.today().isoformat()}.",
        "Three models with the phase 1 configuration (BatchNorm, k = 10, raw distances, H=64, L=6, 20 epochs), "
        "trained on TSP-20 with 12k (phase 1), 25k, 50k and 100k instances; evaluated on TSP-20/50/100 with test "
        "graphs k = 10 and k = 0.5 n. Seeds: 0, 1, 2 at 12k and 100k; seed 0 only at 25k and 50k.",
        "",
        NOTE,
        "",
        "## Summary",
        "",
        "\n".join(f"{i}. {p}" for i, p in enumerate(summary_points(data, comparisons), 1)),
        "",
        "## Figure",
        "",
        "![Gap vs. training instances](figs/phase2_datasize_gap.png)",
        "",
        "Solid: test graphs k = 10; dashed: k = 0.5 n (TSP-50; identical to k = 10 on TSP-20). Points: seed 0. "
        "Vertical bars at 12k and 100k: min–max over the 3 seeds. Dotted grey: classical greedy (+ 2-opt).",
        "",
        "## Table 1 — mean gap to LKH (%), seed 0",
        "",
        "\"missing\" = expected but not found (failed or not yet run).",
        "",
        "### Test graphs k = 10",
        "",
        table_seed0(data, baselines, "k10"),
        "",
        "### Test graphs k = 0.5 n",
        "",
        table_seed0(data, baselines, "kfrac"),
        "",
        "## Table 2 — mean ± std across seeds 0, 1, 2 (12k and 100k)",
        "",
        "### Test graphs k = 10",
        "",
        table_seeds(data, "k10"),
        "",
        "### Test graphs k = 0.5 n",
        "",
        table_seeds(data, "kfrac"),
        "",
        "## Table 3 — paired per-instance comparisons, 100k vs. 12k",
        "",
        f"For each seed s, the gap of each test instance with 100k training instances is compared with phase 1 "
        f"(12k, seed s); the table shows means across the 3 seed pairs. *Equal* = difference ≤ {TIE_TOL:g} pp. "
        f"Diff. = 100k − 12k (negative = more data helps). Two-sided Wilcoxon test (ties dropped); *max p* is the "
        f"largest p-value across seeds. Significant only if max p < {threshold:.2e} (Bonferroni: {ALPHA}/"
        f"{len(tests)} tests) and the median has the same sign in every seed.",
        "",
        table_paired(comparisons) if comparisons else "No results yet.",
        "",
        "## Selected epochs",
        "",
        "Epoch with the best TSP-20 validation gap (greedy), which is the checkpoint evaluated above.",
        "",
        table_epochs(data),
        "",
        "## Decisions",
        "",
        "Choices made without asking (the most conservative option each time):",
        "",
        "\n".join(f"{i}. {d}" for i, d in enumerate(DECISIONS, 1)),
        "",
        "## Missing results",
        "",
        ("\n".join(f"- {x}" for x in missing) if missing else "None."),
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description="Phase 2, step 5 report (training-set size).")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/phase2_datasize_report.md")
    args = parser.parse_args()
    report = build_report(args.results)
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    data = load_all(args.results)
    plot(data, load_baselines(args.results), os.path.join(out_dir, "figs"))
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"saved: {args.out} and {os.path.join(out_dir, 'figs', 'phase2_datasize_gap.png')}")


if __name__ == "__main__":
    main()
