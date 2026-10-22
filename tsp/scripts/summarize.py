"""Gather the results of the baselines and of all runs into one report.

Reads results/baselines.json (+ _per_instance.npz) and results/runs/{gcn,gat,gated}_seed*/
(eval.json, eval_per_instance.npz, history.json) and produces:
    results/table.md                       tables (gap ± std across seeds; paired comparisons)
    results/figs/val_curves.{png,pdf}      validation gap per epoch, one line per model
    results/figs/gaps_by_method.{png,pdf}  gaps per method on TSP-20 and TSP-50, with error bars
Example:
    python scripts/summarize.py
"""

import argparse
import datetime
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")  # no window: only save files
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

MODELS = {"gcn": "GCN", "gat": "GATv2", "gated": "Gated GCN"}
BASELINES = {"greedy": "Classical greedy", "christofides": "Christofides"}
DATASETS = {"tsp20_test": "TSP-20", "tsp50_test": "TSP-50"}
TIE_TOL = 1e-6   # gap difference (% points) below which two instances count as a tie
ALPHA = 0.05

# Colours (reference palette, validated for colour blindness). Each colour has a single meaning:
MODEL_COLORS = {"gcn": "#2a78d6", "gat": "#eb6834", "gated": "#1baf7a"}   # curves figure
VARIANT_COLORS = {"greedy": "#6250d6", "greedy+2opt": "#eda100"}          # bars figure
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# The two paired comparisons: (model method, baseline method, description)
COMPARISONS = [("greedy+2opt", "greedy+2opt", "model + 2-opt vs. greedy + 2-opt"),
               ("greedy", "greedy", "model without 2-opt vs. classical greedy without 2-opt")]


# ---------------------------------------------------------------- loading

def load_runs(runs_dir):
    """Runs {model: [run, ...]} sorted by seed. Ignores folders such as quick_*."""
    runs = {m: [] for m in MODELS}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        match = re.fullmatch(r"(gcn|gat|gated)_seed(\d+)", os.path.basename(path))
        files = ("eval.json", "history.json", "eval_per_instance.npz")
        if not match or not all(os.path.exists(os.path.join(path, f)) for f in files):
            continue  # another kind of name (e.g. quick_gcn) or a run still in progress
        with open(os.path.join(path, "eval.json")) as f:
            ev = json.load(f)
        with open(os.path.join(path, "history.json")) as f:
            history = json.load(f)
        with np.load(os.path.join(path, "eval_per_instance.npz")) as f:
            per_instance = {k: f[k] for k in f.files}
        runs[match.group(1)].append({"seed": int(match.group(2)), "eval": ev,
                                     "history": history, "per_instance": per_instance})
    for m in runs:
        runs[m].sort(key=lambda r: r["seed"])
    return runs


def check_consistency(runs):
    """Warn if the runs of a model differ in anything other than the seed. Returns the commits used."""
    commits = set()
    for m, rs in runs.items():
        configs = [{k: v for k, v in r["eval"]["meta"]["config"].items()
                    if k not in ("seed", "pos_weight")} for r in rs]
        if any(c != configs[0] for c in configs):
            print(f"WARNING: the runs of {m} have different configurations: {configs}")
        for r in rs:
            meta = r["eval"]["meta"]
            commits.add(meta["train_git_commit"] + (" (uncommitted changes)" if meta["git_dirty"] else ""))
    if len(commits) > 1:
        print(f"WARNING: the runs come from different commits: {sorted(commits)}")
    return sorted(commits)


# ---------------------------------------------------------------- statistics

def mean_std(values):
    """'mean ± std' across seeds (only the mean if there is a single seed)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return f"{values.mean():.2f}"
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def paired_compare(model_gaps, base_gaps, tol=TIE_TOL):
    """Compare two methods instance by instance (gaps in %; lower is better).

    Returns the percentages of instances where the model is better/equal/worse, the mean and
    median of the difference (model - baseline) and the p-value of the two-sided Wilcoxon test
    (exact ties are dropped by the test; if everything is a tie, p = 1).
    """
    diff = np.asarray(model_gaps) - np.asarray(base_gaps)
    nonzero = np.abs(diff) > tol
    p = wilcoxon(diff[nonzero]).pvalue if nonzero.sum() > 0 else 1.0
    return {"better": 100 * np.mean(diff < -tol), "equal": 100 * np.mean(~nonzero),
            "worse": 100 * np.mean(diff > tol), "mean_diff": float(diff.mean()),
            "median_diff": float(np.median(diff)), "p": float(p)}


# ---------------------------------------------------------------- tables

def table_gaps(baselines, runs):
    """Table 1: mean gap ± std across seeds."""
    cols = [(d, m) for d in DATASETS for m in ("greedy", "greedy+2opt")]
    head = "| Method | " + " | ".join(f"{DATASETS[d]} {'greedy' if m == 'greedy' else '+ 2-opt'}"
                                      for d, m in cols) + " | Seeds | Selected epochs |"
    lines = [head, "|" + "---|" * (len(cols) + 3)]
    for key, name in BASELINES.items():
        cells = [f"{baselines[d][key if m == 'greedy' else key + '+2opt']['gap_mean']:.2f}" for d, m in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " | — (deterministic) | — |")
    for key, name in MODELS.items():
        rs = runs[key]
        if not rs:
            continue
        cells = [mean_std([r["eval"][d][m]["gap_mean"] for r in rs]) for d, m in cols]
        seeds = ", ".join(str(r["seed"]) for r in rs)
        epochs = ", ".join(str(r["eval"]["meta"]["best_epoch"]) for r in rs)
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {seeds} | {epochs} |")
    return "\n".join(lines)


def table_paired(base_per_instance, runs, threshold):
    """Table 2: paired per-instance comparisons (one per seed, then aggregated)."""
    lines = ["| Comparison | Dataset | Model | Better | Equal | Worse | Mean diff. (pp) "
             "| Median diff. (pp) | max p (seeds) | Conclusion |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for model_method, base_method, desc in COMPARISONS:
        for d in DATASETS:
            base = base_per_instance[f"{d}__{base_method}"]
            for key, name in MODELS.items():
                if not runs[key]:
                    continue
                res = [paired_compare(r["per_instance"][f"{d}__{model_method}"], base) for r in runs[key]]
                avg = {k: np.mean([x[k] for x in res]) for k in res[0]}
                p_max = max(x["p"] for x in res)
                same_sign = len({np.sign(x["median_diff"]) for x in res}) == 1
                # Direction from the median; if the median is 0 (many ties), from the mean.
                direction = avg["median_diff"] if avg["median_diff"] != 0 else avg["mean_diff"]
                if p_max < threshold and same_sign:
                    verdict = "**better**" if direction < 0 else "**worse**"
                else:
                    verdict = "n.s."  # not significant
                lines.append(f"| {desc} | {DATASETS[d]} | {name} | {avg['better']:.1f}% | {avg['equal']:.1f}% "
                             f"| {avg['worse']:.1f}% | {avg['mean_diff']:+.2f} | {avg['median_diff']:+.2f} "
                             f"| {p_max:.1e} | {verdict} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- figures

def style_axes(ax):
    """Discreet axes: no box at the top/right, light horizontal grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, figs_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figs_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_curves(runs, figs_dir):
    """Validation gap per epoch: mean over seeds (line), min–max (band), selected epochs (dots)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ends = []
    for key, name in MODELS.items():
        rs = runs[key]
        if not rs:
            continue
        gaps = np.array([[h["val_gap_greedy"] for h in r["history"]] for r in rs])  # (seeds, epochs)
        epochs = np.arange(1, gaps.shape[1] + 1)
        color = MODEL_COLORS[key]
        ax.fill_between(epochs, gaps.min(0), gaps.max(0), color=color, alpha=0.15, linewidth=0)
        ax.plot(epochs, gaps.mean(0), color=color, linewidth=2, label=f"{name} (n={len(rs)})")
        best = [(r["eval"]["meta"]["best_epoch"], r["eval"]["meta"]["best_val_gap"]) for r in rs]
        ax.scatter(*zip(*best), s=36, color=color, edgecolors="white", linewidths=1.5, zorder=3)
        ends.append([gaps.mean(0)[-1], name])

    # Names at the end of the lines (dark ink text), pushed apart if they are too close.
    ends.sort()
    min_sep = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + min_sep)
    for y, name in ends:
        ax.text(epochs[-1] + 0.4, y, name, va="center", color=INK, fontsize=9)

    style_axes(ax)
    ax.set_xlim(0.5, epochs[-1] + 2.8)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Epoch", color=INK)
    ax.set_ylabel("Validation gap, greedy without 2-opt (%)", color=INK)
    ax.set_title("Training curves (TSP-20): mean over seeds, band = min–max, ● = selected epoch",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(0.88, 1.0))
    save(fig, figs_dir, "val_curves")


def plot_bars(baselines, runs, figs_dir):
    """Mean gap per method, greedy vs. + 2-opt, on TSP-20 and TSP-50 (shared y axis)."""
    methods = [(k, n, None) for k, n in BASELINES.items()] + [(k, n, runs[k]) for k, n in MODELS.items() if runs[k]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    width = 0.38
    x = np.arange(len(methods))
    for ax, d in zip(axes, DATASETS):
        for v_idx, variant in enumerate(("greedy", "greedy+2opt")):
            means, errs = [], []
            for key, _, rs in methods:
                if rs is None:  # baseline: deterministic, no error bar
                    means.append(baselines[d][key if variant == "greedy" else key + "+2opt"]["gap_mean"])
                    errs.append(0.0)
                else:
                    vals = [r["eval"][d][variant]["gap_mean"] for r in rs]
                    means.append(np.mean(vals))
                    errs.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            offset = (v_idx - 0.5) * width
            ax.bar(x + offset, means, width, color=VARIANT_COLORS[variant], edgecolor="white", linewidth=2,
                   label="Greedy" if variant == "greedy" else "Greedy + 2-opt")
            # Error bars only where there is variation across seeds (the baselines have none).
            has_err = np.array(errs) > 0
            ax.errorbar((x + offset)[has_err], np.array(means)[has_err], yerr=np.array(errs)[has_err],
                        fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        style_axes(ax)
        ax.set_xticks(x, [n.replace(" ", "\n", 1) for _, n, _ in methods], fontsize=9)  # names on 2 lines
        ax.set_title(DATASETS[d], color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("Gap to LKH (%)", color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")  # TSP-20 has free space at the top
    fig.suptitle("Mean gap per method (test). Error bars: std across seeds; "
                 "baselines are deterministic", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, figs_dir, "gaps_by_method")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Report: tables and figures of the results.")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    with open(os.path.join(args.results, "baselines.json")) as f:
        baselines = json.load(f)
    with np.load(os.path.join(args.results, "baselines_per_instance.npz")) as f:
        base_per_instance = {k: f[k] for k in f.files}
    runs = load_runs(os.path.join(args.results, "runs"))
    commits = check_consistency(runs)

    n_tests = len(COMPARISONS) * len(DATASETS) * sum(1 for m in MODELS if runs[m])
    threshold = ALPHA / n_tests  # Bonferroni correction over all the tests

    report = "\n".join([
        "# Phase 1 results",
        "",
        f"Generated by `scripts/summarize.py` on {datetime.date.today().isoformat()}. "
        f"Training commits: {', '.join(commits)}; baselines: {baselines['meta']['git_commit']}.",
        "",
        "## Table 1 — mean gap to LKH (%)",
        "",
        "Models: mean ± std **across seeds** (run-to-run variation). "
        "Baselines: deterministic. Trained on TSP-20; TSP-50 measures generalization.",
        "",
        table_gaps(baselines, runs),
        "",
        "## Table 2 — paired per-instance comparisons",
        "",
        f"For each seed, the gap of each instance is compared with the baseline's; the table shows the means "
        f"across seeds. *Equal* = difference ≤ {TIE_TOL:g} pp. Diff. = model − baseline (negative = model better). "
        f"Two-sided Wilcoxon test (ties dropped); *max p* is the largest p-value across seeds. "
        f"A conclusion is significant only if max p < {threshold:.2e} (Bonferroni: {ALPHA}/{n_tests} tests) "
        f"and the sign of the median is the same in every seed.",
        "",
        table_paired(base_per_instance, runs, threshold),
        "",
        "## Figures",
        "",
        "- `figs/val_curves.png` — validation gap per epoch",
        "- `figs/gaps_by_method.png` — gap per method on TSP-20 and TSP-50",
        "",
    ])
    with open(os.path.join(args.results, "table.md"), "w") as f:
        f.write(report)

    figs_dir = os.path.join(args.results, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    plot_curves(runs, figs_dir)
    plot_bars(baselines, runs, figs_dir)

    print(report)
    print(f"runs: { {m: [r['seed'] for r in rs] for m, rs in runs.items()} }")
    print(f"saved: {args.results}/table.md and {figs_dir}/val_curves.(png|pdf), gaps_by_method.(png|pdf)")


if __name__ == "__main__":
    main()
