"""Phase 2 checks report (points 1 to 4).

    1. LayerNorm retrained with three checkpoints per run: best validation gap (best.pt, as before), best
       validation loss (best_loss.pt) and last epoch (last.pt); each evaluated with k = 10 and k = 0.5 n,
       plus the validation curves, to see why the best validation gap comes so early.
    2. Control on TSP-50 trained with k = 0.5 n (train50_kfrac).
    3. TSP-20 results for every configuration (trade-off between the training size and generalization).
    4. Paired Wilcoxon tests of every configuration + 2-opt against classical greedy + 2-opt and
       Christofides + 2-opt on TSP-50 and TSP-100.
Missing results (e.g. a failed step) are reported as "missing" instead of stopping the report.

Writes the report and the figure figs/phase2_checks_ln_curves.{png,pdf} next to it.
Example:
    python scripts/phase2_checks_report.py
"""

import argparse
import datetime
import importlib.util
import json
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
DATASETS, METHODS, SEEDS = batch2.DATASETS, batch2.METHODS, batch2.SEEDS
B2, CK = batch2.B2, "phase2_checks"  # folders (inside results/) of batch 2 and of these checks
SELECTIONS = {"gap": "best.pt", "loss": "best_loss.pt", "last": "last.pt"}


def ln_run(m, s):
    return f"{m}_layernorm_ckpt_seed{s}"


def ln_folder(m, s, sel, graph):
    """Evaluation folder of one LayerNorm checkpoint (best.pt with k = 10 is the training folder itself)."""
    if sel == "gap" and graph == "k10":
        return f"runs/{ln_run(m, s)}"
    return f"{CK}/ln/{sel}_{graph}/{ln_run(m, s)}"


# name -> (description, function(model, seed) -> folders relative to results/). When several folders hold the
# same dataset, the later one wins: for the k = 0.5 n configurations the k = 10 folder comes first and only
# supplies TSP-20, where k = 0.5 × 20 = 10 is the same graph.
CONFIGS = {
    "phase1": batch2.CONFIGS["phase1"],
    "knn_mean": batch2.CONFIGS["knn_mean"],
    "kfrac": ("Phase 1 models, test graphs k = 0.5 n",
              lambda m, s: [f"runs/{m}_seed{s}", f"{B2}/kfrac05/{m}_seed{s}"]),
    "knn_mean_kfrac": ("knn_mean models, test graphs k = 0.5 n",
                       lambda m, s: [f"runs/{m}_knnmean_seed{s}", f"{B2}/kfrac05/{m}_knnmean_seed{s}"]),
    "bn_batch": batch2.CONFIGS["bn_batch"],
    "layernorm": ("Exp. 3 LayerNorm (batch 2 runs, best val gap), k = 10", batch2.CONFIGS["layernorm"][1]),
    "layernorm_kfrac": ("Exp. 3 LayerNorm (batch 2 runs, best val gap), k = 0.5 n",
                        batch2.CONFIGS["layernorm_kfrac"][1]),
    **{f"ln_{sel}{suffix}": (f"Check 1: LayerNorm rerun, checkpoint {file}, {gname}",
                             (lambda sel, g: lambda m, s: [ln_folder(m, s, sel, g)])(sel, g))
       for sel, file in SELECTIONS.items()
       for g, suffix, gname in (("k10", "", "k = 10"), ("kfrac05", "_kfrac", "k = 0.5 n"))},
    "train50": ("Control: trained on TSP-50, k = 10",
                lambda m, s: [f"runs/{m}_train50_seed{s}", f"{CK}/train50_tsp20/{m}_train50_seed{s}"]),
    "train50_kfrac": ("Check 2: trained on TSP-50 with k = 0.5 n (train and test graphs)",
                      lambda m, s: [f"runs/{m}_train50kfrac_seed{s}"]),
}
NEW = [c for c in CONFIGS if c.startswith("ln_")] + ["train50_kfrac"]
# Direct paired comparisons (configuration, reference) for checks 1 and 2.
DIRECT = ([(f"ln_{sel}{suf}", f"ln_gap{suf}") for suf in ("", "_kfrac") for sel in ("loss", "last")]
          + [("ln_gap", "layernorm"), ("ln_gap_kfrac", "layernorm_kfrac"), ("train50_kfrac", "train50")]
          + [(c, "phase1") for c in NEW])
BASE_REFS = {"greedy+2opt": "Classical greedy + 2-opt", "christofides+2opt": "Christofides + 2-opt"}

DECISIONS = [
    "Check 1 retrains the 9 LayerNorm models (same command as experiment 3 plus `--extra-checkpoints`) into new "
    "folders `results/runs/<model>_layernorm_ckpt_seed<s>`; the experiment 3 runs are untouched. Training on the "
    "GPU is **not bit-reproducible** (PyG scatter operations use atomics: two identical runs differ by ~2e-3 in "
    "the weights after 3 epochs, checked before launching), so the rerun's best-val-gap checkpoint (`ln_gap`) is "
    "a new sample, not a copy of experiment 3; Table 3 compares the two.",
    "`--extra-checkpoints` only adds files (`best_loss.pt`, `last.pt`); `best.pt` is still the best validation "
    "gap and is the checkpoint evaluated at the end of training. The flag is off by default, so step 5 "
    "(training-set size) keeps the phase 1 selection by validation gap.",
    "\"Best validation loss\" uses the same weighted BCE as training (pos_weight from the training data), on "
    "TSP-20 validation. Ties keep the earliest epoch.",
    "Each LayerNorm checkpoint is evaluated on TSP-20/50/100 with k = 10 and with k = 0.5 n (the k = 0.5 n "
    "evaluation includes TSP-20, where it is the same graph as k = 10).",
    "Check 2 (`train50_kfrac`): `--k-frac 0.5` for training, validation and test graphs (k = 25 on TSP-50, "
    "k = 50 on TSP-100, k = 10 on TSP-20); everything else as in the batch 2 control. It is also evaluated on "
    "TSP-20 for check 3.",
    "Check 3: the TSP-50-trained models of batch 2 (`train50`) are evaluated on TSP-20 in a new folder "
    "(`results/phase2_checks/train50_tsp20/`). For `kfrac` and `knn_mean_kfrac` the TSP-20 column is the "
    "k = 10 evaluation (same graph). Every configuration now has TSP-20.",
    "Check 4 tests **every** configuration with 2-opt against the two baselines with 2-opt (not only the "
    "best ones), because picking the best configuration on the test set and then testing it on the same test "
    "set would bias the test; the best configuration per model is then read from the table. Bonferroni is "
    "applied over all tests of Table 5. Baselines are deterministic, so seed s of the model is compared with the "
    "same baseline gaps for every seed.",
    "Conservative reading of the rules: nothing from phase 1, experiment 1 or batch 2 was rewritten (new folders "
    "only); a failed step is logged and the steps that depend on it are skipped; the report is copied to `docs/` "
    "but not committed.",
]


# ---------------------------------------------------------------- loading

def load_base_per_instance(results):
    out = {}
    for path in (os.path.join(results, "baselines_per_instance.npz"),
                 os.path.join(results, "phase2_exp1", "baselines_tsp100_per_instance.npz")):
        if os.path.exists(path):
            with np.load(path) as f:
                out.update({k: f[k] for k in f.files})
    return out


def load_histories(results):
    """{(model, seed): (history list, {selection: epoch})} for the check 1 runs that finished."""
    out = {}
    for m in MODELS:
        for s in SEEDS:
            path = os.path.join(results, "runs", ln_run(m, s), "history.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                hist = json.load(f)
            loss = [h["val_loss"] for h in hist]
            gap = [h["val_gap_greedy"] for h in hist]
            out[m, s] = (hist, {"gap": int(np.argmin(gap)) + 1, "loss": int(np.argmin(loss)) + 1, "last": len(hist)})
    return out


# ---------------------------------------------------------------- comparisons

def paired(a_list, ref_list, threshold):
    """Paired comparison a − ref, one per seed, aggregated as in phase 1 (max p, same sign of the median)."""
    res = [summarize.paired_compare(a, r) for a, r in zip(a_list, ref_list)]
    avg = {k: float(np.mean([x[k] for x in res])) for k in res[0]}
    p_max = max(x["p"] for x in res)
    same_sign = len({np.sign(x["median_diff"]) for x in res}) == 1
    direction = avg["median_diff"] if avg["median_diff"] != 0 else avg["mean_diff"]
    verdict = ("better" if direction < 0 else "worse") if p_max < threshold and same_sign else "n.s."
    return {**avg, "p_max": p_max, "verdict": verdict}


def direct_tests(configs):
    return [(c, ref, m, d, x) for c, ref in DIRECT for m in MODELS for d in DATASETS for x in METHODS
            if d in configs[c][m] and d in configs[ref][m]]


def baseline_tests(configs, base):
    return [(c, b, m, d) for c in CONFIGS for b in BASE_REFS for m in MODELS for d in ("tsp50_test", "tsp100_test")
            if d in configs[c][m] and f"{d}__{b}" in base]


def row(label, r):
    verdict = f"**{r['verdict']}**" if r["verdict"] != "n.s." else "n.s."
    return (f"| {label} | {r['better']:.1f}% | {r['equal']:.1f}% | {r['worse']:.1f}% | {r['mean_diff']:+.2f} "
            f"| {r['median_diff']:+.2f} | {r['p_max']:.1e} | {verdict} |")


HEADER = ("Better | Equal | Worse | Mean diff. (pp) | Median diff. (pp) | max p (seeds) | Conclusion |",
          "---|---|---|---|---|---|---|")


# ---------------------------------------------------------------- tables

def table_gaps(baselines, configs):
    """Mean ± std across seeds for every configuration and dataset (all of them are expected now)."""
    cols = [(d, x) for d in DATASETS for x in METHODS]
    lines = ["| Model | Configuration | " + " | ".join(f"{DATASETS[d]} {METHODS[x]}" for d, x in cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for key, name in summarize.BASELINES.items():
        cells = [f"{baselines[d][key if x == 'greedy' else key + '+2opt']['gap_mean']:.2f}" if d in baselines
                 else "missing" for d, x in cols]
        lines.append(f"| {name} | baseline | " + " | ".join(cells) + " |")
    for m, mname in MODELS.items():
        for c, cfg in configs.items():
            cells = [batch2.gap_cell(cfg, m, d, x, list(DATASETS)) for d, x in cols]
            lines.append(f"| {mname} | {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_tradeoff(configs):
    """Greedy gap on TSP-20/50/100 and the change vs. phase 1 (same seeds): the trade-off view of check 3."""
    ref = configs["phase1"]
    lines = ["| Model | Configuration | TSP-20 | Δ TSP-20 | TSP-50 | Δ TSP-50 | TSP-100 | Δ TSP-100 |",
             "|---|---|---|---|---|---|---|---|"]
    for m, mname in MODELS.items():
        for c, cfg in configs.items():
            cells = []
            for d in DATASETS:
                g = batch2.mean_gap(cfg, m, d, "greedy")
                r = batch2.mean_gap(ref, m, d, "greedy")
                cells += ["missing" if g is None else f"{g:.2f}",
                          "—" if c == "phase1" or None in (g, r) else f"{g - r:+.2f}"]
            lines.append(f"| {mname} | {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_epochs(histories):
    lines = ["| Model | Seed | Epoch: best val gap | Epoch: best val loss | val gap at (gap / loss / last) "
             "| val loss at (gap / loss / last) |", "|---|---|---|---|---|---|"]
    for (m, s), (hist, ep) in sorted(histories.items()):
        gaps = " / ".join(f"{hist[ep[k] - 1]['val_gap_greedy']:.2f}" for k in SELECTIONS)
        losses = " / ".join(f"{hist[ep[k] - 1]['val_loss']:.4f}" for k in SELECTIONS)
        lines.append(f"| {MODELS[m]} | {s} | {ep['gap']} | {ep['loss']} | {gaps} | {losses} |")
    return "\n".join(lines)


def table_direct(configs, threshold):
    lines = ["| Configuration − reference | Model | Dataset | Method | " + HEADER[0], "|---|---|---|---|" + HEADER[1]]
    out = {}
    for c, ref, m, d, x in direct_tests(configs):
        key = f"{d}__{x}"
        r = paired([pi[key] for pi in configs[c][m][d]["per_instance"]],
                   [pi[key] for pi in configs[ref][m][d]["per_instance"]], threshold)
        out[c, ref, m, d, x] = r
        lines.append(row(f"{c} − {ref} | {MODELS[m]} | {DATASETS[d]} | {METHODS[x]}", r))
    return "\n".join(lines), out


def table_baselines(configs, base, threshold):
    lines = ["| Configuration + 2-opt vs. | Model | Dataset | Model gap | Baseline gap | " + HEADER[0],
             "|---|---|---|---|---|" + HEADER[1]]
    out = {}
    for c, b, m, d in baseline_tests(configs, base):
        ref = base[f"{d}__{b}"]
        model = [pi[f"{d}__greedy+2opt"] for pi in configs[c][m][d]["per_instance"]]
        r = paired(model, [ref] * len(model), threshold)
        out[c, b, m, d] = r
        gap = batch2.mean_gap(configs[c], m, d, "greedy+2opt")
        lines.append(row(f"{c} vs. {BASE_REFS[b]} | {MODELS[m]} | {DATASETS[d]} | {gap:.2f} | {ref.mean():.2f}", r))
    return "\n".join(lines), out


# ---------------------------------------------------------------- figure

def plot_curves(histories, figs_dir):
    """Validation gap (top) and validation/training loss (bottom) per epoch, one column per model."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    markers = {0: "o", 1: "s", 2: "^"}
    for col, (m, mname) in enumerate(MODELS.items()):
        ax_gap, ax_loss = axes[0, col], axes[1, col]
        for s in SEEDS:
            if (m, s) not in histories:
                continue
            hist, ep = histories[m, s]
            epochs = [h["epoch"] for h in hist]
            color = MODEL_COLORS[m]
            alpha = 1.0 - 0.3 * s
            ax_gap.plot(epochs, [h["val_gap_greedy"] for h in hist], color=color, alpha=alpha, marker=markers[s],
                        markersize=3, linewidth=1.4)
            ax_loss.plot(epochs, [h["val_loss"] for h in hist], color=color, alpha=alpha, marker=markers[s],
                         markersize=3, linewidth=1.4)
            ax_loss.plot(epochs, [h["train_loss"] for h in hist], color=color, alpha=alpha * 0.6, linestyle="--",
                         linewidth=1)
            for ax, key in ((ax_gap, "val_gap_greedy"), (ax_loss, "val_loss")):
                for sel, edge in (("gap", INK), ("loss", "#d62728")):
                    e = ep[sel]
                    ax.scatter(e, hist[e - 1][key], s=70, facecolors="none", edgecolors=edge, linewidths=1.4,
                               zorder=3)
        for ax in (ax_gap, ax_loss):
            summarize.style_axes(ax)
        ax_gap.set_title(f"{mname} (LayerNorm)", color=INK, fontsize=10, loc="left")
        ax_loss.set_xlabel("Epoch", color=INK)
        ax_loss.set_xticks([1, 5, 10, 15, 20])
    axes[0, 0].set_ylabel("Validation gap, greedy (%)", color=INK)
    axes[1, 0].set_ylabel("Weighted BCE loss", color=INK)
    handles = [Line2D([], [], color=MUTED, marker=markers[s], markersize=4, label=f"seed {s}") for s in SEEDS]
    handles += [Line2D([], [], color=MUTED, linestyle="--", linewidth=1, label="training loss"),
                Line2D([], [], color="none", marker="o", markersize=9, markerfacecolor="none", markeredgecolor=INK,
                       label="epoch with best val gap"),
                Line2D([], [], color="none", marker="o", markersize=9, markerfacecolor="none",
                       markeredgecolor="#d62728", label="epoch with best val loss")]
    fig.suptitle("LayerNorm reruns: validation curves on TSP-20", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    os.makedirs(figs_dir, exist_ok=True)
    summarize.save(fig, figs_dir, "phase2_checks_ln_curves")


# ---------------------------------------------------------------- summary

def summary_points(configs, histories, direct, vs_base):
    points = []
    # 1. When are the checkpoints selected?
    if histories:
        eps = {sel: [ep[sel] for _, ep in histories.values()] for sel in ("gap", "loss")}
        early = sum(e <= 5 for e in eps["gap"])
        points.append(f"**Check 1 — selected epochs** ({len(histories)} LayerNorm reruns): best val gap at epochs "
                      f"{sorted(eps['gap'])} ({early} at epoch ≤ 5); best val loss at epochs {sorted(eps['loss'])}.")
    else:
        points.append("**Check 1**: no LayerNorm reruns found (see the log).")
    # 2. Does the checkpoint choice change generalization?
    for suf, gname in (("", "k = 10"), ("_kfrac", "k = 0.5 n")):
        parts = []
        for sel in ("loss", "last"):
            for d in ("tsp20_test", "tsp50_test", "tsp100_test"):
                sub = [r for (c, ref, m, dd, x), r in direct.items()
                       if c == f"ln_{sel}{suf}" and ref == f"ln_gap{suf}" and dd == d and x == "greedy"]
                if sub:
                    parts.append(f"{sel} vs. gap on {DATASETS[d]}: {sum(r['verdict'] == 'better' for r in sub)} "
                                 f"better / {sum(r['verdict'] == 'worse' for r in sub)} worse of {len(sub)} "
                                 f"(mean {np.mean([r['mean_diff'] for r in sub]):+.2f} pp)")
        if parts:
            points.append(f"**Check 1 — checkpoint choice, greedy, {gname}**: " + "; ".join(parts) + ".")
    # 3. Rerun vs. experiment 3.
    sub = [r for (c, ref, *_), r in direct.items() if (c, ref) in (("ln_gap", "layernorm"),
                                                                   ("ln_gap_kfrac", "layernorm_kfrac"))]
    if sub:
        points.append(f"**Check 1 — rerun vs. experiment 3** (same command, best val gap): "
                      f"{sum(r['verdict'] != 'n.s.' for r in sub)} of {len(sub)} cells differ significantly; "
                      f"mean |diff.| {np.mean([abs(r['mean_diff']) for r in sub]):.2f} pp.")
    # 4. Control with k = 0.5 n.
    parts = []
    for m, mname in MODELS.items():
        a = batch2.mean_gap(configs["train50_kfrac"], m, "tsp50_test", "greedy")
        b = batch2.mean_gap(configs["train50"], m, "tsp50_test", "greedy")
        ln = batch2.mean_gap(configs["ln_gap_kfrac"], m, "tsp50_test", "greedy")
        if None not in (a, b):
            parts.append(f"{mname} {a:.2f}% (k = 10: {b:.2f}%" + (f"; LayerNorm TSP-20 + k = 0.5 n: {ln:.2f}%)"
                                                                   if ln is not None else ")"))
    points.append("**Check 2 — TSP-50 control with k = 0.5 n**, greedy gap on TSP-50: "
                  + ("; ".join(parts) if parts else "no results") + ".")
    # 5. Trade-off on TSP-20.
    ref = configs["phase1"]
    worse20 = []
    for c, cfg in configs.items():
        for m, mname in MODELS.items():
            g20, r20 = batch2.mean_gap(cfg, m, "tsp20_test", "greedy"), batch2.mean_gap(ref, m, "tsp20_test", "greedy")
            g50, r50 = batch2.mean_gap(cfg, m, "tsp50_test", "greedy"), batch2.mean_gap(ref, m, "tsp50_test", "greedy")
            if None not in (g20, r20, g50, r50) and c != "phase1" and g50 < r50 and g20 - r20 > 0.5:
                worse20.append(f"{c}/{mname} (TSP-20 {g20 - r20:+.2f}, TSP-50 {g50 - r50:+.2f})")
    points.append("**Check 3 — trade-off** (greedy, vs. phase 1): configurations that improve TSP-50 but lose "
                  "more than 0.5 pp on TSP-20: " + (", ".join(worse20) if worse20 else "none") + ".")
    # 6. Against the baselines.
    for d in ("tsp50_test", "tsp100_test"):
        both = {}
        for (c, b, m, dd), r in vs_base.items():
            if dd == d:
                both.setdefault((c, m), []).append(r["verdict"])
        wins = [f"{c}/{MODELS[m]}" for (c, m), v in both.items() if v == ["better", "better"]]
        best = []
        for m, mname in MODELS.items():
            cands = [(batch2.mean_gap(configs[c], m, d, "greedy+2opt"), c) for c in CONFIGS
                     if d in configs[c][m]]
            if cands:
                g, c = min(cands)
                best.append(f"{mname}: {c} {g:.2f}%")
        points.append(f"**Check 4 — {DATASETS[d]} with 2-opt**: {len(wins)} of {len(both)} configuration/model "
                      f"pairs are significantly better than **both** baselines + 2-opt. Lowest mean gap per model: "
                      + "; ".join(best) + ".")
    return points[:10]


# ---------------------------------------------------------------- main

def build_report(results):
    configs = {c: batch2.load_config(results, folders) for c, (_, folders) in CONFIGS.items()}
    baselines = batch2.load_baselines(results)
    base = load_base_per_instance(results)
    histories = load_histories(results)
    n_direct, n_base = len(direct_tests(configs)), len(baseline_tests(configs, base))
    th_direct, th_base = ALPHA / max(n_direct, 1), ALPHA / max(n_base, 1)
    t_direct, direct = table_direct(configs, th_direct)
    t_base, vs_base = table_baselines(configs, base, th_base)
    missing = [f"{c} / {MODELS[m]} / {DATASETS[d]}" for c in CONFIGS for m in MODELS for d in DATASETS
               if d not in configs[c][m]]
    legend = "\n".join(f"- `{c}` — {desc}" for c, (desc, _) in CONFIGS.items())
    test_text = (f"*Equal* = difference ≤ {TIE_TOL:g} pp. Two-sided Wilcoxon test per seed (ties dropped); "
                 f"*max p* is the largest p-value across the 3 seeds; significant only if max p is below the "
                 f"Bonferroni threshold and the median has the same sign in every seed.")
    return "\n".join([
        "# Phase 2 — checks (points 1 to 4)",
        "",
        f"Generated by `scripts/phase2_checks_report.py` on {datetime.date.today().isoformat()}.",
        "All models: H=64, L=6, 12000 training instances, 20 epochs, seeds 0, 1, 2; trained on TSP-20 unless "
        "stated otherwise.",
        "",
        "Configurations:",
        "",
        legend,
        "",
        "## Summary",
        "",
        "\n".join(f"{i}. {p}" for i, p in enumerate(summary_points(configs, histories, direct, vs_base), 1)),
        "",
        "## Table 1 — mean gap to LKH (%), every configuration on TSP-20, TSP-50 and TSP-100 (check 3)",
        "",
        "Models: mean ± std across the 3 seeds. \"missing\" = expected but not found.",
        "",
        table_gaps(baselines, configs),
        "",
        "## Table 2 — trade-off: greedy gap and change vs. phase 1 (pp, negative = better)",
        "",
        table_tradeoff(configs),
        "",
        "## Check 1 — LayerNorm checkpoints",
        "",
        "![LayerNorm validation curves](figs/phase2_checks_ln_curves.png)",
        "",
        "Top: validation gap (greedy, TSP-20) per epoch. Bottom: validation loss (solid) and training loss "
        "(dashed). Black circle: epoch with the best validation gap (`best.pt`); red circle: best validation "
        "loss (`best_loss.pt`); the last epoch is `last.pt`.",
        "",
        table_epochs(histories) if histories else "No LayerNorm reruns found.",
        "",
        "## Table 3 — paired comparisons for checks 1 and 2",
        "",
        f"Diff. = configuration − reference (negative = the configuration is better). Bonferroni: {ALPHA}/"
        f"{n_direct} tests, threshold {th_direct:.2e}. " + test_text,
        "",
        t_direct if direct else "No results yet.",
        "",
        "## Table 4 — baselines (deterministic)",
        "",
        "| Baseline | " + " | ".join(f"{DATASETS[d]} {METHODS[x]}" for d in DATASETS for x in METHODS) + " |",
        "|---|" + "---|" * 6,
        "\n".join(f"| {name} | " + " | ".join(
            f"{baselines[d][k if x == 'greedy' else k + '+2opt']['gap_mean']:.2f}" if d in baselines else "missing"
            for d in DATASETS for x in METHODS) + " |" for k, name in summarize.BASELINES.items()),
        "",
        "## Table 5 — every configuration + 2-opt against the baselines + 2-opt (check 4)",
        "",
        f"Diff. = model − baseline (negative = the model is better). Bonferroni: {ALPHA}/{n_base} tests, threshold "
        f"{th_base:.2e}. " + test_text,
        "",
        t_base if vs_base else "No results yet.",
        "",
        "## Decisions",
        "",
        "Choices made without asking (the most conservative option each time):",
        "",
        "\n".join(f"{i}. {d}" for i, d in enumerate(DECISIONS, 1)),
        "",
        "## Missing results",
        "",
        "\n".join(f"- {x}" for x in missing) if missing else "None: every configuration has all 3 seeds.",
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description="Phase 2 checks report (points 1 to 4).")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/phase2_checks_report.md")
    args = parser.parse_args()
    report = build_report(args.results)
    out_dir = os.path.dirname(args.out) or "."
    plot_curves(load_histories(args.results), os.path.join(out_dir, "figs"))
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"saved: {args.out} and {os.path.join(out_dir, 'figs', 'phase2_checks_ln_curves.png')}")


if __name__ == "__main__":
    main()
