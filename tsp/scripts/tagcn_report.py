"""TAGCN report: the fourth encoder (Du et al. 2017) next to GCN, GATv2 and Gated GCN, phase 1 and phase 2.

Reads (TAGCN, K = 2 and 3, seeds 0, 1, 2; written by scripts/run_tagcn.sh):
    results/runs/tagcn_k<K>_seed<s>/                      phase 1 training (best.pt, k = 10, TSP-20/50)
        kfrac05/                                          best.pt on TSP-50/100 with k = 0.5 n
        beam/                                             greedy / beam 10 / beam 100 (± 2-opt), TSP-20/50/100, k = 0.5 n
    results/runs/tagcn_k<K>_layernorm_seed<s>/last_kfrac05/   LayerNorm retraining, last.pt, k = 0.5 n
    results/runs/tagcn_k<K>_ckpt_seed<s>/                 BatchNorm rerun with last.pt (scripts/run_tagcn_ckpt.sh)
        last_beam/                                        last.pt: greedy / beam 100 (± 2-opt), TSP-20/50/100, k = 0.5 n
        last_k10/                                         last.pt: greedy (± 2-opt), TSP-20/50, k = 10 (phase 1 graphs)
and the earlier results of the other three models with the same protocol (nothing is recomputed):
    results/runs/<m>_seed<s>/, results/phase2_batch2/kfrac05/<m>_seed<s>/,
    results/phase2_checks/ln/last_kfrac05/<m>_layernorm_ckpt_seed<s>/, results/phase2_beam/phase1/<m>_seed<s>/.
Writes results/tagcn_report.md and results/figs/tagcn_curves.{png,pdf}. Missing results are reported as
"missing" instead of stopping the report. Hand-written notes (the "three lines") are read from
results/tagcn_notes.md if that file exists.

Example:
    python scripts/tagcn_report.py
"""

import argparse
import datetime
import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")  # no window: only save files
import matplotlib.pyplot as plt
import numpy as np

from tsp_gnn.models import EdgeGNN, count_parameters, default_layers

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarize = _load("summarize")
batch2 = _load("phase2_batch2_report")
checks = _load("phase2_checks_report")
beam_report = _load("phase2_beam_report")
ALPHA, TIE_TOL, INK, MUTED = summarize.ALPHA, summarize.TIE_TOL, summarize.INK, summarize.MUTED
DATASETS, SEEDS = batch2.DATASETS, batch2.SEEDS
TAG_KS = (2, 3)

# Architectures, in table order: key -> (name, colour). Colours 1-3 are the phase 1 ones; 4-5 the next two
# slots of the same validated palette (checked with the dataviz validator; lines are also labelled directly).
ARCHS = {"gcn": ("GCN", "#2a78d6"), "gat": ("GATv2", "#eb6834"), "gated": ("Gated GCN", "#1baf7a"),
         "tagcn_k2": ("TAGCN K=2", "#eda100"), "tagcn_k3": ("TAGCN K=3", "#e87ba4")}
TAGCN = ("tagcn_k2", "tagcn_k3")


def tag_run(a, s, norm=""):
    """Run folder name of TAGCN architecture a ('tagcn_k2' / 'tagcn_k3')."""
    return f"{a}{'_layernorm' if norm else ''}_seed{s}"


# block -> (description, datasets, methods, folder of (arch, seed) relative to results/)
BLOCKS = {
    "phase1": ("Phase 1: k = 10, BatchNorm, best val gap", ("tsp20_test", "tsp50_test"), ("greedy", "greedy+2opt"),
               lambda a, s: f"runs/{tag_run(a, s)}" if a in TAGCN else f"runs/{a}_seed{s}"),
    "kfrac": ("Phase 1 checkpoints, test graphs k = 0.5 n", ("tsp50_test", "tsp100_test"), ("greedy", "greedy+2opt"),
              lambda a, s: f"runs/{tag_run(a, s)}/kfrac05" if a in TAGCN else f"phase2_batch2/kfrac05/{a}_seed{s}"),
    "ln": ("LayerNorm retraining, last epoch, k = 0.5 n", tuple(DATASETS), ("greedy", "greedy+2opt"),
           lambda a, s: (f"runs/{tag_run(a, s, 'ln')}/last_kfrac05" if a in TAGCN
                         else f"phase2_checks/ln/last_kfrac05/{a}_layernorm_ckpt_seed{s}")),
    "beam": ("Phase 1 checkpoints, beam search, k = 0.5 n", tuple(DATASETS),
             tuple(k for dec in ("greedy", "beam10", "beam100") for k in (dec, dec + "+2opt")),
             lambda a, s: f"runs/{tag_run(a, s)}/beam" if a in TAGCN else f"phase2_beam/phase1/{a}_seed{s}"),
}
# BatchNorm reruns with --extra-checkpoints: last.pt of the phase 1 protocol (TAGCN only).
CKPT_RUN = lambda a, s: f"runs/{a}_ckpt_seed{s}"
LAST_METHODS = ("greedy", "greedy+2opt", "beam100", "beam100+2opt")
METHOD_LABELS = {"greedy": "greedy", "greedy+2opt": "+ 2-opt", "beam10": "B=10", "beam10+2opt": "B=10 + 2-opt",
                 "beam100": "B=100", "beam100+2opt": "B=100 + 2-opt"}
BASE_LABELS = {"greedy": "Classical greedy", "christofides": "Christofides"}


# ---------------------------------------------------------------- loading

def load_all(results):
    """{block: {arch: [per seed: (eval.json, per-instance dict) or None]}}."""
    return {b: {a: [batch2.load_folder(os.path.join(results, folder(a, s))) for s in SEEDS] for a in ARCHS}
            for b, (_, _, _, folder) in BLOCKS.items()}


def load_last(results):
    """last.pt with k = 0.5 n, last.pt with k = 10 and the rerun's best.pt: {arch: [per seed: loaded or None]}."""
    load = lambda sub: {a: [batch2.load_folder(os.path.join(results, CKPT_RUN(a, s), sub)) for s in SEEDS]
                        for a in TAGCN}
    return load("last_beam"), load("last_k10"), load("")


def complete(runs, d):
    return all(r is not None and d in r[0] for r in runs)


def gaps(runs, d, meth):
    """Mean gap of each seed, or None if a seed is missing."""
    return [r[0][d][meth]["gap_mean"] for r in runs] if complete(runs, d) else None


def history_path(results, a, s, ln):
    if a in TAGCN:
        return os.path.join(results, "runs", tag_run(a, s, "ln" if ln else ""), "history.json")
    return os.path.join(results, "runs", f"{a}_layernorm_ckpt_seed{s}" if ln else f"{a}_seed{s}", "history.json")


def load_histories(results):
    """{(ln, arch): [history per seed that exists]}."""
    out = {}
    for ln in (False, True):
        for a in ARCHS:
            hs = []
            for s in SEEDS:
                path = history_path(results, a, s, ln)
                if os.path.exists(path):
                    with open(path) as f:
                        hs.append(json.load(f))
            out[ln, a] = hs
    return out


def run_config(results, a, s):
    """config of the phase 1 checkpoint (from its eval.json), or None."""
    loaded = batch2.load_folder(os.path.join(results, BLOCKS["phase1"][3](a, s)))
    return loaded[0]["meta"]["config"] if loaded else None


# ---------------------------------------------------------------- tables

def cell(values, std=False):
    if values is None:
        return "missing"
    return summarize.mean_std(values) if std else f"{np.mean(values):.2f}"


def base_cell(baselines, d, meth):
    """Baseline gap for a method name (beam methods have no baseline equivalent)."""
    if meth.startswith("beam"):
        return "—"
    return f"{baselines[d][meth]['gap_mean']:.2f}" if d in baselines else "missing"


MAIN_COLS = [("phase1", "tsp20_test", "Phase 1 TSP-20"), ("phase1", "tsp50_test", "Phase 1 TSP-50"),
             ("kfrac", "tsp50_test", "k=0.5n TSP-50"), ("kfrac", "tsp100_test", "k=0.5n TSP-100"),
             ("ln", "tsp50_test", "LayerNorm TSP-50"), ("ln", "tsp100_test", "LayerNorm TSP-100"),
             ("beam", "tsp50_test", "B=100 TSP-50"), ("beam", "tsp100_test", "B=100 TSP-100")]


def table_main(data, baselines, last=None, last_k10=None):
    """Mean gap over the 3 seeds: 'without 2-opt / with 2-opt' in each cell; bold = best model in the column.

    last: TAGCN last.pt rows (BatchNorm reruns), shown below and not counted for the bold.
    """
    lines = ["| Method | " + " | ".join(lab for _, _, lab in MAIN_COLS) + " |", "|---|" + "---|" * len(MAIN_COLS)]
    for b, name in BASE_LABELS.items():
        cells = [f"{base_cell(baselines, d, b)} / {base_cell(baselines, d, b + '+2opt')}" for _, d, _ in MAIN_COLS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    best = {}
    for blk, d, _ in MAIN_COLS:
        pre = "beam100" if blk == "beam" else "greedy"
        for suffix in ("", "+2opt"):
            vals = {a: np.mean(g) for a in ARCHS if (g := gaps(data[blk][a], d, pre + suffix)) is not None}
            best[blk, d, suffix] = min(vals, key=vals.get) if vals else None
    for a, (name, _) in ARCHS.items():
        cells = []
        for blk, d, _ in MAIN_COLS:
            pre = "beam100" if blk == "beam" else "greedy"
            parts = []
            for suffix in ("", "+2opt"):
                txt = cell(gaps(data[blk][a], d, pre + suffix))
                parts.append(f"**{txt}**" if best[blk, d, suffix] == a else txt)
            cells.append(" / ".join(parts))
        lines.append(f"| {name}{' (best.pt)' if a in TAGCN else ''} | " + " | ".join(cells) + " |")
    for a in TAGCN if last else ():
        cells = []
        for blk, d, _ in MAIN_COLS:
            if blk == "ln":
                cells.append("—")      # LayerNorm is a different training
                continue
            pre = "beam100" if blk == "beam" else "greedy"
            src = last_k10[a] if blk == "phase1" else last[a]   # phase 1 columns: k = 10 graphs
            cells.append(" / ".join(cell(gaps(src, d, pre + suffix)) for suffix in ("", "+2opt")))
        lines.append(f"| {ARCHS[a][0]} (last.pt, rerun) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_block(data, baselines, blk, results):
    """Mean ± std over seeds for every dataset and method of one block (+ selected epochs in phase 1)."""
    _, datasets, methods, _ = BLOCKS[blk]
    cols = [(d, m) for d in datasets for m in methods]
    extra = " Selected epochs |" if blk == "phase1" else ""
    lines = ["| Method | " + " | ".join(f"{DATASETS[d]} {METHOD_LABELS[m]}" for d, m in cols) + " |" + extra,
             "|---|" + "---|" * (len(cols) + (1 if extra else 0))]
    for b, name in BASE_LABELS.items():
        cells = [base_cell(baselines, d, m.replace("greedy", b)) if not m.startswith("beam") else "—"
                 for d, m in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |" + (" — |" if extra else ""))
    for a, (name, _) in ARCHS.items():
        runs = data[blk][a]
        cells = [cell(gaps(runs, d, m), std=True) for d, m in cols]
        row = f"| {name} | " + " | ".join(cells) + " |"
        if extra:
            row += " " + ", ".join(str(r[0]["meta"]["best_epoch"]) if r else "?" for r in runs) + " |"
        lines.append(row)
    return "\n".join(lines)


def table_params(results, histories):
    """Parameters (counted by building each model), reach, peak GPU memory and training time per run."""
    lines = ["| Model | Layers L | Hops per layer | Reach (hops) | Parameters | Encoder layers only | "
             "Peak GPU (MB, max over runs) | Training time per run (min, mean) |",
             "|---|---|---|---|---|---|---|---|"]
    for a, (name, _) in ARCHS.items():
        kind, tag_k = ("tagcn", int(a[-1])) if a in TAGCN else (a, 1)
        layers = default_layers(kind, tag_k)
        model = EdgeGNN(kind, hidden=64, layers=layers, tag_k=max(tag_k, 1))
        total = count_parameters(model)
        encoder = sum(count_parameters(m) for m in model.convs) + sum(
            count_parameters(m) for m in getattr(model, "norms", []))
        cfgs = [c for s in SEEDS if (c := run_config(results, a, s))]
        logged = {c["num_parameters"] for c in cfgs if "num_parameters" in c}
        check = "" if not logged or logged == {total} else f" (logged: {sorted(logged)}!)"
        hs = histories[False, a] + histories[True, a]
        peak = f"{max(h['peak_gpu_mb'] for hist in hs for h in hist):.0f}" if hs else "missing"
        minutes = f"{np.mean([sum(h['seconds'] for h in hist) for hist in hs]) / 60:.1f}" if hs else "missing"
        lines.append(f"| {name} | {layers} | {tag_k} | {layers * tag_k} | {total:,}{check} | {encoder:,} | "
                     f"{peak} | {minutes} |")
    return "\n".join(lines)


def table_last(last, rerun, baselines):
    """TAGCN last.pt of the BatchNorm reruns: mean ± std over seeds, all datasets and methods."""
    cols = [(d, m) for d in DATASETS for m in LAST_METHODS]
    lines = ["| Method | " + " | ".join(f"{DATASETS[d]} {METHOD_LABELS[m]}" for d, m in cols) + " | Epoch |",
             "|---|" + "---|" * (len(cols) + 1)]
    for b, name in BASE_LABELS.items():
        cells = [base_cell(baselines, d, m.replace("greedy", b)) for d, m in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " | — |")
    for a in TAGCN:
        cells = [cell(gaps(last[a], d, m), std=True) for d, m in cols]
        epochs = ", ".join(str(r[0]["meta"]["epoch"]) if r else "?" for r in last[a])
        lines.append(f"| {ARCHS[a][0]} last.pt | " + " | ".join(cells) + f" | {epochs} |")
    return "\n".join(lines)


def table_rerun(data, rerun, results):
    """best.pt of the first run vs. best.pt of the rerun (same seed and code), per run."""
    lines = ["| Run | Selected epoch (first / rerun) | Val gap at that epoch (first / rerun) | "
             "Max abs. diff. of the val loss curves | TSP-20 greedy (first / rerun) | TSP-50 greedy (first / rerun) | "
             "Identical per-instance gaps, TSP-20 + TSP-50 greedy |",
             "|---|---|---|---|---|---|---|"]
    for a in TAGCN:
        for s, first, again in zip(SEEDS, data["phase1"][a], rerun[a]):
            if first is None or again is None:
                lines.append(f"| {tag_run(a, s)} | missing | | | | | |")
                continue
            hists = []
            for path in (os.path.join(results, "runs", tag_run(a, s), "history.json"),
                         os.path.join(results, CKPT_RUN(a, s), "history.json")):
                with open(path) as f:
                    hists.append([h["val_loss"] for h in json.load(f)])
            dloss = max(abs(x - y) for x, y in zip(*hists))
            m1, m2 = first[0]["meta"], again[0]["meta"]
            same, total = 0, 0
            for d in ("tsp20_test", "tsp50_test"):
                diff = np.abs(first[1][f"{d}__greedy"] - again[1][f"{d}__greedy"])
                same, total = same + int((diff <= TIE_TOL).sum()), total + len(diff)
            g = lambda r, d: f"{r[0][d]['greedy']['gap_mean']:.2f}"
            lines.append(f"| {tag_run(a, s)} | {m1['best_epoch']} / {m2['best_epoch']} | "
                         f"{m1['best_val_gap']:.2f} / {m2['best_val_gap']:.2f} | {dloss:.1e} | "
                         f"{g(first, 'tsp20_test')} / {g(again, 'tsp20_test')} | "
                         f"{g(first, 'tsp50_test')} / {g(again, 'tsp50_test')} | {same}/{total} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- statistical tests

def phase1_tests(data, base):
    """TAGCN vs. each baseline, same decoder (greedy vs. greedy, + 2-opt vs. + 2-opt), TSP-20 and TSP-50."""
    tests = []
    for a in TAGCN:
        runs = data["phase1"][a]
        for d in BLOCKS["phase1"][1]:
            for meth in ("greedy", "greedy+2opt"):
                for b, lab in BASE_LABELS.items():
                    key = f"{d}__{meth.replace('greedy', b)}"
                    if complete(runs, d) and key in base:
                        tests.append((a, d, METHOD_LABELS[meth], lab, [r[1][f"{d}__{meth}"] for r in runs],
                                      [base[key]] * len(runs)))
    return tests


def kfrac_greedy_tests(data, last, base):
    """Greedy (no 2-opt) with k = 0.5 n vs. Christofides without 2-opt, TSP-50 and TSP-100.

    TAGCN K=2 / K=3 best.pt and last.pt (reruns) and GCN best.pt.
    """
    models = [(f"{ARCHS[a][0]} best.pt", data["kfrac"][a]) for a in TAGCN]
    models += [(f"{ARCHS[a][0]} last.pt", last[a]) for a in TAGCN]
    models += [("GCN best.pt", data["kfrac"]["gcn"])]
    tests = []
    for name, runs in models:
        for d in ("tsp50_test", "tsp100_test"):
            key = f"{d}__christofides"
            if complete(runs, d) and key in base:
                tests.append((name, d, "greedy", "Christofides", [r[1][f"{d}__greedy"] for r in runs],
                              [base[key]] * len(runs)))
    return tests


def beam_tests(data, base):
    """TAGCN beam 100 + 2-opt vs. each baseline + 2-opt, TSP-20/50/100."""
    tests = []
    for a in TAGCN:
        runs = data["beam"][a]
        for d in DATASETS:
            for b, lab in BASE_LABELS.items():
                key = f"{d}__{b}+2opt"
                if complete(runs, d) and key in base:
                    tests.append((a, d, "B=100 + 2-opt", f"{lab} + 2-opt", [r[1][f"{d}__beam100+2opt"] for r in runs],
                                  [base[key]] * len(runs)))
    return tests


def table_tests(tests):
    """Rows of one family of tests, Bonferroni over the family. Returns (markdown, verdicts)."""
    threshold = ALPHA / max(len(tests), 1)
    lines = [f"Bonferroni: {ALPHA}/{len(tests)} tests, threshold {threshold:.2e}.", "",
             "| Model | Dataset | Decoder | Reference | Model gap | Ref. gap | " + checks.HEADER[0],
             "|---|---|---|---|---|---|" + checks.HEADER[1]]
    verdicts = []
    for a, d, meth, lab, x, ref in tests:
        r = beam_report.paired(x, ref, threshold)
        verdicts.append((a, d, meth, lab, r))
        label = (f"{ARCHS.get(a, (a,))[0]} | {DATASETS[d]} | {meth} | {lab} | {np.mean([v.mean() for v in x]):.2f} | "
                 f"{np.mean([v.mean() for v in ref]):.2f}")
        lines.append(checks.row(label, r))
    return "\n".join(lines), verdicts


def consistency(data):
    """TAGCN greedy gaps in the beam folder vs. the same checkpoint's earlier evaluation (same graphs)."""
    lines = ["| Model | Dataset | Compared with | Instances | Different (> 1e-6 pp) | max abs. diff. (pp) |",
             "|---|---|---|---|---|---|"]
    for a in TAGCN:
        for d, blk in (("tsp20_test", "phase1"), ("tsp50_test", "kfrac"), ("tsp100_test", "kfrac")):
            n, diff_count, max_diff = 0, 0, 0.0
            for rb, ro in zip(data["beam"][a], data[blk][a]):
                key = f"{d}__greedy"
                if rb is None or ro is None or key not in rb[1] or key not in ro[1]:
                    continue
                diff = np.abs(rb[1][key] - ro[1][key])
                n, diff_count, max_diff = n + len(diff), diff_count + int((diff > TIE_TOL).sum()), max(max_diff, diff.max())
            where = "phase 1 eval (k = 10 = 0.5 × 20)" if blk == "phase1" else "kfrac05 eval"
            lines.append(f"| {ARCHS[a][0]} | {DATASETS[d]} | {where} | {n or 'missing'} | "
                         f"{diff_count if n else '—'} | {f'{max_diff:.2e}' if n else '—'} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- figure

def plot_curves(histories, figs_dir):
    """Validation gap (top) and loss (bottom) per epoch; BatchNorm (phase 1) left, LayerNorm right.

    Line = mean over seeds, band = min–max over seeds; each line is labelled at its end.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, ln in enumerate((False, True)):
        for row, key in enumerate(("val_gap_greedy", "val_loss")):
            ax = axes[row, col]
            ends = []
            for a, (name, color) in ARCHS.items():
                hs = histories[ln, a]
                if not hs or len({len(h) for h in hs}) != 1:
                    continue
                vals = np.array([[h[key] for h in hist] for hist in hs])
                epochs = np.arange(1, vals.shape[1] + 1)
                ax.fill_between(epochs, vals.min(0), vals.max(0), color=color, alpha=0.15, linewidth=0)
                ax.plot(epochs, vals.mean(0), color=color, linewidth=2, label=f"{name} (n={len(hs)})")
                ends.append([vals.mean(0)[-1], name])
            summarize.style_axes(ax)
            if not ends:
                continue
            ends.sort()
            lo, hi = ax.get_ylim()
            for i in range(1, len(ends)):
                ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.05 * (hi - lo))
            for y, name in ends:
                ax.text(20.4, y, name, va="center", color=INK, fontsize=8)
            ax.set_xlim(0.5, 23.5)
            ax.set_xticks([1, 5, 10, 15, 20])
        axes[0, col].set_title(("LayerNorm retraining" if ln else "Phase 1 (BatchNorm)") + " — TSP-20 validation",
                               color=INK, fontsize=10, loc="left")
        axes[1, col].set_xlabel("Epoch", color=INK)
    axes[0, 0].set_ylabel("Validation gap, greedy (%)", color=INK)
    axes[1, 0].set_ylabel("Validation loss (weighted BCE)", color=INK)
    axes[0, 1].legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(0.86, 1.0))
    fig.suptitle("Validation curves: mean over 3 seeds, band = min–max", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(figs_dir, exist_ok=True)
    summarize.save(fig, figs_dir, "tagcn_curves")


# ---------------------------------------------------------------- report

DECISIONS = [
    "Layer: `torch_geometric.nn.TAGConv(64, 64, K)` (installed PyG 2.8.1 has it), i.e. Σ_{k=0..K} Â^k H W_k + b with "
    "Â = D^-1/2 A D^-1/2 of the k-NN graph *without* self-loops (PyG default; the k = 0 term already is the node "
    "itself). Like the other encoders, each layer is wrapped as h ← h + ReLU(Norm(TAGConv(h))) — residual + "
    "BatchNorm/LayerNorm — instead of the bare ReLU(Σ Â^k H W_k), so the only difference from the GCN encoder is the "
    "convolution. Like GCN, TAGCN does not see the edge lengths in the encoder (only the head does, through e_ij).",
    "Layers: round(6 / K), so the reach L × K is exactly 6 hops for K = 2 (L = 3) and K = 3 (L = 2), as the 6 "
    "one-hop layers of the other models. Everything else is the phase 1 protocol (H = 64, 20 epochs, batch 64, "
    "Adam 1e-3 + cosine, pos_weight from the data, selection by greedy val gap, seeds 0/1/2).",
    "LayerNorm retraining uses `--extra-checkpoints`; `last.pt` (epoch 20) is evaluated, as for the other models "
    "(`results/phase2_checks/ln/last_kfrac05/`). The best-val-gap `best.pt` is evaluated with k = 10 at the end of "
    "training (in the run folder) but not used in the tables.",
    "Beam search uses the phase 1 checkpoints (best.pt), like the `phase1` configuration of the beam report.",
    "All new outputs are inside `results/runs/tagcn_*` (evaluations in subfolders of the run they evaluate); "
    "nothing earlier was recomputed or rewritten.",
    "Statistical rule (same as the corrected beam report): two-sided Wilcoxon per seed (ties dropped), max p over "
    "the 3 seeds, Bonferroni within each family; the medians of the seeds must agree in sign, a seed with a zero "
    f"median (|median| ≤ {TIE_TOL:g} pp) is neutral, and if every median is zero the direction comes from the mean.",
]


def build_report(results, figs_dir=None):
    data = load_all(results)
    baselines = batch2.load_baselines(results)
    base = checks.load_base_per_instance(results)
    histories = load_histories(results)
    last, last_k10, rerun = load_last(results)
    t_kfrac, _ = table_tests(kfrac_greedy_tests(data, last, base))
    if figs_dir:
        plot_curves(histories, figs_dir)
    t1, _ = table_tests(phase1_tests(data, base))
    t2, _ = table_tests(beam_tests(data, base))
    notes_path = os.path.join(results, "tagcn_notes.md")
    notes = open(notes_path).read().strip() if os.path.exists(notes_path) else "(to be written after reading the results)"
    missing = [BLOCKS[b][3](a, s) for b in BLOCKS for a in TAGCN for s, r in zip(SEEDS, data[b][a]) if r is None]
    missing += [f"{CKPT_RUN(a, s)}/last_beam" for a in TAGCN for s, r in zip(SEEDS, last[a]) if r is None]
    missing += [CKPT_RUN(a, s) for a in TAGCN for s, r in zip(SEEDS, rerun[a]) if r is None]
    missing += [f"{CKPT_RUN(a, s)}/last_k10" for a in TAGCN for s, r in zip(SEEDS, last_k10[a]) if r is None]
    lines = [
        "# TAGCN — fourth encoder, phase 1 and phase 2 protocol",
        "",
        f"Generated by `scripts/tagcn_report.py` on {datetime.date.today().isoformat()}. "
        "TAGCN (Topology Adaptive Graph Convolutional Networks, Du, Zhang, Wu, Moura & Kar, 2017), K = 2 and K = 3, "
        "against GCN, GATv2 and Gated GCN with the same protocol. Gaps to LKH in %, mean over seeds 0, 1, 2; "
        "baselines are deterministic. The GCN / GATv2 / Gated GCN numbers are the earlier results, not recomputed.",
        "",
        "## Three things that look most important or suspicious",
        "",
        notes,
        "",
        "## Table 1 — main table (mean gap %, *without 2-opt / with 2-opt*)",
        "",
        "Phase 1: k = 10 test graphs. k=0.5n: the same phase 1 checkpoints on test graphs with k = 0.5 n "
        "(25 on TSP-50, 50 on TSP-100). LayerNorm: retrained with LayerNorm, last epoch, k = 0.5 n. B=100: beam search "
        "of width 100 on the phase 1 checkpoints, k = 0.5 n; baseline cells there are the plain baselines. "
        "Bold = best model in that column and variant (best.pt rows only). *last.pt, rerun*: last epoch of the "
        "BatchNorm reruns (Table 7); its phase 1 columns use k = 10 graphs (`last_k10/`), the others k = 0.5 n; "
        "LayerNorm is a different training, so those cells are empty.",
        "",
        table_main(data, baselines, last, last_k10),
        "",
        "## Table 2 — parameters",
        "",
        "Counted by building each model with H = 64. *Encoder layers only* = convolutions + their normalizations "
        "(GATv2 and Gated GCN layers include their own). The rest (embeddings 2→64 and 1→64, head 192→64→64→1) is "
        "the same 16,897 parameters for every model.",
        "",
        table_params(results, histories),
        "",
        "## Table 3 — detailed results (mean ± std over seeds)",
        "",
        *[x for blk in BLOCKS for x in (f"### {BLOCKS[blk][0]}", "", table_block(data, baselines, blk, results), "")],
        "## Validation curves",
        "",
        "![Validation curves](figs/tagcn_curves.png)",
        "",
        "`figs/tagcn_curves.png`: validation gap (greedy) and validation loss per epoch on TSP-20, phase 1 runs "
        "(BatchNorm) and LayerNorm retraining; mean over seeds with a min–max band. The values are in each run's "
        "`history.json`.",
        "",
        "## Table 4 — phase 1: TAGCN against the baselines (paired Wilcoxon)",
        "",
        "Diff. = TAGCN − baseline with the same decoder (negative = TAGCN better). k = 10 test graphs. "
        f"*Equal* = difference ≤ {TIE_TOL:g} pp.",
        "",
        t1,
        "",
        "## Table 5 — beam search: TAGCN B=100 + 2-opt against the baselines + 2-opt",
        "",
        "k = 0.5 n test graphs. The GCN / GATv2 / Gated GCN versions of these tests are in "
        "`phase2_beam_report.md` (Table 3, Bonferroni over 54 tests there).",
        "",
        t2,
        "",
        "## Table 5b — greedy with k = 0.5 n against Christofides (no 2-opt)",
        "",
        "Diff. = model greedy − Christofides (negative = model better), no 2-opt on either side, k = 0.5 n test "
        "graphs. TAGCN best.pt (first runs), TAGCN last.pt (BatchNorm reruns) and GCN best.pt "
        "(`phase2_batch2/kfrac05/`). Bonferroni over this family only.",
        "",
        t_kfrac,
        "",
        "## Table 6 — consistency check: TAGCN greedy in the beam run vs. earlier evaluation of the same checkpoint",
        "",
        consistency(data),
        "",
        "## Table 7 — TAGCN last.pt (BatchNorm reruns), k = 0.5 n",
        "",
        "The first TAGCN batch trained the BatchNorm runs without `--extra-checkpoints`, so it has no last.pt. "
        "The 6 runs were trained again with the same code, seeds and protocol into `results/runs/tagcn_k<K>_ckpt_seed<s>/` "
        "and their last.pt (epoch 20) was evaluated (greedy and B=100, without / with 2-opt). Mean ± std over seeds.",
        "",
        table_last(last, rerun, baselines),
        "",
        "## Table 8 — best.pt of the first run vs. the rerun",
        "",
        "Same code, seed, data and batch order. The two runs are still not bit-for-bit identical: the message "
        "sums on the GPU (`scatter` with atomic additions) are done in a non-deterministic order, so the floating "
        "point roundings differ from run to run, and the small differences grow during training. Because the "
        "validation gap is nearly flat over the epochs (see note 2), such differences can change which epoch is "
        "selected. Hence last.pt (Table 7) and the best.pt of Tables 1–5 come from different, equivalent runs.",
        "",
        table_rerun(data, rerun, results),
        "",
        "## Decisions",
        "",
        *[f"- {d}" for d in DECISIONS],
        "",
        "## Missing results",
        "",
        ("None." if not missing else "Not found (failed or not yet run): " + ", ".join(missing) + "."),
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TAGCN report.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/tagcn_report.md")
    args = parser.parse_args()
    report = build_report(args.results, figs_dir=os.path.join(args.results, "figs"))
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"saved: {args.out} and {args.results}/figs/tagcn_curves.(png|pdf)")


if __name__ == "__main__":
    main()
