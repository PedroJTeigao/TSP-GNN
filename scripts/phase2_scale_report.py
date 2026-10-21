"""Phase 2 scale report: how far does the whole-instance GNN go? Gap and time vs. n, from TSP-20 to TSP-10000.

Reads (written by scripts/run_phase2_scale.sh):
    data/tsp<n>_test.npz                               test sets (LKH times for the new ones)
    results/phase2_scale/baselines/tsp<n>_test.*       greedy edge and Christofides (± 2-opt per k-mode)
    results/phase2_scale/<run>/<k-mode>/tsp<n>_test.*  models (best.pt): greedy, beam 10 / 100, ± 2-opt
and, for the consistency check, the earlier dense evaluations with k = 0.5 n:
    results/phase2_beam/phase1/<m>_seed<s>/ and results/runs/tagcn_k3_seed<s>/beam/.
Writes results/phase2_scale_report.md and results/figs/phase2_scale_{gap,time}.{png,pdf}.
Missing results are reported as "missing" instead of stopping the report. Hand-written notes (the
"three lines") are read from results/phase2_scale_notes.md if that file exists.

Example:
    python scripts/phase2_scale_report.py
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

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarize = _load("summarize")
beam = _load("phase2_beam_report")
INK, MUTED, ALPHA = summarize.INK, summarize.MUTED, summarize.ALPHA

SIZES = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
SEEDS = (0, 1, 2)
# Models, in table order: key -> (name, colour). Same colours as in the TAGCN report.
MODELS = {"gcn": ("GCN", "#2a78d6"), "gat": ("GATv2", "#eb6834"), "gated": ("Gated GCN", "#1baf7a"),
          "tagcn_k3": ("TAGCN K=3", "#e87ba4")}
K_MODES = {"k50": "k = 50", "khalf100": "k = min(0.5 n, 100)"}
MIN_TEST = 20           # Wilcoxon only where there are at least this many instances
# Paired comparisons (model method, baseline method, label); "{k}" is the k-mode of the model.
COMPARISONS = [("greedy", "greedy", "greedy vs. greedy edge"),
               ("greedy+2opt", "greedy+2opt@{k}", "greedy + 2-opt vs. greedy edge + 2-opt"),
               ("beam10+2opt", "greedy+2opt@{k}", "B=10 + 2-opt vs. greedy edge + 2-opt"),
               ("greedy", "christofides", "greedy vs. Christofides"),
               ("greedy+2opt", "christofides+2opt@{k}", "greedy + 2-opt vs. Christofides + 2-opt"),
               ("beam10+2opt", "christofides+2opt@{k}", "B=10 + 2-opt vs. Christofides + 2-opt")]
# Same tour <=> same gap up to this tolerance (pp): the earlier evaluations divided by the LKH length stored as
# float32 (Data.opt_len), the scale experiment by the float64 one, so identical tours differ by ~1e-6 pp.
SAME_TOL = 1e-4
OLD_DENSE = {m: (lambda s, m=m: f"phase2_beam/phase1/{m}_seed{s}") for m in ("gcn", "gat", "gated")}
OLD_DENSE["tagcn_k3"] = lambda s: f"runs/tagcn_k3_seed{s}/beam"


def ds(n):
    return f"tsp{n}_test"


# ---------------------------------------------------------------- loading

def _read(path_no_ext):
    """(summary dict, per-instance dict) of a .json/.npz pair, or None if the .json is missing."""
    if not os.path.exists(path_no_ext + ".json"):
        return None
    with open(path_no_ext + ".json") as f:
        summary = json.load(f)
    with np.load(path_no_ext + ".npz") as z:
        return summary, {k: z[k] for k in z.files}


def load_all(results, data_dir):
    """{"models": {(m, mode, n): [result per seed or None]}, "base": {n: result or None}, "data": {n: info}}."""
    root = os.path.join(results, "phase2_scale")
    models = {(m, mode, n): [_read(os.path.join(root, f"{m}_seed{s}", mode, ds(n))) for s in SEEDS]
              for m in MODELS for mode in K_MODES for n in SIZES}
    base = {n: _read(os.path.join(root, "baselines", ds(n))) for n in SIZES}
    data = {}
    for n in SIZES:
        path = os.path.join(data_dir, ds(n) + ".npz")
        if os.path.exists(path):
            with np.load(path) as z:
                data[n] = {"num": len(z["lengths"]), "seed": int(z["seed"]) if "seed" in z.files else None,
                           "lkh_runs": int(z["lkh_runs"]) if "lkh_runs" in z.files else 10,
                           "lkh_s": z["lkh_seconds"] if "lkh_seconds" in z.files else None}
    return {"models": models, "base": base, "data": data}


def complete(runs):
    return all(r is not None for r in runs)


def seed_means(runs, method):
    """Mean gap of each seed (NaN-free means, 2-opt timeouts excluded), or None if anything is missing."""
    if not complete(runs) or any(method not in r[0] or r[0][method]["gap_mean"] is None for r in runs):
        return None
    return [r[0][method]["gap_mean"] for r in runs]


def fmt(values, digits=2):
    if values is None:
        return "missing"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}" if len(values) > 1 \
        else f"{values[0]:.{digits}f}"


def base_gap(base, n, method):
    b = base.get(n)
    if b is None:
        return None
    entry = b[0].get(method)
    return None if entry is None or entry["gap_mean"] is None else entry["gap_mean"]


# ---------------------------------------------------------------- tables

def table_data(data, base):
    lines = ["| n | Instances | Seed | LKH runs | LKH time per instance (s): mean / max |", "|---|---|---|---|---|"]
    for n in SIZES:
        if n not in data:
            lines.append(f"| {n} | missing | | | |")
            continue
        d = data[n]
        t = (f"{d['lkh_s'].mean():.1f} / {d['lkh_s'].max():.1f}" if d["lkh_s"] is not None
             else "— (phase 1 set, not timed)")
        lines.append(f"| {n} | {d['num']} | {d['seed']} | {d['lkh_runs']} | {t} |")
    return "\n".join(lines)


def table_gaps(R, mode, methods):
    """Mean gap ± std over seeds; one column per n. methods: list of (model method, label)."""
    head = "| Method | " + " | ".join(str(n) for n in SIZES) + " |"
    lines = [head, "|---|" + "---|" * len(SIZES)]
    base_rows = [("greedy", "Greedy edge"), (f"greedy+2opt@{mode}", "Greedy edge + 2-opt"),
                 ("christofides", "Christofides"), (f"christofides+2opt@{mode}", "Christofides + 2-opt")]
    for key, label in base_rows:
        cells = []
        for n in SIZES:
            g = base_gap(R["base"], n, key)
            cells.append(f"{g:.2f}" if g is not None else ("—" if R["base"].get(n) is not None else "missing"))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for m, (name, _) in MODELS.items():
        for method, label in methods:
            cells = []
            for n in SIZES:
                runs = R["models"][m, mode, n]
                if complete(runs) and method not in runs[0][0]:
                    cells.append("—")                      # decoder not run at this size
                else:
                    cells.append(fmt(seed_means(runs, method)))
            lines.append(f"| {name} {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_tests(R):
    """{(n, mode, m, comparison index): result} for every n with >= MIN_TEST instances; Bonferroni per n."""
    tests = {}
    for n in SIZES:
        b = R["base"].get(n)
        if b is None or b[0]["meta"]["num"] < MIN_TEST:
            continue
        family = []
        for mode in K_MODES:
            for m in MODELS:
                runs = R["models"][m, mode, n]
                if not complete(runs):
                    continue
                for ci, (meth, ref, _) in enumerate(COMPARISONS):
                    ref = ref.format(k=mode)
                    if meth not in runs[0][1] or ref not in b[1]:
                        continue
                    family.append(((n, mode, m, ci), [r[1][meth] for r in runs], b[1][ref]))
        threshold = ALPHA / max(len(family), 1)
        for key, a_list, ref in family:
            tests[key] = {**beam.paired(a_list, [ref] * len(a_list), threshold), "threshold": threshold,
                          "family": len(family)}
    return tests


def table_tests(R, tests, ci):
    """One comparison: rows = model × k-mode, columns = n (only sizes with tests)."""
    sizes = [n for n in SIZES if any(k[0] == n for k in tests)]
    if not sizes:
        return "No size with at least 20 instances has complete results yet."
    lines = ["| Model | k | " + " | ".join(str(n) for n in sizes) + " |", "|---|---|" + "---|" * len(sizes)]
    for mode, mode_label in K_MODES.items():
        for m, (name, _) in MODELS.items():
            cells = []
            for n in sizes:
                t = tests.get((n, mode, m, ci))
                if t is None:
                    cells.append("—")
                    continue
                v = f"**{t['verdict']}**" if t["verdict"] != "n.s." else "n.s."
                cells.append(f"{v} {t['mean_diff']:+.2f} (p {t['p_max']:.1e})")
            lines.append(f"| {name} | {mode_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_small(R):
    """Sizes with fewer than MIN_TEST instances: mean difference and instances where the model is better."""
    sizes = [n for n in SIZES if R["base"].get(n) is not None and R["base"][n][0]["meta"]["num"] < MIN_TEST]
    if not sizes:
        return "No size with fewer than 20 instances has results yet."
    lines = ["| Model | k | Comparison | " + " | ".join(str(n) for n in sizes) + " |",
             "|---|---|---|" + "---|" * len(sizes)]
    for mode, mode_label in K_MODES.items():
        for m, (name, _) in MODELS.items():
            for meth, ref, label in COMPARISONS[:2]:
                cells = []
                for n in sizes:
                    runs, b = R["models"][m, mode, n], R["base"][n]
                    ref_k = ref.format(k=mode)
                    if not complete(runs) or meth not in runs[0][1] or ref_k not in b[1]:
                        cells.append("missing")
                        continue
                    diff = np.concatenate([r[1][meth] - b[1][ref_k] for r in runs])
                    ok = ~np.isnan(diff)
                    cells.append(f"{np.mean(diff[ok]):+.2f} ({int((diff[ok] < 0).sum())}/{int(ok.sum())} better)"
                                 if ok.any() else "timeout")
                lines.append(f"| {name} | {mode_label} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def crossover(R, tests):
    """Per model and k-mode: up to which n the model beats greedy edge (+ 2-opt), by mean and by test."""
    lines = ["| Model | k | Decoder | Mean gap below greedy edge up to n = | First n where it is not | "
             "Last n significantly better | First n significantly worse |",
             "|---|---|---|---|---|---|---|"]
    for mode, mode_label in K_MODES.items():
        for m, (name, _) in MODELS.items():
            for ci, (meth, ref, _) in enumerate(COMPARISONS[:2]):
                below, first_not = None, None
                for n in SIZES:
                    g = seed_means(R["models"][m, mode, n], meth)
                    bg = base_gap(R["base"], n, ref.format(k=mode))
                    if g is None or bg is None:
                        continue
                    if np.mean(g) < bg and first_not is None:
                        below = n
                    elif first_not is None:
                        first_not = n
                sig = [n for n in SIZES if tests.get((n, mode, m, ci), {}).get("verdict") == "better"]
                worse = [n for n in SIZES if tests.get((n, mode, m, ci), {}).get("verdict") == "worse"]
                lines.append(f"| {name} | {mode_label} | {meth.replace('+2opt', ' + 2-opt')} | {below or 'never'} | "
                             f"{first_not or '—'} | {max(sig) if sig else 'none'} | {min(worse) if worse else 'none'} |")
    return "\n".join(lines)


def _mean_ms(runs, key):
    if not complete(runs) or key not in runs[0][1]:
        return None
    return float(np.mean([np.nanmean(r[1][key]) for r in runs]))


def table_times(R, mode):
    """Mean time per instance (ms) of each step; model steps averaged over the 3 seeds."""
    rows = [("k-NN graph (KD-tree)", lambda n: _mean_ms(R["models"]["gcn", mode, n], "graph__ms"))]
    for m, (name, _) in MODELS.items():
        rows.append((f"Network: {name}", lambda n, m=m: _mean_ms(R["models"][m, mode, n], "gnn__ms")))
    avg = lambda key: (lambda n: (lambda v: float(np.mean(v)) if v else None)(
        [x for x in (_mean_ms(R["models"][m, mode, n], key) for m in MODELS) if x is not None]))
    rows += [("Greedy decoding (mean of models)", avg("greedy__ms")),
             ("2-opt after greedy (mean of models)", avg("greedy+2opt__ms")),
             ("Beam B=10 (mean of models)", avg("beam10__ms")),
             ("Beam B=100 (mean of models)", avg("beam100__ms"))]
    bms = lambda key: (lambda n: float(np.mean(R["base"][n][1][key])) if R["base"].get(n) is not None
                       and key in R["base"][n][1] else None)
    rows += [("Greedy edge", bms("greedy__ms")), ("2-opt after greedy edge", bms(f"greedy+2opt@{mode}__ms")),
             ("Christofides", bms("christofides__ms")),
             ("LKH (label)", lambda n: 1000 * float(R["data"][n]["lkh_s"].mean())
              if n in R["data"] and R["data"][n]["lkh_s"] is not None else None)]
    lines = ["| Step | " + " | ".join(str(n) for n in SIZES) + " |", "|---|" + "---|" * len(SIZES)]
    for label, fn in rows:
        cells = []
        for n in SIZES:
            v = fn(n)
            cells.append("—" if v is None else (f"{v:.1f}" if v < 100 else f"{v:,.0f}"))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines), rows


def table_memory(R):
    lines = ["| Model | k | " + " | ".join(str(n) for n in SIZES) + " |", "|---|---|" + "---|" * len(SIZES)]
    for mode, mode_label in K_MODES.items():
        for m, (name, _) in MODELS.items():
            cells = []
            for n in SIZES:
                runs = R["models"][m, mode, n]
                if not complete(runs):
                    cells.append("missing")
                    continue
                peak = max(r[0]["meta"]["peak_gpu_gib"] for r in runs)
                dev = {r[0]["meta"]["gnn_device"] for r in runs}
                cells.append(f"{1024 * peak:.0f} MiB" + ("" if dev == {"cuda"} else f" ({'/'.join(sorted(dev))})"))
            lines.append(f"| {name} | {mode_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def timeouts(R):
    out = []
    for (m, mode, n), runs in R["models"].items():
        for s, r in zip(SEEDS, runs):
            if r is None:
                continue
            for key, v in r[0].items():
                if isinstance(v, dict) and v.get("timeouts"):
                    out.append(f"{m}_seed{s} {mode} TSP-{n} {key}: {v['timeouts']} instance(s)")
    for n, b in R["base"].items():
        if b is not None:
            for key, v in b[0].items():
                if isinstance(v, dict) and v.get("timeouts"):
                    out.append(f"baselines TSP-{n} {key}: {v['timeouts']} instance(s)")
    return out


def consistency(R, results):
    """k = min(0.5 n, 100) on TSP-20/50/100 equals the earlier k = 0.5 n evaluations (dense decoders, full 2-opt)."""
    lines = ["| Model | n | Greedy: identical instances | B=10: identical instances | "
             "+ 2-opt mean gap, neighbour lists / full 2-opt |", "|---|---|---|---|---|"]
    for m, (name, _) in MODELS.items():
        for n in (20, 50, 100):
            runs = R["models"][m, "khalf100", n]
            old = [os.path.join(results, OLD_DENSE[m](s), "eval_per_instance.npz") for s in SEEDS]
            if not complete(runs) or not all(os.path.exists(p) for p in old):
                lines.append(f"| {name} | {n} | missing | missing | missing |")
                continue
            same = {"greedy": [0, 0], "beam10": [0, 0]}
            new2, old2 = [], []
            for r, p in zip(runs, old):
                with np.load(p) as z:
                    for key in same:
                        a, b = r[1][key], z[f"{ds(n)}__{key}"]
                        same[key][0] += int(np.sum(np.abs(a - b) <= SAME_TOL)); same[key][1] += len(a)
                    new2.append(np.nanmean(r[1]["greedy+2opt"])); old2.append(z[f"{ds(n)}__greedy+2opt"].mean())
            lines.append(f"| {name} | {n} | {same['greedy'][0]}/{same['greedy'][1]} | {same['beam10'][0]}/"
                         f"{same['beam10'][1]} | {np.mean(new2):.2f} / {np.mean(old2):.2f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- figures

def _label_ends(ax, ends, x):
    """Direct labels at the right end of the lines, pushed apart so they do not overlap (log-y aware)."""
    if not ends:
        return
    ends.sort()
    lo, hi = ax.get_ylim()
    step = 0.045 * (hi - lo)
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + step)
    for y, name in ends:
        ax.text(x, y, name, va="center", color=INK, fontsize=7)


def plot_gaps(R, figs_dir):
    """Gap vs. n (log x): rows = k-mode, columns = greedy / + 2-opt / B=10 + 2-opt. Line = mean over seeds."""
    cols = [("greedy", "greedy", "christofides", "Greedy decoding"),
            ("greedy+2opt", "greedy+2opt@{k}", "christofides+2opt@{k}", "Greedy + 2-opt"),
            ("beam10+2opt", "greedy+2opt@{k}", "christofides+2opt@{k}", "Beam B=10 + 2-opt (n ≤ 1000)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for row, mode in enumerate(K_MODES):
        for col, (meth, gref, cref, title) in enumerate(cols):
            ax = axes[row, col]
            ends = []
            for key, label, style, color in ((gref.format(k=mode), "Greedy edge" + (" + 2-opt" if col else ""), "--", INK),
                                             (cref.format(k=mode), "Christofides" + (" + 2-opt" if col else ""), ":", MUTED)):
                pts = [(n, base_gap(R["base"], n, key)) for n in SIZES]
                pts = [(n, g) for n, g in pts if g is not None]
                if col == 2:
                    pts = [(n, g) for n, g in pts if n <= 1000]
                if pts:
                    ax.plot(*zip(*pts), style, color=color, linewidth=2, marker="o", markersize=4, label=label)
                    ends.append([pts[-1][1], label, pts[-1][0]])
            for m, (name, color) in MODELS.items():
                pts = []
                for n in SIZES:
                    v = seed_means(R["models"][m, mode, n], meth)
                    if v is not None:
                        pts.append((n, np.mean(v), np.min(v), np.max(v)))
                if not pts:
                    continue
                x, y, lo, hi = map(np.array, zip(*pts))
                ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
                ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=name)
                ends.append([y[-1], name, x[-1]])
            ax.set_xscale("log")
            summarize.style_axes(ax)
            ax.set_xticks(SIZES)
            ax.set_xticklabels([str(n) for n in SIZES], fontsize=7)
            ax.minorticks_off()
            ax.set_xlim(15, 10000 * 3.2)
            xmax = max((e[2] for e in ends), default=10000)
            _label_ends(ax, [[e[0], e[1]] for e in ends], xmax * 1.12)
            ax.set_title(f"{title} — {K_MODES[mode]}", color=INK, fontsize=9, loc="left")
            if col == 0:
                ax.set_ylabel("Optimality gap to LKH (%)", color=INK)
            if row == 1:
                ax.set_xlabel("n (cities, log scale)", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Gap vs. n: mean over 3 seeds (band = min–max); models trained on TSP-20 only",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(figs_dir, exist_ok=True)
    summarize.save(fig, figs_dir, "phase2_scale_gap")


def plot_times(R, figs_dir, mode="khalf100"):
    """Time per instance vs. n (log-log), one panel per step (small multiples)."""
    _, rows = table_times(R, mode)
    named = dict(rows)
    panels = [("k-NN graph (KD-tree)", ["k-NN graph (KD-tree)"]),
              ("Network forward pass", [f"Network: {name}" for name, _ in MODELS.values()]),
              ("Decoding", ["Greedy decoding (mean of models)", "Beam B=10 (mean of models)"]),
              ("2-opt (neighbour lists)", ["2-opt after greedy (mean of models)", "2-opt after greedy edge"]),
              ("References", ["Greedy edge", "Christofides", "LKH (label)"])]
    colors = {f"Network: {name}": c for name, c in MODELS.values()}
    styles = ["-", "--", ":"]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4), sharey=True)
    for ax, (title, labels) in zip(axes, panels):
        ends = []
        for i, label in enumerate(labels):
            pts = [(n, named[label](n)) for n in SIZES]
            pts = [(n, v) for n, v in pts if v is not None and v > 0]
            if not pts:
                continue
            ax.plot(*zip(*pts), styles[i % 3] if label not in colors else "-", color=colors.get(label, INK),
                    linewidth=2, marker="o", markersize=4, label=label)
            short = label.replace(" (mean of models)", "").replace("Network: ", "")
            ends.append([np.log10(pts[-1][1]), short, pts[-1][0]])
        ax.set_xscale("log"); ax.set_yscale("log")
        summarize.style_axes(ax)
        ax.set_xticks([20, 100, 1000, 10000]); ax.set_xticklabels(["20", "100", "1000", "10000"])
        ax.minorticks_off()
        ax.set_xlim(15, 10000 * 4)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("n (log scale)", color=INK)
        if ends:   # labels in log space
            ends.sort()
            for i in range(1, len(ends)):
                ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.35)
            for y, name, x in ends:
                ax.text(x * 1.15, 10 ** y, name, va="center", color=INK, fontsize=7)
    axes[0].set_ylabel("Time per instance (ms, log scale)", color=INK)
    fig.suptitle(f"Time per instance vs. n ({K_MODES[mode]}; model steps averaged over 3 seeds)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    summarize.save(fig, figs_dir, "phase2_scale_time")


# ---------------------------------------------------------------- report

def build_report(results, data_dir="data", figs_dir=None):
    R = load_all(results, data_dir)
    figs_dir = figs_dir or os.path.join(results, "figs")
    tests = run_tests(R)
    plot_gaps(R, figs_dir)
    plot_times(R, figs_dir)
    notes_path = os.path.join(results, "phase2_scale_notes.md")
    notes = open(notes_path).read().strip() if os.path.exists(notes_path) else "(to be written after reading the results)"
    families = sorted({(k[0], t["family"], t["threshold"]) for k, t in tests.items()})
    missing = [f"{m}_seed{s}/{mode}/{ds(n)}" for (m, mode, n), runs in R["models"].items()
               for s, r in zip(SEEDS, runs) if r is None]
    missing += [f"baselines/{ds(n)}" for n, b in R["base"].items() if b is None]
    commits = sorted({r[0]["meta"]["git_commit"] for runs in R["models"].values() for r in runs if r is not None}
                     | {b[0]["meta"]["git_commit"] for b in R["base"].values() if b is not None})
    gaps_main = [("greedy", "greedy"), ("greedy+2opt", "greedy + 2-opt")]
    gaps_beam = [("beam10", "B=10"), ("beam10+2opt", "B=10 + 2-opt"), ("beam100", "B=100"), ("beam100+2opt", "B=100 + 2-opt")]
    t_main, _ = table_times(R, "khalf100")
    t_k50, _ = table_times(R, "k50")
    to = timeouts(R)
    lines = [
        "# Phase 2 — scale experiment: how far does the whole-instance GNN go?",
        "",
        f"Generated by `scripts/phase2_scale_report.py` on {datetime.date.today().isoformat()} "
        f"(results from commit(s) {', '.join(commits) or '—'}).",
        "",
        "## Setup",
        "",
        "- **No retraining.** GCN, GATv2 and Gated GCN (phase 1) and TAGCN K=3, BatchNorm, best.pt, seeds 0–2, all "
        "trained on TSP-20 with k = 10 and raw distances.",
        "- **Test sets.** TSP-20/50/100 are the phase 1 test sets; TSP-200 … TSP-10000 are new (uniform in [0,1]², "
        "LKH labels from the coordinates, `scripts/make_scale_data.py`). LKH uses 10 runs up to n = 1000 (as the "
        "phase 1 sets) and 1 run above, so the reference for n ≥ 2000 is slightly weaker; its time per instance is "
        "in Table 1. TSP-10000: the 3 instances were solved in parallel, so 3 instances took the wall time of one.",
        "- **Graphs.** k-NN with a KD-tree (same edges as the phase 1 code): (a) k = 50; (b) k = min(0.5 n, 100) "
        "(= the earlier k = 0.5 n up to TSP-200). Never more than n − 1.",
        "- **Decoding.** Sparse versions of the phase 1 decoders that give the same tours (tests/test_scale.py; Table 9): "
        "greedy at every n, beam B=10 up to n = 1000, B=100 on TSP-200 only.",
        "- **2-opt.** Neighbour lists (the same k nearest neighbours) with don't-look bits: O(n·k) per sweep, "
        "limit 10 min per call (a timeout is reported, never hidden). It differs from the full 2-opt of phase 1, so "
        "every method here — baselines included — uses this 2-opt, with the k of the column.",
        "- **Baselines.** Greedy edge (+ 2-opt) at every n; Christofides (+ 2-opt) up to n = 1000 (networkx, needs "
        "the complete graph).",
        f"- **Tests.** Paired Wilcoxon per instance, where there are ≥ {MIN_TEST} instances (n ≤ 1000); one test per "
        "seed, max p over the 3 seeds, the seeds' medians must agree in sign (a zero median is neutral); Bonferroni "
        "within each size (family = every model × k × comparison at that n). Below 20 instances only the mean "
        "difference and the count of better instances are given.",
        "",
        "## Table 1 — Test sets and LKH time",
        "",
        table_data(R["data"], R["base"]),
        "",
        "## Table 2 — Gap (%) vs. n, greedy and greedy + 2-opt",
        "",
        "Mean ± std over the 3 seeds of each seed's mean gap. Baselines are deterministic.",
        "",
        *[x for mode, label in K_MODES.items() for x in (f"### {label}", "", table_gaps(R, mode, gaps_main), "")],
        "![Gap vs. n](figs/phase2_scale_gap.png)",
        "",
        "## Table 3 — Beam search (%)",
        "",
        *[x for mode, label in K_MODES.items() for x in (f"### {label}", "", table_gaps(R, mode, gaps_beam), "")],
        "## Table 4 — Paired tests against the baselines",
        "",
        "Cell: verdict, mean difference model − baseline (pp; negative = model better), max p over seeds. "
        "Bonferroni thresholds: " + ("; ".join(f"n = {n}: 0.05/{f} = {t:.2e}" for n, f, t in families) or "—") + ".",
        "",
        *[x for ci, (_, _, label) in enumerate(COMPARISONS) for x in (f"### {label}", "", table_tests(R, tests, ci), "")],
        "## Table 5 — Sizes with fewer than 20 instances (no test)",
        "",
        "Mean difference model − baseline (pp) and instances where the model is better, over the 3 seeds.",
        "",
        table_small(R),
        "",
        "## Table 6 — Where each model stops beating greedy edge",
        "",
        "By mean gap (all sizes) and by the Table 4 tests (n ≤ 1000).",
        "",
        crossover(R, tests),
        "",
        "## Table 7 — Time per instance (ms)",
        "",
        "Network: one instance at a time on the GPU (CUDA kernels warmed up), including the copy to the GPU. "
        "Decoding and 2-opt: one CPU core per instance (4 workers in parallel). 2-opt rows: the 2-opt alone.",
        "",
        "### k = min(0.5 n, 100)",
        "",
        t_main,
        "",
        "### k = 50",
        "",
        t_k50,
        "",
        "![Time vs. n](figs/phase2_scale_time.png)",
        "",
        "## Table 8 — Peak GPU memory of the network (max over seeds)",
        "",
        "PyTorch allocator peak during the forward passes of one test set (cap 5.5 GiB). A device other than "
        "cuda means the graph did not fit and the network ran on the CPU.",
        "",
        table_memory(R),
        "",
        "## Table 9 — Consistency with the earlier dense evaluations (k = 0.5 n)",
        "",
        "Same checkpoints and graphs as `phase2_beam` (GCN/GATv2/Gated) and the TAGCN `beam/` folders: the sparse "
        "greedy and beam decoders should give the same per-instance gaps (beam: up to float near-ties; "
        "\"identical\" = within 1e-4 pp, because the earlier runs divided by the LKH length in float32); the 2-opt "
        "is different by design (neighbour lists vs. full), so only its mean is compared.",
        "",
        consistency(R, results),
        "",
        "## 2-opt timeouts",
        "",
        "\n".join(f"- {x}" for x in to) if to else "None.",
        "",
        "## Three lines",
        "",
        notes,
        "",
        "## Missing results",
        "",
        "\n".join(f"- `{x}`" for x in missing) if missing else "None.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 scale report.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/phase2_scale_report.md")
    args = parser.parse_args()
    text = build_report(args.results, args.data)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
