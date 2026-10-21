"""Phase 3 window refinement report: re-ordering windows of the greedy edge + 2-opt tour.

Reads results/phase3_window/{tagcn_k3_seed<s>,lkh,2opt}/tsp<n>_test.{json,npz} (scripts/window_eval.py)
and, for the consistency check and context, results/phase2_scale/baselines/ and
results/phase2_scale/tagcn_k3_seed<s>/khalf100/. Writes results/phase3_window_report.md and
results/figs/phase3_window_gap.{png,pdf}. Missing results are reported as "missing". Hand-written notes
(the "three lines") are read from results/phase3_window_notes.md.

Example:
    python scripts/phase3_window_report.py
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

SIZES = (200, 500, 1000, 2000, 5000, 10000)
SEEDS = (0, 1, 2)
WIDTHS = (20, 30, 50)
# method -> (folders, label, colour); model methods have one folder per seed
METHODS = {"greedy": ([f"tagcn_k3_seed{s}" for s in SEEDS], "TAGCN K=3, greedy paths", "#2a78d6"),
           "beam10": ([f"tagcn_k3_seed{s}" for s in SEEDS], "TAGCN K=3, beam B=10 paths", "#eb6834"),
           "lkh": (["lkh"], "LKH on the window (upper bound)", "#1baf7a"),
           "2opt": (["2opt"], "2-opt on the window (control)", "#eda100")}
MIN_TEST = 20


def ds(n):
    return f"tsp{n}_test"


def _read(path_no_ext):
    if not os.path.exists(path_no_ext + ".json"):
        return None
    with open(path_no_ext + ".json") as f:
        summary = json.load(f)
    with np.load(path_no_ext + ".npz") as z:
        return summary, {k: z[k] for k in z.files}


def load_all(results):
    folders = sorted({f for fs, _, _ in METHODS.values() for f in fs})
    win = {(f, n): _read(os.path.join(results, "phase3_window", f, ds(n))) for f in folders for n in SIZES}
    base = {n: _read(os.path.join(results, "phase2_scale", "baselines", ds(n))) for n in SIZES}
    whole = {n: [_read(os.path.join(results, "phase2_scale", f"tagcn_k3_seed{s}", "khalf100", ds(n))) for s in SEEDS]
             for n in SIZES}
    return {"win": win, "base": base, "whole": whole}


def runs_of(R, method, n):
    """Per-folder results of a method at size n (one per seed for the model), or None if any is missing."""
    runs = [R["win"][f, n] for f in METHODS[method][0]]
    return runs if all(r is not None for r in runs) else None


def fmt(values, digits=2):
    if values is None:
        return "missing"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}" if len(values) > 1 \
        else f"{values[0]:.{digits}f}"


def per_run_mean(R, method, n, field, w):
    runs = runs_of(R, method, n)
    if runs is None:
        return None
    return [float(np.mean(r[1][f"{method}_w{w}__{field}"])) for r in runs]


def initial_gap(R, n):
    """Mean gap of the initial tour (any folder; they are all the same tour)."""
    for f in ("2opt", "lkh", "tagcn_k3_seed0"):
        r = R["win"].get((f, n))
        if r is not None:
            return float(np.mean(r[1]["initial__gap"]))
    return None


# ---------------------------------------------------------------- tables

def table_consistency(R):
    lines = ["| n | Instances | Initial tour = greedy edge + 2-opt of the scale experiment (all folders) |",
             "|---|---|---|"]
    for n in SIZES:
        b = R["base"].get(n)
        rs = [(f, r) for (f, m), r in R["win"].items() if m == n and r is not None]
        if b is None or not rs:
            lines.append(f"| {n} | missing | missing |")
            continue
        ref = b[1]["greedy+2opt@khalf100"]
        same = sum(int(np.sum(np.abs(r[1]["initial__gap"] - ref) <= 1e-6)) for _, r in rs)
        total = sum(len(ref) for _ in rs)
        lines.append(f"| {n} | {len(ref)} | {same}/{total} identical |")
    return "\n".join(lines)


def table_values(R, field, digits=2):
    lines = ["| Method | w | " + " | ".join(str(n) for n in SIZES) + " |", "|---|---|" + "---|" * len(SIZES)]
    if field == "gap":
        lines.append("| Greedy edge + 2-opt (initial tour) | — | " + " | ".join(
            f"{g:.2f}" if (g := initial_gap(R, n)) is not None else "missing" for n in SIZES) + " |")
        lines.append("| Whole instance TAGCN K=3 + 2-opt (scale experiment) | — | " + " | ".join(
            fmt([float(np.mean(w[1]["greedy+2opt"])) for w in R["whole"][n]]) if all(R["whole"][n]) else "missing"
            for n in SIZES) + " |")
    for m, (_, label, _) in METHODS.items():
        for w in WIDTHS:
            lines.append(f"| {label} | {w} | " + " | ".join(fmt(per_run_mean(R, m, n, field, w), digits)
                                                            for n in SIZES) + " |")
    return "\n".join(lines)


def table_share_of_lkh(R):
    """Share of the LKH window gain that each model variant gets (gain / LKH gain, same w)."""
    lines = ["| Method | w | " + " | ".join(str(n) for n in SIZES) + " |", "|---|---|" + "---|" * len(SIZES)]
    for m in ("greedy", "beam10", "2opt"):
        for w in WIDTHS:
            cells = []
            for n in SIZES:
                g, ref = per_run_mean(R, m, n, "gain", w), per_run_mean(R, "lkh", n, "gain", w)
                cells.append("missing" if g is None or ref is None or ref[0] <= 0 else f"{100 * np.mean(g) / ref[0]:.0f} %")
            lines.append(f"| {METHODS[m][1]} | {w} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_windows(R):
    lines = ["| Method | w | " + " | ".join(str(n) for n in SIZES) + " |", "|---|---|" + "---|" * len(SIZES)]
    for m, (_, label, _) in METHODS.items():
        for w in WIDTHS:
            cells = []
            for n in SIZES:
                runs = runs_of(R, m, n)
                if runs is None:
                    cells.append("missing")
                    continue
                imp = sum(r[1][f"{m}_w{w}__improved"].sum() for r in runs)
                tot = sum(r[1][f"{m}_w{w}__windows"].sum() for r in runs)
                passes = np.mean([r[1][f"{m}_w{w}__passes"].mean() for r in runs])
                cells.append(f"{100 * imp / tot:.1f} % ({passes:.1f} passes)")
            lines.append(f"| {label} | {w} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_times(R):
    lines = ["| Step | w | " + " | ".join(str(n) for n in SIZES) + " |", "|---|---|" + "---|" * len(SIZES)]

    def cell(v):
        return "missing" if v is None else (f"{v:.1f}" if v < 100 else f"{v:,.0f}")

    init = []
    for n in SIZES:
        r = R["win"].get(("2opt", n))
        init.append(cell(float(np.mean(r[1]["initial__ms"])) if r else None))
    lines.append("| Initial tour (greedy edge + 2-opt) | — | " + " | ".join(init) + " |")
    for m, (_, label, _) in METHODS.items():
        for w in WIDTHS:
            v = [per_run_mean(R, m, n, "ms", w) for n in SIZES]
            lines.append(f"| Refinement: {label} | {w} | " + " | ".join(cell(np.mean(x) if x else None) for x in v) + " |")
    for m in ("greedy", "beam10"):
        v = [per_run_mean(R, m, n, "ms_model", 20) for n in SIZES]
        lines.append(f"|   of which network ({m}) | 20 | " + " | ".join(cell(np.mean(x) if x else None) for x in v) + " |")
    return "\n".join(lines)


def run_tests(R):
    """{(n, method, w): result}: final gap vs. the initial tour (greedy edge + 2-opt), n with >= 20 instances."""
    tests = {}
    for n in SIZES:
        family = []
        for m in METHODS:
            runs = runs_of(R, m, n)
            if runs is None or runs[0][0]["meta"]["num"] < MIN_TEST:
                continue
            for w in WIDTHS:
                family.append(((n, m, w), [r[1][f"{m}_w{w}__gap"] for r in runs], [r[1]["initial__gap"] for r in runs]))
        for key, a, ref in family:
            tests[key] = {**beam.paired(a, ref, ALPHA / len(family)), "threshold": ALPHA / len(family),
                          "family": len(family)}
    return tests


def table_tests(tests):
    sizes = sorted({k[0] for k in tests})
    if not sizes:
        return "No size with at least 20 instances has complete results yet."
    lines = ["| Method | w | " + " | ".join(str(n) for n in sizes) + " |", "|---|---|" + "---|" * len(sizes)]
    for m, (_, label, _) in METHODS.items():
        for w in WIDTHS:
            cells = []
            for n in sizes:
                t = tests.get((n, m, w))
                if t is None:
                    cells.append("—")
                    continue
                v = f"**{t['verdict']}**" if t["verdict"] != "n.s." else "n.s."
                cells.append(f"{v} {t['mean_diff']:+.2f} (p {t['p_max']:.1e})")
            lines.append(f"| {label} | {w} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_small(R):
    sizes = [n for n in SIZES if (r := R["win"].get(("2opt", n))) is not None and r[0]["meta"]["num"] < MIN_TEST]
    if not sizes:
        return "No size with fewer than 20 instances has results yet."
    lines = ["| Method | w | " + " | ".join(str(n) for n in sizes) + " |", "|---|---|" + "---|" * len(sizes)]
    for m, (_, label, _) in METHODS.items():
        for w in WIDTHS:
            cells = []
            for n in sizes:
                runs = runs_of(R, m, n)
                if runs is None:
                    cells.append("missing")
                    continue
                gain = np.concatenate([r[1][f"{m}_w{w}__gain"] for r in runs])
                cells.append(f"{-gain.mean():+.2f} ({int((gain > 1e-9).sum())}/{len(gain)} better)")
            lines.append(f"| {label} | {w} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- figure

def plot_gaps(R, figs_dir):
    fig, axes = plt.subplots(1, len(WIDTHS), figsize=(15, 4.6), sharey=True)
    for ax, w in zip(axes, WIDTHS):
        ends = []
        pts = [(n, initial_gap(R, n)) for n in SIZES]
        pts = [(n, g) for n, g in pts if g is not None]
        if pts:
            ax.plot(*zip(*pts), "--", color=INK, linewidth=2, marker="o", markersize=4, label="Greedy edge + 2-opt")
            ends.append([pts[-1][1], "Greedy edge + 2-opt"])
        for m, (_, label, color) in METHODS.items():
            vals = [(n, per_run_mean(R, m, n, "gap", w)) for n in SIZES]
            vals = [(n, np.mean(v), min(v), max(v)) for n, v in vals if v is not None]
            if not vals:
                continue
            x, y, lo, hi = map(np.array, zip(*vals))
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=label)
            ends.append([y[-1], label.split(" (")[0].replace("TAGCN K=3, ", "")])
        ax.set_xscale("log")
        summarize.style_axes(ax)
        ax.set_xticks(SIZES); ax.set_xticklabels([str(n) for n in SIZES], fontsize=8)
        ax.minorticks_off()
        ax.set_xlim(150, 10000 * 4)
        if ends:
            ends.sort()
            lo_, hi_ = ax.get_ylim()
            for i in range(1, len(ends)):
                ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.05 * (hi_ - lo_))
            for y, name in ends:
                ax.text(10000 * 1.12, y, name, va="center", color=INK, fontsize=7)
        ax.set_title(f"Windows of w = {w} cities", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("n (cities, log scale)", color=INK)
    axes[0].set_ylabel("Gap to LKH (%)", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Window refinement of the greedy edge + 2-opt tour (model: mean over 3 seeds, band = min–max)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(figs_dir, exist_ok=True)
    summarize.save(fig, figs_dir, "phase3_window_gap")


# ---------------------------------------------------------------- report

def build_report(results, figs_dir=None):
    R = load_all(results)
    figs_dir = figs_dir or os.path.join(results, "figs")
    tests = run_tests(R)
    plot_gaps(R, figs_dir)
    notes_path = os.path.join(results, "phase3_window_notes.md")
    notes = open(notes_path).read().strip() if os.path.exists(notes_path) else "(to be written after reading the results)"
    families = sorted({(k[0], t["family"], t["threshold"]) for k, t in tests.items()})
    missing = [f"phase3_window/{f}/{ds(n)}" for (f, n), r in R["win"].items() if r is None]
    commits = sorted({r[0]["meta"]["git_commit"] for r in R["win"].values() if r is not None})
    lines = [
        "# Phase 3 — window refinement of a tour with the TSP-20 model",
        "",
        f"Generated by `scripts/phase3_window_report.py` on {datetime.date.today().isoformat()} "
        f"(results from commit(s) {', '.join(commits) or '—'}).",
        "",
        "## Setup",
        "",
        "- **Initial tour.** Greedy edge + neighbour-list 2-opt (k = min(0.5 n, 100)): the baseline of the scale "
        "experiment (Table 1 checks it is the same tour).",
        "- **Windows** (`src/tsp_gnn/window.py`). Segments of w = 20, 30 or 50 consecutive cities of the current "
        "tour; the two end cities stay, the interior is re-ordered by a path solver and the segment is replaced only "
        "if the path is shorter, so the tour never gets longer. One pass = two half-passes with offsets 0 and w/2; "
        "each half-pass cuts the cycle into disjoint windows (solved together, the network in one batch), so "
        "consecutive windows overlap by w/2. Passes repeat until one brings no improvement (at most 5).",
        "- **Path solvers.** TAGCN K=3 (BatchNorm, best.pt, seeds 0–2, trained on TSP-20; the w cities rescaled to "
        "[0, 1]², k-NN graph with k = round(0.5 w)) decoded as a Hamiltonian path between the ends: greedy, or beam "
        "B=10 (`hier.greedy_path` / `hier.beam_path`). LKH (1 run) on the window plus a dummy city joined only to the "
        "two ends: what a window of this size can give at best. 2-opt on the segment alone: control without model.",
        f"- **Tests.** Paired Wilcoxon of the final gap against the initial tour where there are ≥ {MIN_TEST} "
        "instances (n ≤ 1000); max p over the 3 seeds for the model; Bonferroni within each size (12 tests). "
        "Improvements are never negative, so a test can only say \"better\" or n.s.",
        "",
        "## Table 1 — Consistency of the initial tour",
        "",
        table_consistency(R),
        "",
        "## Table 2 — Final gap (%)",
        "",
        "Mean ± std over the 3 seeds for the model; LKH and 2-opt windows are deterministic.",
        "",
        table_values(R, "gap"),
        "",
        "![Gap vs. n](figs/phase3_window_gap.png)",
        "",
        "## Table 3 — Gain over greedy edge + 2-opt (pp)",
        "",
        table_values(R, "gain"),
        "",
        "## Table 4 — Share of the LKH window gain obtained",
        "",
        table_share_of_lkh(R),
        "",
        "## Table 5 — Windows that improved (all passes) and passes made",
        "",
        table_windows(R),
        "",
        "## Table 6 — Paired tests against greedy edge + 2-opt",
        "",
        "Cell: verdict, mean difference final − initial (pp), max p over seeds. Bonferroni: "
        + ("; ".join(f"n = {n}: 0.05/{f} = {t:.2e}" for n, f, t in families) or "—") + ".",
        "",
        table_tests(tests),
        "",
        "### n with fewer than 20 instances (no test): mean difference final − initial and instances improved",
        "",
        table_small(R),
        "",
        "## Table 7 — Time per instance (ms)",
        "",
        "One CPU process per solver/seed (5 in parallel), network on the GPU in batches.",
        "",
        table_times(R),
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
    parser = argparse.ArgumentParser(description="Phase 3 window refinement report.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/phase3_window_report.md")
    args = parser.parse_args()
    text = build_report(args.results)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
