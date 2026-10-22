"""Phase 3 window size vs. solver report: where does the model pay off against LKH?

Reads results/phase3_wsize/{lkh,tagcn_k3_seed<s>,tagcn_k3_tsp50_seed<s>,tagcn_k3_tsp100_seed<s>}/<dataset>.{json,npz}
(scripts/wsize_eval.py) and the training results results/runs/tagcn_k3_{tsp50,tsp100}_seed<s>/eval.json
(TSP-20 models: results/runs/tagcn_k3_seed<s>/kfrac05/eval.json). Writes results/phase3_wsize_report.md and
results/figs/phase3_wsize_{pareto,cost}.{png,pdf}. Missing results are reported as "missing". Hand-written
notes (the "three lines") are read from results/phase3_wsize_notes.md.

Example:
    python scripts/phase3_wsize_report.py
"""

import argparse
import datetime
import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")  # no window: only save files
import matplotlib.pyplot as plt
import matplotlib.ticker
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

SETS = {1000: "tsp1000_test", 10000: "tsp10000_test"}
SEEDS = (0, 1, 2)
WIDTHS = (20, 50, 100, 200, 300)
TRAIN_SIZE = {"tsp20": 20, "tsp50": 50, "tsp100": 100}
MIN_TEST = 20


def model_folder(train, s):
    return f"tagcn_k3_seed{s}" if train == "tsp20" else f"tagcn_k3_{train}_seed{s}"


def matched(w):
    """Training size closest to w (ties: the smaller)."""
    return min(TRAIN_SIZE, key=lambda t: (abs(TRAIN_SIZE[t] - w), TRAIN_SIZE[t]))


# solver -> (label, colour); "matched" = model trained on the size closest to w
SOLVERS = {"lkh": ("LKH on the window (1 run)", "#1baf7a"),
           "matched": ("TAGCN trained on the size closest to w, beam10", "#2a78d6"),
           "tsp20": ("TAGCN trained on TSP-20, beam10", "#eb6834")}


def folders(solver, w):
    if solver == "lkh":
        return ["lkh"], "lkh"
    train = "tsp20" if solver == "tsp20" else matched(w)
    return [model_folder(train, s) for s in SEEDS], "beam10"


def _read(path_no_ext):
    if not os.path.exists(path_no_ext + ".json"):
        return None
    with open(path_no_ext + ".json") as f:
        summary = json.load(f)
    with np.load(path_no_ext + ".npz") as z:
        return summary, {k: z[k] for k in z.files}


def load_all(results):
    names = {"lkh"} | {model_folder(t, s) for t in TRAIN_SIZE for s in SEEDS}
    win = {(f, n): _read(os.path.join(results, "phase3_wsize", f, ds)) for f in names for n, ds in SETS.items()}
    return {"win": win, "runs_dir": os.path.join(results, "runs")}


def runs_of(R, solver, n, w):
    """Per-folder per-instance arrays for (solver, w) at size n, or None if any is missing."""
    fs, method = folders(solver, w)
    runs = [R["win"][f, n] for f in fs]
    if any(r is None or f"{method}_w{w}__gap" not in r[1] for r in runs):
        return None
    return [{k.split("__")[1]: v for k, v in r[1].items() if k.startswith(f"{method}_w{w}__")} | {
        "initial_gap": r[1]["initial__gap"], "initial_ms": r[1]["initial__ms"]} for r in runs]


def initial(R, n):
    for (f, m), r in R["win"].items():
        if m == n and r is not None:
            return r[1]
    return None


def fmt(values, digits=2):
    if values is None:
        return "missing"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}" if len(values) > 1 \
        else f"{values[0]:.{digits}f}"


def point(R, solver, n, w):
    """Per-run means of the quantities in the tables, or None."""
    runs = runs_of(R, solver, n, w)
    if runs is None:
        return None
    return {"gap": [float(r["gap"].mean()) for r in runs],
            "gain": [float(r["gain"].mean()) for r in runs],
            "s": [float((r["initial_ms"] + r["ms"]).mean()) / 1000 for r in runs],
            "s_refine": [float(r["ms"].mean()) / 1000 for r in runs],
            "ms_window": [float(r["ms"].sum() / r["windows"].sum()) for r in runs],
            "ms_net_window": [float(r["ms_model"].sum() / r["windows"].sum()) for r in runs],
            "pp_per_s": [float(r["gain"].mean() / (r["ms"].mean() / 1000)) for r in runs],
            "improved": [100 * float(r["improved"].sum() / r["windows"].sum()) for r in runs],
            "passes": [float(r["passes"].mean()) for r in runs],
            "limited": [int(r["limited"].sum()) for r in runs]}


# ---------------------------------------------------------------- tables

def table_training(R):
    lines = ["| Training set | Seed | Best epoch | Val gap (greedy) | TSP-50 test greedy / + 2-opt | TSP-100 test greedy / + 2-opt |",
             "|---|---|---|---|---|---|"]
    for train in TRAIN_SIZE:
        for s in SEEDS:
            path = os.path.join(R["runs_dir"], model_folder(train, s), *(["kfrac05"] if train == "tsp20" else []), "eval.json")
            if not os.path.exists(path):
                lines.append(f"| TSP-{TRAIN_SIZE[train]} | {s} | missing | | | |")
                continue
            with open(path) as f:
                e = json.load(f)
            meta = e["meta"]
            if train == "tsp20":
                with open(os.path.join(R["runs_dir"], model_folder(train, s), "eval.json")) as f:
                    meta = json.load(f)["meta"]
            cells = [f"{e[d]['greedy']['gap_mean']:.2f} / {e[d]['greedy+2opt']['gap_mean']:.2f}" if d in e else "—"
                     for d in ("tsp50_test", "tsp100_test")]
            lines.append(f"| TSP-{TRAIN_SIZE[train]} | {s} | {meta['best_epoch']} | {meta['best_val_gap']:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_cost(R):
    head = " | ".join(f"n = {n}: {lab}" for n in SETS for lab in ("LKH", "beam (TSP-20)", "beam (matched)", "network"))
    lines = ["| w | " + head + " |", "|---|" + "---|" * (4 * len(SETS))]
    for w in WIDTHS:
        cells = []
        for n in SETS:
            pts = [point(R, s, n, w) for s in ("lkh", "tsp20", "matched")]
            cells += [fmt(p["ms_window"], 1) if p else "missing" for p in pts]
            cells.append(fmt(pts[2]["ms_net_window"], 2) if pts[2] else "missing")
        lines.append(f"| {w} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_main(R, n):
    ini = initial(R, n)
    if ini is None:
        return "missing"
    lines = ["| w | Solver | Model trained on | Gap (%) | Gain (pp) | Share of LKH's gain | Total time (s) | ms per window | "
             "Gain per second (pp/s) | Windows improved | Passes |",
             "|---|---|---|---|---|---|---|---|---|---|---|",
             f"| — | Greedy edge + 2-opt (initial tour) | — | {ini['initial__gap'].mean():.2f} | — | — | "
             f"{ini['initial__ms'].mean() / 1000:.2f} | — | — | — | — |"]
    for w in WIDTHS:
        lkh = point(R, "lkh", n, w)
        for solver in SOLVERS:
            if solver == "matched" and matched(w) == "tsp20":
                continue                                  # w = 20: the matched model is the TSP-20 model
            p = point(R, solver, n, w)
            trained = "—" if solver == "lkh" else ("TSP-20" if solver == "tsp20" else matched(w).upper().replace("TSP", "TSP-"))
            name = SOLVERS[solver][0].split(",")[0].replace(" trained on the size closest to w", "")
            if p is None:
                lines.append(f"| {w} | {name} | {trained} | missing | | | | | | | |")
                continue
            share = f"{100 * np.mean(p['gain']) / np.mean(lkh['gain']):.0f} %" if lkh and solver != "lkh" else "—"
            passes = fmt(p["passes"], 1) + (f" (1 pass: LKH > 5 s per window)" if sum(p["limited"]) else "")
            lines.append(f"| {w} | {name} | {trained} | {fmt(p['gap'])} | {fmt(p['gain'])} | {share} | {fmt(p['s'], 1)} | "
                         f"{fmt(p['ms_window'], 1)} | {fmt(p['pp_per_s'], 3)} | {fmt(p['improved'], 1)} % | {passes} |")
    return "\n".join(lines)


def run_tests(R):
    """{(w, a, ref): result} on the sets with >= 20 instances; Bonferroni over the whole family of that set."""
    tests = {}
    for n in SETS:
        ini = initial(R, n)
        if ini is None or len(ini["initial__gap"]) < MIN_TEST:
            continue
        family = []
        for w in WIDTHS:
            pairs = [("lkh", "baseline"), ("tsp20", "baseline"), ("tsp20", "lkh")]
            if matched(w) != "tsp20":
                pairs += [("matched", "baseline"), ("matched", "tsp20"), ("matched", "lkh")]
            for a, ref in pairs:
                ra = runs_of(R, a, n, w)
                rr = None if ref == "baseline" else runs_of(R, ref, n, w)
                if ra is None or (ref != "baseline" and rr is None):
                    continue
                refs = [r["initial_gap"] for r in ra] if ref == "baseline" else \
                    [r["gap"] for r in (rr * len(ra) if len(rr) == 1 else rr)]
                gaps = [r["gap"] for r in ra] if len(ra) == len(refs) else [ra[0]["gap"]] * len(refs)
                family.append(((n, w, a, ref), gaps, refs))
        for key, a, ref in family:
            tests[key] = {**beam.paired(a, ref, ALPHA / len(family)), "threshold": ALPHA / len(family),
                          "family": len(family)}
    return tests


def table_tests(tests):
    if not tests:
        return "No set with at least 20 instances has complete results yet."
    names = {"lkh": "LKH", "tsp20": "TSP-20 model", "matched": "matched model", "baseline": "greedy edge + 2-opt"}
    lines = ["| n | w | Comparison | Verdict | Mean difference (pp) | Largest p |", "|---|---|---|---|---|---|"]
    for (n, w, a, ref), t in sorted(tests.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        v = f"**{t['verdict']}**" if t["verdict"] != "n.s." else "n.s."
        lines.append(f"| {n} | {w} | {names[a]} vs. {names[ref]} | {v} | {t['mean_diff']:+.2f} | {t['p_max']:.1e} |")
    fam = {k[0]: (t["family"], t["threshold"]) for k, t in tests.items()}
    lines += ["", "Difference = final gap of the first − the second (negative = first shorter). Model rows: the largest p "
              "over the 3 seeds. Bonferroni over the set: " +
              "; ".join(f"n = {n}: 0.05/{f} = {th:.4f}" for n, (f, th) in sorted(fam.items())) + "."]
    return "\n".join(lines)


def table_small(R):
    sizes = [n for n in SETS if (ini := initial(R, n)) is not None and len(ini["initial__gap"]) < MIN_TEST]
    if not sizes:
        return "No set with fewer than 20 instances has results yet."
    lines = ["| n | w | Matched model vs. TSP-20 model | Matched model vs. LKH | TSP-20 model vs. LKH |", "|---|---|---|---|---|"]
    for n in sizes:
        for w in WIDTHS:
            cells = []
            for a, ref in (("matched", "tsp20"), ("matched", "lkh"), ("tsp20", "lkh")):
                if a == "matched" and matched(w) == "tsp20":
                    cells.append("— (w = 20: the matched model is the TSP-20 model)")
                    continue
                ra, rr = runs_of(R, a, n, w), runs_of(R, ref, n, w)
                if ra is None or rr is None:
                    cells.append("missing")
                    continue
                rr = rr * len(ra) if len(rr) == 1 else rr
                diff = np.concatenate([x["gap"] - y["gap"] for x, y in zip(ra, rr)])
                cells.append(f"{diff.mean():+.2f} ({int((diff < -1e-6).sum())}/{len(diff)} shorter)")
            lines.append(f"| {n} | {w} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- figures

def _log_x(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def plot_pareto(R, figs_dir):
    fig, axes = plt.subplots(1, len(SETS), figsize=(13, 5))
    for ax, n in zip(axes, SETS):
        ini = initial(R, n)
        if ini is None:
            ax.text(0.5, 0.5, "missing", ha="center", color=MUTED, transform=ax.transAxes)
            continue
        ax.plot(ini["initial__ms"].mean() / 1000, ini["initial__gap"].mean(), "s", color=INK, markersize=8,
                label="Greedy edge + 2-opt")
        for solver, (label, color) in SOLVERS.items():
            pts = [(w, point(R, solver, n, w)) for w in WIDTHS]
            pts = [(w, p) for w, p in pts if p is not None]
            if not pts:
                continue
            x = [np.mean(p["s"]) for _, p in pts]
            y = [np.mean(p["gap"]) for _, p in pts]
            ax.plot(x, y, "-o", color=color, linewidth=2, markersize=7, label=label)
            if solver == "tsp20":
                continue                                  # same w as the matched model's points, just next to them
            for (w, _), xx, yy in zip(pts, x, y):
                ax.annotate(f"w={w}", (xx, yy), textcoords="offset points", xytext=(6, 4), fontsize=7, color=INK)
        _log_x(ax)
        summarize.style_axes(ax)
        ax.set_title(f"n = {n}", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("Total time per instance (s, log scale)", color=INK)
    axes[0].set_ylabel("Gap to LKH (%)", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Final gap vs. total time, one point per (w, solver) (≤ 2 passes; models: mean over 3 seeds)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs(figs_dir, exist_ok=True)
    summarize.save(fig, figs_dir, "phase3_wsize_pareto")


def plot_cost(R, figs_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for solver, (label, color) in SOLVERS.items():
        for ax, key, ylabel in ((axes[0], "ms_window", "Time per window (ms, log scale)"),
                                (axes[1], "pp_per_s", "Gain per second of refinement (pp/s, log scale)")):
            for n, style in zip(SETS, ("-", "--")):
                pts = [(w, point(R, solver, n, w)) for w in WIDTHS]
                pts = [(w, np.mean(p[key])) for w, p in pts if p is not None and np.mean(p[key]) > 0]
                if pts:
                    ax.plot(*zip(*pts), style, marker="o", color=color, linewidth=2, markersize=6,
                            label=f"{label.split(' (')[0].split(',')[0]}, n = {n}")
            ax.set_ylabel(ylabel, color=INK)
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(WIDTHS); ax.set_xticklabels([str(w) for w in WIDTHS])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        summarize.style_axes(ax)
        ax.set_xlabel("Window size w (cities, log scale)", color=INK)
    axes[0].axhline(5000, color=MUTED, linestyle=":", linewidth=1)
    axes[0].text(20, 5000 * 1.15, "5 s per window", color=MUTED, fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Cost of a window and gain per second vs. window size (solid: n = 1000, dashed: n = 10000)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    os.makedirs(figs_dir, exist_ok=True)
    summarize.save(fig, figs_dir, "phase3_wsize_cost")


# ---------------------------------------------------------------- report

def build_report(results, figs_dir=None):
    R = load_all(results)
    figs_dir = figs_dir or os.path.join(results, "figs")
    tests = run_tests(R)
    plot_pareto(R, figs_dir)
    plot_cost(R, figs_dir)
    notes_path = os.path.join(results, "phase3_wsize_notes.md")
    notes = open(notes_path).read().strip() if os.path.exists(notes_path) else "(to be written after reading the results)"
    expected = [("lkh", n) for n in SETS] + [(model_folder("tsp20", s), n) for s in SEEDS for n in SETS] + \
        [(model_folder(t, s), n) for t in ("tsp50", "tsp100") for s in SEEDS for n in SETS]
    missing = [f"phase3_wsize/{f}/{SETS[n]}" for f, n in expected if R["win"][f, n] is None]
    commits = sorted({r[0]["meta"]["git_commit"] for r in R["win"].values() if r is not None})
    limited = sorted({(n, w) for (f, n), r in R["win"].items() if r is not None for w in r[0]["meta"].get("limited_widths", [])})
    lines = [
        "# Phase 3 — window size vs. solver: where does the model pay off against LKH?",
        "",
        f"Generated by `scripts/phase3_wsize_report.py` on {datetime.date.today().isoformat()} "
        f"(results from commit(s) {', '.join(commits) or '—'}).",
        "",
        "## Three lines",
        "",
        notes,
        "",
        "## Setup",
        "",
        "- **Tour and windows:** greedy edge + 2-opt tour (the baseline of the scale experiment), refined with windows "
        "of w = 20, 50, 100, 200, 300 consecutive tour cities (ends fixed, replaced only if shorter, step w/2), at most "
        "2 passes; a pass runs only if the previous one improved the tour.",
        "- **Solvers on a window:**",
        "  - LKH with 1 run (a dummy city joined only to the two ends gives the best path between them);",
        "  - TAGCN K=3 (best.pt by validation gap, seeds 0–2), beam B = 10 paths: the model trained on TSP-20 at every w, "
        "and the model trained on the size closest to w (w = 50: TSP-50; w ≥ 100: TSP-100).",
        "- **New models:** TSP-50 (data/tsp50_train, 12000 instances) and TSP-100 (data/tsp100_train, 12000 instances, "
        "seed 19; validation 500, seed 20), phase 1 protocol (20 epochs, Adam 1e-3 with cosine decay, batch 64, "
        "hidden 64, 2 layers) with k = 0.5 n (`--k-frac 0.5`), the k = round(0.5 m) the windows use. For TSP-20 this is "
        "the k = 10 of phase 1.",
        "- **LKH time limit:** if the first pass of LKH on the first instance of a set takes more than 5 s per window, that "
        "width gets 1 pass on every instance of the set. Applied to: " +
        (", ".join(f"n = {n}, w = {w}" for n, w in limited) if limited else "none") + ".",
        "- **Time** = initial tour + refinement (network on the shared T4 GPU, beam decoding in single-thread Python, LKH "
        "single-thread), per instance. Gain per second = gain (pp) / refinement time (s). Up to 8 jobs shared 8 cores, so "
        "absolute times include some contention; all solvers ran under the same conditions.",
        "",
        "## 1. The new models",
        "",
        "TSP-20 models: their phase 1 run (selected epoch, validation gap) and k = 0.5 n evaluation on TSP-50/100.",
        "",
        table_training(R),
        "",
        "## 2. Cost of a window",
        "",
        "Milliseconds per window (refinement time / windows solved), mean ± std over seeds; network = the model's "
        "share inside the matched beam.",
        "",
        table_cost(R),
        "",
        "![Cost per window and gain per second](figs/phase3_wsize_cost.png)",
        "",
        "## 3. Gain and time per (w, solver)",
        "",
    ]
    for n in SETS:
        lines += [f"### n = {n}", "", table_main(R, n), ""]
    lines += [
        "Mean ± std over the 3 seeds (LKH and the initial tour have no seed). Share of LKH's gain: same w, means.",
        "",
        "![Gap vs. time](figs/phase3_wsize_pareto.png)",
        "",
        "## 4. Paired tests (n = 1000, 20 instances)",
        "",
        table_tests(tests),
        "",
        "## 5. n = 10000 (3 instances, no test)",
        "",
        "Mean gap difference (pp) and the instance × seed pairs where the first is shorter.",
        "",
        table_small(R),
        "",
        "## Missing results",
        "",
        "\n".join(f"- `{m}`" for m in missing) if missing else "None.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 window size vs. solver report.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/phase3_wsize_report.md")
    args = parser.parse_args()
    text = build_report(args.results)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
