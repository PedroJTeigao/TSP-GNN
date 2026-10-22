"""Phase 3 report: multilevel hierarchical TSP with the TSP-20 model (TAGCN K=3), vs. the phase 2 baselines.

Reads results/phase3_hier/tagcn_k3_seed<s>/tsp<n>_test.{json,npz} (scripts/hier_eval.py) and, from the
scale experiment, results/phase2_scale/baselines/tsp<n>_test.* (greedy edge + 2-opt, k = min(0.5 n, 100))
and results/phase2_scale/tagcn_k3_seed<s>/khalf100/tsp<n>_test.* (the whole instance, same model and seed).
Writes results/phase3_hier_report.md and results/figs/phase3_hier_gap.{png,pdf}. Missing results are
reported as "missing". Hand-written notes (the "three lines") are read from results/phase3_hier_notes.md.

Example:
    python scripts/phase3_hier_report.py
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
PARTITIONS = ("kmeans", "kdtree")
ATTRACTS = ("distance", "heatmap")
CANDS = (1, 3)
DECODERS = ("greedy", "beam10")
VARIANTS = [f"{p}/{a}/c{c}/{d}" for p in PARTITIONS for a in ATTRACTS for c in CANDS for d in DECODERS]
MIN_TEST = 20
# attract × cands -> colour (slots 1-4 of the validated palette); the path decoder is the line style
LINE_COLORS = {("distance", 1): "#2a78d6", ("distance", 3): "#eb6834", ("heatmap", 1): "#1baf7a",
               ("heatmap", 3): "#eda100"}
WHOLE = "Whole instance (TAGCN K=3)"


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
    hier = {n: [_read(os.path.join(results, "phase3_hier", f"tagcn_k3_seed{s}", ds(n))) for s in SEEDS]
            for n in SIZES}
    base = {n: _read(os.path.join(results, "phase2_scale", "baselines", ds(n))) for n in SIZES}
    whole = {n: [_read(os.path.join(results, "phase2_scale", f"tagcn_k3_seed{s}", "khalf100", ds(n))) for s in SEEDS]
             for n in SIZES}
    return {"hier": hier, "base": base, "whole": whole}


def complete(runs):
    return all(r is not None for r in runs)


def fmt(values, digits=2):
    if values is None:
        return "missing"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}" if len(values) > 1 \
        else f"{values[0]:.{digits}f}"


def seed_means(runs, key):
    if not complete(runs) or key not in runs[0][1]:
        return None
    return [float(np.nanmean(r[1][key])) for r in runs]


def base_rows(R, n):
    """{label: per-seed means or a single value} of the baselines for size n."""
    out = {}
    b = R["base"].get(n)
    out["Greedy edge"] = [b[0]["greedy"]["gap_mean"]] if b else None
    out["Greedy edge + 2-opt"] = [b[0]["greedy+2opt@khalf100"]["gap_mean"]] if b else None
    out[f"{WHOLE}, greedy"] = seed_means(R["whole"][n], "greedy")
    out[f"{WHOLE}, greedy + 2-opt"] = seed_means(R["whole"][n], "greedy+2opt")
    return out


# ---------------------------------------------------------------- tables

def table_hierarchy(R):
    lines = ["| n | Partition | Points per level | Group sizes per level: min / mean / max |", "|---|---|---|---|"]
    for n in SIZES:
        runs = R["hier"][n]
        if not complete(runs):
            lines.append(f"| {n} | — | missing | missing |")
            continue
        for p in PARTITIONS:
            info = runs[0][0]["meta"]["hierarchy"].get(p)     # the hierarchy does not depend on the seed
            if not info:
                continue
            levels = np.array([x["points_per_level"] for x in info])
            pts = " → ".join(f"{v:.0f}" for v in levels.mean(axis=0))
            sizes = []
            for l in range(levels.shape[1] - 1):
                s = np.concatenate([x["group_sizes"][l] for x in info])
                sizes.append(f"L{l + 1}: {s.min()} / {s.mean():.1f} / {s.max()}")
            lines.append(f"| {n} | {p} | {pts} | {'; '.join(sizes)} |")
    return "\n".join(lines)


def table_gaps(R, field):
    head = "| Method | " + " | ".join(str(n) for n in SIZES) + " |"
    lines = [head, "|---|" + "---|" * len(SIZES)]
    rows = {}
    for n in SIZES:
        for label, v in base_rows(R, n).items():
            rows.setdefault(label, {})[n] = v
    for label, cells in rows.items():
        lines.append(f"| {label} | " + " | ".join(fmt(cells.get(n)) for n in SIZES) + " |")
    for v in VARIANTS:
        lines.append(f"| {v} | " + " | ".join(fmt(seed_means(R["hier"][n], f"{v}__{field}")) for n in SIZES) + " |")
    return "\n".join(lines)


def run_tests(R):
    """{(n, variant, ref): result}: after-2-opt gap vs. greedy edge + 2-opt and vs. the whole instance."""
    tests = {}
    for n in SIZES:
        runs, b, whole = R["hier"][n], R["base"].get(n), R["whole"][n]
        if not complete(runs) or b is None or runs[0][0]["meta"]["num"] < MIN_TEST:
            continue
        family = []
        for v in VARIANTS:
            a_list = [r[1][f"{v}__after"] for r in runs]
            family.append(((n, v, "greedy edge + 2-opt"), a_list, [b[1]["greedy+2opt@khalf100"]] * len(runs)))
            if complete(whole):
                family.append(((n, v, "whole instance + 2-opt"), a_list, [w[1]["greedy+2opt"] for w in whole]))
        threshold = ALPHA / len(family)
        for key, a_list, refs in family:
            tests[key] = {**beam.paired(a_list, refs, threshold), "threshold": threshold, "family": len(family)}
    return tests


def table_tests(tests, ref):
    sizes = sorted({k[0] for k in tests})
    if not sizes:
        return "No size with at least 20 instances has complete results yet."
    lines = ["| Variant | " + " | ".join(str(n) for n in sizes) + " |", "|---|" + "---|" * len(sizes)]
    for v in VARIANTS:
        cells = []
        for n in sizes:
            t = tests.get((n, v, ref))
            if t is None:
                cells.append("—")
                continue
            verdict = f"**{t['verdict']}**" if t["verdict"] != "n.s." else "n.s."
            cells.append(f"{verdict} {t['mean_diff']:+.2f} (p {t['p_max']:.1e})")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_small(R):
    """n with fewer than 20 instances: mean difference and instances better than greedy edge + 2-opt."""
    sizes = [n for n in SIZES if complete(R["hier"][n]) and R["hier"][n][0][0]["meta"]["num"] < MIN_TEST
             and R["base"].get(n)]
    if not sizes:
        return "No size with fewer than 20 instances has results yet."
    lines = ["| Variant | " + " | ".join(str(n) for n in sizes) + " |", "|---|" + "---|" * len(sizes)]
    for v in VARIANTS:
        cells = []
        for n in sizes:
            diff = np.concatenate([r[1][f"{v}__after"] - R["base"][n][1]["greedy+2opt@khalf100"]
                                   for r in R["hier"][n]])
            ok = ~np.isnan(diff)
            cells.append(f"{diff[ok].mean():+.2f} ({int((diff[ok] < 0).sum())}/{int(ok.sum())} better)")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_attract(R):
    """Heatmap − distance attraction, paired on the same instances (pp; negative = heatmap better)."""
    lines = ["| Partition / cands / paths | Gap | " + " | ".join(str(n) for n in SIZES) + " |",
             "|---|---|" + "---|" * len(SIZES)]
    for p in PARTITIONS:
        for c in CANDS:
            for d in DECODERS:
                for field in ("before", "after"):
                    cells = []
                    for n in SIZES:
                        runs = R["hier"][n]
                        if not complete(runs):
                            cells.append("missing")
                            continue
                        diffs = [r[1][f"{p}/heatmap/c{c}/{d}__{field}"] - r[1][f"{p}/distance/c{c}/{d}__{field}"]
                                 for r in runs]
                        allv = np.concatenate(diffs)
                        cell = f"{np.nanmean(allv):+.2f}"
                        if runs[0][0]["meta"]["num"] >= MIN_TEST:
                            t = beam.paired([r[1][f"{p}/heatmap/c{c}/{d}__{field}"] for r in runs],
                                            [r[1][f"{p}/distance/c{c}/{d}__{field}"] for r in runs], ALPHA / 16)
                            cell += f" ({t['verdict']}, p {t['p_max']:.0e})"
                        cells.append(cell)
                    lines.append(f"| {p} / c{c} / {d} | {field} 2-opt | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_cands(R):
    """c = 3 − c = 1 (pp, after 2-opt) and how often a link needed more candidates than c."""
    lines = ["| Partition / attract / paths | " + " | ".join(str(n) for n in SIZES) + " |", "|---|" + "---|" * len(SIZES)]
    for p in PARTITIONS:
        for a in ATTRACTS:
            for d in DECODERS:
                cells = []
                for n in SIZES:
                    runs = R["hier"][n]
                    if not complete(runs):
                        cells.append("missing")
                        continue
                    diff = np.concatenate([r[1][f"{p}/{a}/c3/{d}__after"] - r[1][f"{p}/{a}/c1/{d}__after"] for r in runs])
                    same = np.mean(np.abs(diff) <= 1e-9)
                    forced = np.mean(np.concatenate([r[1][f"{p}/{a}/c1/{d}__c_used"] > 1 for r in runs]))
                    cells.append(f"{np.nanmean(diff):+.2f}; same tour {100 * same:.0f} %; c1 expanded {100 * forced:.0f} %")
                lines.append(f"| {p} / {a} / {d} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_times(R):
    """Mean time per instance (ms) of each step, averaged over the seeds and the variants named."""
    def mean_of(n, keys):
        runs = R["hier"][n]
        if not complete(runs):
            return None
        vals = [np.nanmean(r[1][k]) for r in runs for k in keys if k in r[1]]
        return float(np.mean(vals)) if vals else None

    rows = [("Partition: k-means", lambda n: mean_of(n, ["kmeans__ms_partition"])),
            ("Partition: k-d bisection", lambda n: mean_of(n, ["kdtree__ms_partition"])),
            ("Top TSP", lambda n: mean_of(n, [f"{v}__ms_top" for v in VARIANTS])),
            ("Stitching: distance", lambda n: mean_of(n, [f"{v}__ms_stitch" for v in VARIANTS if "/distance/" in v])),
            ("Stitching: heatmap", lambda n: mean_of(n, [f"{v}__ms_stitch" for v in VARIANTS if "/heatmap/" in v])),
            ("Paths: greedy", lambda n: mean_of(n, [f"{v}__ms_paths" for v in VARIANTS if v.endswith("greedy")])),
            ("Paths: beam B=10", lambda n: mean_of(n, [f"{v}__ms_paths" for v in VARIANTS if v.endswith("beam10")])),
            ("  of which network (distance / greedy variants)",
             lambda n: mean_of(n, [f"{v}__ms_model" for v in VARIANTS if "/distance/" in v and v.endswith("greedy")])),
            ("  of which network (heatmap / greedy variants)",
             lambda n: mean_of(n, [f"{v}__ms_model" for v in VARIANTS if "/heatmap/" in v and v.endswith("greedy")])),
            ("Final 2-opt", lambda n: mean_of(n, [f"{v}__ms_2opt" for v in VARIANTS])),
            ("Greedy edge + 2-opt (baseline)", lambda n: float(np.mean(R["base"][n][1]["greedy__ms"])
                                                         + np.mean(R["base"][n][1]["greedy+2opt@khalf100__ms"]))
             if R["base"].get(n) else None),
            ("Whole instance: graph + network + greedy + 2-opt", lambda n: float(np.mean(
                [np.mean(w[1]["graph__ms"]) + np.mean(w[1]["gnn__ms"]) + np.mean(w[1]["greedy__ms"])
                 + np.mean(w[1]["greedy+2opt__ms"]) for w in R["whole"][n]])) if complete(R["whole"][n]) else None)]
    lines = ["| Step | " + " | ".join(str(n) for n in SIZES) + " |", "|---|" + "---|" * len(SIZES)]
    for label, fn in rows:
        cells = []
        for n in SIZES:
            v = fn(n)
            cells.append("missing" if v is None else (f"{v:.1f}" if v < 100 else f"{v:,.0f}"))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_errors(R):
    """Where the wrong edges are (before 2-opt): between level-0 groups (stitching) or inside them (paths)."""
    lines = ["| n | Partition | Attract | Paths | Boundary edges (% of n) | Wrong at boundary | Wrong inside | "
             "Share of wrong edges at the boundary | Cities moved between groups by 2-opt (% of n) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n in SIZES:
        runs = R["hier"][n]
        if not complete(runs):
            lines.append(f"| {n} | missing | | | | | | | |")
            continue
        for p in PARTITIONS:
            for a in ATTRACTS:
                for d in DECODERS:
                    keys = [f"{p}/{a}/c{c}/{d}" for c in CANDS]
                    get = lambda f: np.concatenate([r[1][f"{k}__{f}"] for r in runs for k in keys])
                    be, bw, ie, iw = (get(f) for f in ("boundary_edges", "boundary_wrong", "interior_edges", "interior_wrong"))
                    moved = get("moved")
                    lines.append(f"| {n} | {p} | {a} | {d} | {100 * be.sum() / (be.sum() + ie.sum()):.1f} | "
                                 f"{100 * bw.sum() / be.sum():.1f} % | {100 * iw.sum() / ie.sum():.1f} % | "
                                 f"{100 * bw.sum() / (bw.sum() + iw.sum()):.1f} % | {100 * moved.mean() / n:.1f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- figure

def plot_gaps(R, figs_dir):
    """Gap after 2-opt vs. n: rows = partition, columns = path decoder; colour = attract × cands."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for row, p in enumerate(PARTITIONS):
        for col, d in enumerate(DECODERS):
            ax = axes[row, col]
            ends = []
            for label, style, color in (("Greedy edge + 2-opt", "--", INK), (f"{WHOLE}, greedy + 2-opt", ":", MUTED)):
                pts = [(n, base_rows(R, n)[label]) for n in SIZES]
                pts = [(n, np.mean(v)) for n, v in pts if v is not None]
                if pts:
                    ax.plot(*zip(*pts), style, color=color, linewidth=2, marker="o", markersize=4, label=label)
                    ends.append([pts[-1][1], label.replace(f"{WHOLE}, greedy + 2-opt", "Whole instance + 2-opt")])
            for (a, c), color in LINE_COLORS.items():
                pts = [(n, seed_means(R["hier"][n], f"{p}/{a}/c{c}/{d}__after")) for n in SIZES]
                pts = [(n, np.mean(v), min(v), max(v)) for n, v in pts if v is not None]
                if not pts:
                    continue
                x, y, lo, hi = map(np.array, zip(*pts))
                ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
                ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=f"{a}, c = {c}")
                ends.append([y[-1], f"{a}, c={c}"])
            ax.set_xscale("log")
            summarize.style_axes(ax)
            ax.set_xticks(SIZES); ax.set_xticklabels([str(n) for n in SIZES], fontsize=8)
            ax.minorticks_off()
            ax.set_xlim(150, 10000 * 3.5)
            if ends:
                ends.sort()
                lo_, hi_ = ax.get_ylim()
                for i in range(1, len(ends)):
                    ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.045 * (hi_ - lo_))
                for y, name in ends:
                    ax.text(10000 * 1.12, y, name, va="center", color=INK, fontsize=7)
            ax.set_title(f"{p} partition, {'greedy' if d == 'greedy' else 'beam B=10'} paths",
                         color=INK, fontsize=9, loc="left")
            if col == 0:
                ax.set_ylabel("Gap after the final 2-opt (%)", color=INK)
            if row == 1:
                ax.set_xlabel("n (cities, log scale)", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Hierarchical TSP (TAGCN K=3 trained on TSP-20): mean over 3 seeds, band = min–max",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(figs_dir, exist_ok=True)
    summarize.save(fig, figs_dir, "phase3_hier_gap")


# ---------------------------------------------------------------- report

def build_report(results, figs_dir=None):
    R = load_all(results)
    figs_dir = figs_dir or os.path.join(results, "figs")
    tests = run_tests(R)
    plot_gaps(R, figs_dir)
    notes_path = os.path.join(results, "phase3_hier_notes.md")
    notes = open(notes_path).read().strip() if os.path.exists(notes_path) else "(to be written after reading the results)"
    families = sorted({(k[0], t["family"], t["threshold"]) for k, t in tests.items()})
    missing = [f"phase3_hier/tagcn_k3_seed{s}/{ds(n)}" for n, runs in R["hier"].items()
               for s, r in zip(SEEDS, runs) if r is None]
    commits = sorted({r[0]["meta"]["git_commit"] for runs in R["hier"].values() for r in runs if r is not None})
    timeouts = [f"seed {s}, TSP-{n}, {v}: {r[0][v]['timeouts']}" for n, runs in R["hier"].items()
                for s, r in zip(SEEDS, runs) if r is not None for v in VARIANTS if r[0].get(v, {}).get("timeouts")]
    lines = [
        "# Phase 3 — multilevel hierarchical TSP with the TSP-20 model",
        "",
        f"Generated by `scripts/phase3_hier_report.py` on {datetime.date.today().isoformat()} "
        f"(results from commit(s) {', '.join(commits) or '—'}).",
        "",
        "## Setup",
        "",
        "- **Model.** TAGCN K=3, BatchNorm, best.pt, seeds 0–2, trained on TSP-20 (no retraining). Every point set "
        "given to it is rescaled to [0, 1]² and gets a k-NN graph with k = round(0.5 m).",
        "- **Hierarchy** (`src/tsp_gnn/hier.py`). Level 0 = cities; each level is split into groups of ~20 and the "
        "centroids form the next level, until ≤ 60 points. *kmeans*: sklearn k-means with k = round(m/20), then a "
        "capacity repair so that every group has 10–30 points. *kdtree*: recursive median bisection of the longer "
        "axis until ≤ 30 points (groups of 15–30).",
        "- **Top.** TSP over the top-level points with the model (greedy + full 2-opt).",
        "- **Descent.** For each consecutive pair of groups (a, b) one child of a (exit) and one child of b (entry) "
        "are linked: *distance* = closest pair; *heatmap* = the model on a ∪ b (the cities if ≤ 100, otherwise the "
        "children's centroids), attraction = summed edge probability between the children. *c* best pairs per link "
        "and dynamic programming around the cycle under the constraint entry ≠ exit (with c = 1 the DP only acts "
        "when the best pairs conflict; if no choice satisfies the constraint, c is doubled, Table 6). Inside each "
        "group the children are visited along a Hamiltonian path from entry to exit decoded from the model's "
        "heatmap on them: *greedy* (the phase 1 greedy with the ends' degree capped at 1) or *beam10*.",
        "- **Final 2-opt.** The neighbour-list 2-opt of the scale experiment on the whole tour, k = min(0.5 n, 100), "
        "the same as the greedy edge + 2-opt baseline (`results/phase2_scale/baselines`).",
        "- **Baselines** (read from the scale experiment, same instances): greedy edge (+ 2-opt) and the whole "
        "instance with the same TAGCN K=3 checkpoint (k = min(0.5 n, 100), greedy + 2-opt), paired seed by seed.",
        f"- **Tests.** Paired Wilcoxon where there are ≥ {MIN_TEST} instances (n ≤ 1000); max p over the 3 seeds, "
        "the seeds' medians must agree in sign; Bonferroni within each size (16 variants × 2 references).",
        "",
        "## Table 1 — Hierarchy (mean over instances)",
        "",
        table_hierarchy(R),
        "",
        "## Table 2 — Gap (%) after the final 2-opt",
        "",
        "Mean ± std over the 3 seeds; variants are partition / attract / c / path decoder.",
        "",
        table_gaps(R, "after"),
        "",
        "![Gap vs. n](figs/phase3_hier_gap.png)",
        "",
        "## Table 3 — Gap (%) before the final 2-opt",
        "",
        table_gaps(R, "before"),
        "",
        "## Table 4 — Paired tests, gap after 2-opt",
        "",
        "Cell: verdict, mean difference hierarchy − reference (pp; negative = hierarchy better), max p over seeds. "
        "Bonferroni: " + ("; ".join(f"n = {n}: 0.05/{f} = {t:.2e}" for n, f, t in families) or "—") + ".",
        "",
        "### vs. greedy edge + 2-opt",
        "",
        table_tests(tests, "greedy edge + 2-opt"),
        "",
        "### vs. the whole instance (same model and seed) + 2-opt",
        "",
        table_tests(tests, "whole instance + 2-opt"),
        "",
        "### n with fewer than 20 instances (no test): difference vs. greedy edge + 2-opt",
        "",
        table_small(R),
        "",
        "## Table 5 — Heatmap vs. distance attraction",
        "",
        "Heatmap − distance on the same instances (pp; negative = heatmap better). For n ≤ 1000: paired test, "
        "Bonferroni over the 16 rows of a size.",
        "",
        table_attract(R),
        "",
        "## Table 6 — c = 3 vs. c = 1",
        "",
        "Mean difference c3 − c1 after 2-opt (pp), share of instances with the same final tour gap, and share of "
        "c = 1 runs where some link had to look past its best pair (entry = exit conflict).",
        "",
        table_cands(R),
        "",
        "## Table 7 — Time per instance (ms)",
        "",
        "One CPU process per seed (3 in parallel), network on the GPU in batches. Steps of the hierarchy are "
        "averaged over the variants named in the row; the network time is included in the top/stitching/paths rows.",
        "",
        table_times(R),
        "",
        "## Table 8 — Where the errors are (before 2-opt) and what the 2-opt moves",
        "",
        "Boundary = tour edges between two level-0 groups (the stitching links); inside = path edges within a "
        "group. Wrong = not in the LKH tour. Averaged over c = 1 and 3.",
        "",
        table_errors(R),
        "",
        "## 2-opt timeouts",
        "",
        "\n".join(f"- {x}" for x in timeouts) if timeouts else "None.",
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
    parser = argparse.ArgumentParser(description="Phase 3 hierarchical TSP report.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/phase3_hier_report.md")
    args = parser.parse_args()
    text = build_report(args.results)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
