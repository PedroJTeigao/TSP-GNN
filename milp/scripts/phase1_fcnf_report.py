"""FCNF phase 1 report: generator statistics, the prior (heatmap AP), decoders × size, paired tests, figure.

Reads data/fcnf*_test.npz (+ data/fcnf100_test_ref600.npz if present), results/runs/<run>/{eval.json,
eval_probs.npz} and results/phase1_decode/<source>/<set>__<group>.npz; writes results/phase1_fcnf_report.md
and results/figs/phase1_fcnf_pareto.png. Missing pieces are listed, not fatal. The three summary lines
come from results/phase1_fcnf_notes.md (written by hand after reading the results).

Reference cost of each instance: the MILP optimum on fcnf20 / fcnf50 (all proven); on fcnf100 the best
known solution (best of the 120 s and 600 s runs). gap = 100 (cost / reference − 1).

Example:
    python scripts/phase1_fcnf_report.py
"""

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score

from fcnf_gnn.data import load_dataset
from fcnf_gnn.stats import paired

SETS = {"fcnf20_test": "fcnf20", "fcnf50_test": "fcnf50", "fcnf100_test": "fcnf100"}
ENCODERS = {"gcn": "GCN", "tagcn_k3": "TAGCN K=3"}
SEEDS = (0, 1, 2)
QS = (20, 30, 50)
GUIDED = [f"prune_q{q}" for q in QS] + ["warm", "delegate"]
BASELINES = ["slope", "lp_round"] + [f"random_q{q}" for q in QS]
LABELS = {"milp": "MILP (full)", "slope": "Slope scaling", "lp_round": "LP relaxation + rounding + repair",
          **{f"random_q{q}": f"Random pruning q = {q} %" for q in QS},
          **{f"prune_q{q}": f"Pruning q = {q} %" for q in QS}, "warm": "Warm start (full MILP)",
          "delegate": "Delegation (slope scaling + MILP on 30 % nodes)"}
ALPHA = 0.05


def fmt(values, digits=2):
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return "missing"
    return f"{np.mean(v):.{digits}f}" + (f" ± {np.std(v, ddof=1):.{digits}f}" if len(v) > 1 else "")


# ---------------------------------------------------------------- loading

def load_all(args):
    data, refs, missing = {}, {}, []
    for s in SETS:
        path = os.path.join(args.data_dir, f"{s}.npz")
        if not os.path.exists(path):
            missing.append(path)
            continue
        d = load_dataset(path)
        ref_cost, ref_y, source = d["cost"].copy(), d["y"].copy(), "MILP optimum (all proven)"
        rerun = os.path.join(args.data_dir, f"{s}_ref600.npz")
        if os.path.exists(rerun):
            r = dict(np.load(rerun))
            ref_cost, ref_y, source = r["ref_cost"], r["ref_y"], "best of the 120 s and 600 s MILP runs"
            d["ref600"] = r
        elif (d["status"] != 0).any():
            source = "MILP 120 s (600 s re-solve missing)"
            missing.append(rerun)
        data[s], refs[s] = d, {"cost": ref_cost, "y": ref_y, "source": source}
    runs = {}
    for enc in ENCODERS:
        for seed in SEEDS:
            name = f"{enc}_seed{seed}"
            path = os.path.join(args.runs, name)
            if os.path.exists(os.path.join(path, "eval.json")):
                with open(os.path.join(path, "eval.json")) as f:
                    runs[name] = {"eval": json.load(f), "probs": dict(np.load(os.path.join(path, "eval_probs.npz")))}
            else:
                missing.append(path)
    decode = {}
    for source in ["baselines"] + [f"{e}_seed{s}" for e in ENCODERS for s in SEEDS]:
        for s in SETS:
            for group in (["baselines"] if source == "baselines" else ["guided", "warm"]):
                path = os.path.join(args.decode, source, f"{s}__{group}.npz")
                if os.path.exists(path):
                    decode.setdefault((source, s), {}).update(dict(np.load(path)))
                else:
                    missing.append(path)
    return data, refs, runs, decode, missing


def method_arrays(s, method, data, refs, decode, source="baselines"):
    """Per-instance gap (%), seconds, first_feasible, kept_frac, solved of one method (None if missing)."""
    ref = refs[s]["cost"]
    if method == "milp":
        d = data[s]
        cost = np.where(d["status"] == 2, np.nan, d["cost"])
        return {"gap": 100 * (cost / ref - 1), "seconds": d["seconds"], "first_feasible": np.ones(len(ref)),
                "kept_frac": np.ones(len(ref)), "solved": np.isfinite(cost).astype(float)}
    a = decode.get((source, s), {})
    if f"{method}__cost" not in a:
        return None
    cost = a[f"{method}__cost"]
    return {"gap": 100 * (cost / ref[:len(cost)] - 1), "seconds": a[f"{method}__seconds"],
            "first_feasible": a[f"{method}__first_feasible"], "kept_frac": a[f"{method}__kept_frac"],
            "solved": a[f"{method}__solved"].astype(float)}


def seed_arrays(s, enc, method, data, refs, decode):
    out = [method_arrays(s, method, data, refs, decode, f"{enc}_seed{seed}") for seed in SEEDS]
    return [x for x in out if x is not None]


# ---------------------------------------------------------------- sections

def generator_section(data, refs):
    lines = ["## 1. Generator and reference solutions", "",
             "| set | instances | arcs (mean) | MILP s mean | MILP s max | open arcs (base rate) | proven optimal "
             "(120 s / none on fcnf20) | reference |", "|---|---|---|---|---|---|---|---|"]
    for s, d in data.items():
        lines.append(f"| {SETS[s]} | {len(d['cost'])} | {np.diff(d['arc_ptr']).mean():.1f} | {d['seconds'].mean():.2f} "
                     f"| {d['seconds'].max():.1f} | {d['y'].mean():.3f} | {(d['status'] == 0).sum()} / {len(d['cost'])} "
                     f"| {refs[s]['source']} |")
    lines += ["", "Fixed-cost ratio F / (c u) ∈ {0.5, 1, 2}, drawn once per instance:", "",
              "| set | ratio 0.5: MILP s / open | ratio 1 | ratio 2 |", "|---|---|---|---|"]
    for s, d in data.items():
        ptr = d["arc_ptr"]
        open_inst = np.array([d["y"][ptr[i]:ptr[i + 1]].mean() for i in range(len(d["cost"]))])
        cells = [f"{d['seconds'][d['ratio'] == r].mean():.2f} s / {open_inst[d['ratio'] == r].mean():.3f}"
                 for r in (0.5, 1.0, 2.0)]
        lines.append(f"| {SETS[s]} | " + " | ".join(cells) + " |")
    lines += ["", "The ratio changes neither the MILP time nor the fraction of open arcs: the difficulty "
              "lever here is the size n, not the ratio (a harder set, e.g. ratio 10, would be generated "
              "separately)."]
    for s, d in data.items():
        if "ref600" in d:
            r = d["ref600"]
            idx = r["rerun_idx"]
            improved = (r["ref_source"][idx] == 1).sum()
            gain = 100 * (d["cost"][idx] - r["ref_cost"][idx]) / d["cost"][idx]
            lines += ["", f"**{SETS[s]}, 600 s re-solve** of the {len(idx)} instances without a proof at 120 s "
                      f"(6 processes, warm start = the 120 s solution): {(r['rerun_status'] == 0).sum()} proven optimal, "
                      f"{(r['ref_proven'][idx] == 0).sum()} still unproven; the cost improved on {improved} "
                      f"(mean {gain.mean():.2f} %, max {gain.max():.2f} %). Mean solver gap of those instances: "
                      f"{100 * d['mip_gap'][idx].mean():.2f} % at 120 s -> {r['ref_gap'][idx].mean():.2f} % after "
                      f"(gap against the best bound of both runs). Overall, {int(r['ref_proven'].sum())} / "
                      f"{len(d['cost'])} fcnf100 references are proven optimal."]
    return lines


def training_section(runs):
    lines = ["## 2. Training", "", "Best epoch by validation F1 at p ≥ 0.5 (20 epochs, Adam 1e-3 with cosine, "
             "batch 64, 64 channels, 3 layers, residual + LayerNorm, BCE with pos_weight from the training arcs).", "",
             "| run | parameters | pos_weight | best epoch | val F1 | val AP | val precision | val recall |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        m = r["eval"]["meta"]
        v = m["best_val"]
        lines.append(f"| {name} | {m['config']['num_parameters']:,} | {m['config']['pos_weight']:.2f} | {m['best_epoch']} "
                     f"| {v['f1']:.3f} | {v['ap']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} |")
    return lines


def prior_section(data, refs, runs):
    lines = ["## 3. The prior: average precision of the heatmap", "",
             "AP for \"arc open in the reference solution\" (all arcs of the set pooled), mean ± std over 3 seeds; "
             "the base rate is the AP of a constant heatmap. On fcnf100 the labels are the reference solutions "
             "(best known, see section 1). *lift* = AP / base rate.", "",
             "| encoder | " + " | ".join(f"{SETS[s]} AP | lift" for s in data) + " |",
             "|---|" + "---|---|" * len(data)]
    lines.append("| base rate | " + " | ".join(f"{refs[s]['y'].mean():.3f} | 1" for s in data) + " |")
    ap = {}
    for enc, label in ENCODERS.items():
        cells = []
        for s in data:
            vals = [float(average_precision_score(refs[s]["y"], runs[f"{enc}_seed{seed}"]["probs"][s]))
                    for seed in SEEDS if f"{enc}_seed{seed}" in runs and s in runs[f"{enc}_seed{seed}"]["probs"]]
            ap[(enc, s)] = vals
            cells += [fmt(vals, 3), f"{np.mean(vals) / refs[s]['y'].mean():.2f}" if vals else "missing"]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "AP relative to fcnf20 (the training size): "
              + "; ".join(f"{ENCODERS[e]} " + ", ".join(f"{SETS[s]} {np.mean(ap[(e, s)]) / np.mean(ap[(e, 'fcnf20_test')]):.2f}"
                                                         for s in data if s != "fcnf20_test" and ap[(e, s)])
                          for e in ENCODERS if ap.get((e, "fcnf20_test"))) + "."]
    return lines


def decoder_section(data, refs, decode, runs, args_decode):
    lines = ["## 4. Decoders × size", "",
             "gap = 100 (cost / reference − 1), mean over the instances (solved ones), mean ± std over 3 seeds for "
             "the heatmap decoders; *s* = solver time per instance (LPs + MILPs + heuristic; the network adds "
             "the inference time below); *1st feas.* = % of instances that needed no repair (pruning: the "
             "restricted MILP had a solution; warm start: y = 1[p ≥ 0.5] admits a flow; LP rounding: the rounded "
             "arcs admit a flow); *kept* = mean fraction of arcs the MILP could open (delegation: free arcs); "
             "*solved* = % with a solution. Every MILP: no limit on fcnf20, 120 s on fcnf50 / fcnf100; "
             "8 instances in parallel, one HiGHS thread each (2 at a time while the 600 s re-solve ran).", ""]
    infer = {s: [r["eval"][s]["inference_ms_per_inst"] for r in runs.values() if s in r["eval"]] for s in data}
    lines.append("Network inference: " + ", ".join(f"{SETS[s]} {np.mean(v):.2f} ms" for s, v in infer.items() if v)
                 + " per instance (batched on the GPU).")
    procs = {}
    for path in sorted(glob.glob(os.path.join(args_decode, "*", "*.json"))):
        with open(path) as f:
            meta = json.load(f)["meta"]
        procs.setdefault(meta["procs"], []).append(f"{meta['dataset'].replace('_test', '')}/{meta['source']}/{meta['group']}")
    if procs:
        lines += ["", "Processes in parallel when each result was measured (the reference MILP times: 8): "
                  + "; ".join(f"**{p}**: {len(v)} result{'s' if len(v) > 1 else ''} ({', '.join(v) if len(v) <= 12 else ', '.join(v[:12]) + ', …'})"
                              for p, v in sorted(procs.items())) + ". Fewer processes means less contention, so "
                  "times measured with 2 are slightly favourable to the method."]
    lines.append("")
    for s in data:
        lines += [f"### {SETS[s]} (reference: {refs[s]['source']})", "",
                  "| method | heatmap | gap (%) | s | 1st feas. (%) | kept | solved (%) |", "|---|---|---|---|---|---|---|"]
        for m in ["milp"] + BASELINES:
            a = method_arrays(s, m, data, refs, decode)
            if a is None:
                lines.append(f"| {LABELS[m]} | — | missing | | | | |")
                continue
            lines.append(f"| {LABELS[m]} | — | {np.nanmean(a['gap']):.2f} | {a['seconds'].mean():.2f} | "
                         f"{100 * a['first_feasible'].mean():.0f} | {a['kept_frac'].mean():.2f} | {100 * a['solved'].mean():.0f} |")
        for m in GUIDED:
            for enc, label in ENCODERS.items():
                xs = seed_arrays(s, enc, m, data, refs, decode)
                if not xs:
                    lines.append(f"| {LABELS[m]} | {label} | missing | | | | |")
                    continue
                note = f" ({len(xs)} seed{'s' if len(xs) > 1 else ''})" if len(xs) < len(SEEDS) else ""
                lines.append(f"| {LABELS[m]} | {label}{note} | {fmt([np.nanmean(x['gap']) for x in xs])} | "
                             f"{fmt([x['seconds'].mean() for x in xs])} | {fmt([100 * x['first_feasible'].mean() for x in xs], 0)} | "
                             f"{fmt([x['kept_frac'].mean() for x in xs])} | {fmt([100 * x['solved'].mean() for x in xs], 0)} |")
        lines.append("")
    return lines


def tests_section(data, refs, decode):
    lines = ["## 5. Paired tests", "",
             "Wilcoxon signed-rank per instance, one test per seed, p = max over seeds (a baseline is repeated "
             "against each seed); Bonferroni within each size (threshold 0.05 / number of tests of that size). "
             "*better* = the first method has the lower gap (or time). Δ = mean difference over the instances, "
             "averaged over seeds (gap in % points, time in s).", ""]
    for s in data:
        tests = []
        for enc, label in ENCODERS.items():
            for q in QS:
                tests.append((f"{label} pruning q{q}", f"random pruning q{q}", "gap", enc, f"prune_q{q}", f"random_q{q}"))
            for q in QS:
                tests.append((f"{label} pruning q{q}", "MILP", "gap", enc, f"prune_q{q}", "milp"))
                tests.append((f"{label} pruning q{q}", "MILP", "seconds", enc, f"prune_q{q}", "milp"))
            tests.append((f"{label} warm start", "MILP", "gap", enc, "warm", "milp"))
            tests.append((f"{label} warm start", "MILP", "seconds", enc, "warm", "milp"))
            tests.append((f"{label} delegation", "slope scaling", "gap", enc, "delegate", "slope"))
            tests.append((f"{label} delegation", "MILP", "seconds", enc, "delegate", "milp"))
            for q in QS:
                tests.append((f"{label} delegation", f"{label} pruning q{q}", "gap", enc, "delegate", f"prune_q{q}"))
        threshold = ALPHA / len(tests)
        lines += [f"### {SETS[s]} ({len(tests)} tests, threshold {threshold:.1e})", "",
                  "| comparison | metric | Δ | better / equal / worse (%) | p max | seeds | verdict |", "|---|---|---|---|---|---|---|"]
        for name, ref_name, metric, enc, m, ref_m in tests:
            xs = seed_arrays(s, enc, m, data, refs, decode)
            if ref_m in GUIDED:
                rs = seed_arrays(s, enc, ref_m, data, refs, decode)
            else:
                r = method_arrays(s, ref_m, data, refs, decode)
                rs = [r] * len(xs) if r is not None else []
            if not xs or len(rs) != len(xs):
                lines.append(f"| {name} vs {ref_name} | {metric} | missing | | | | |")
                continue
            pairs = []
            for x, r in zip(xs, rs):
                ok = np.isfinite(x[metric]) & np.isfinite(r[metric])
                pairs.append((x[metric][ok], r[metric][ok]))
            res = paired([p[0] for p in pairs], [p[1] for p in pairs], threshold)
            lines.append(f"| {name} vs {ref_name} | {metric} | {res['mean_diff']:+.3f} | {res['better']:.0f} / "
                         f"{res['equal']:.0f} / {res['worse']:.0f} | {res['p_max']:.1e} | {res['seeds']} | {res['verdict']} |")
        lines.append("")
    return lines


def figure(data, refs, decode, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

    colors = {"milp": "#222222", "slope": "#8c564b", "lp_round": "#7f7f7f", "random": "#bcbd22",
              "prune": "#1f77b4", "warm": "#d62728", "delegate": "#2ca02c"}
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 4.2), squeeze=False)
    for ax, s in zip(axes[0], data):
        def point(a, label, color, marker):
            ring = {} if marker == "x" else {"edgecolor": "white", "linewidth": 1.5}
            ax.scatter(np.mean(a["seconds"]), np.nanmean(a["gap"]), s=55, color=color, marker=marker, zorder=3,
                       label=label, **ring)
        # the full MILP sits on the same point as the warm starts: a large hollow square drawn on top
        a = method_arrays(s, "milp", data, refs, decode)
        ax.scatter(np.mean(a["seconds"]), np.nanmean(a["gap"]), s=170, facecolors="none", edgecolors=colors["milp"],
                   linewidths=1.8, marker="s", zorder=5, label="MILP (full)")
        for m in BASELINES:
            a = method_arrays(s, m, data, refs, decode)
            if a is not None:
                point(a, LABELS[m], colors["random" if m.startswith("random") else m], "x" if m.startswith("random") else "D")
        for enc, marker in (("gcn", "o"), ("tagcn_k3", "^")):
            for m in GUIDED:
                xs = seed_arrays(s, enc, m, data, refs, decode)
                if xs:
                    a = {"seconds": np.mean([x["seconds"].mean() for x in xs]), "gap": np.mean([np.nanmean(x["gap"]) for x in xs])}
                    point(a, f"{LABELS[m]} ({ENCODERS[enc]})", colors[m.split("_")[0]], marker)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("solver time per instance (s, log)")
        ax.set_ylabel("cost gap to the reference (%)")
        ax.set_title(SETS[s])
        ax.grid(alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="FCNF phase 1 report.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--runs", default="results/runs")
    parser.add_argument("--decode", default="results/phase1_decode")
    parser.add_argument("--notes", default="results/phase1_fcnf_notes.md")
    parser.add_argument("--out", default="results/phase1_fcnf_report.md")
    parser.add_argument("--fig", default="results/figs/phase1_fcnf_pareto.png")
    args = parser.parse_args()
    data, refs, runs, decode, missing = load_all(args)

    notes = open(args.notes).read().strip() if os.path.exists(args.notes) else "(to be written after reading the results)"
    lines = ["# FCNF phase 1: a GNN prior for fixed-charge network flow", "",
             "Instances: directed graphs with k = 4 nearest-neighbour arcs both ways, 20 % supplies and 20 % demands; "
             "formulation in `docs/fcnf_formulation.md`. Labels: open arcs of the MILP solution (HiGHS). Models "
             "trained on fcnf20 only and applied to fcnf20 / fcnf50 / fcnf100.", "", "## Summary", "", notes, ""]
    lines += generator_section(data, refs) + [""]
    if runs:
        lines += training_section(runs) + [""] + prior_section(data, refs, runs) + [""]
    lines += decoder_section(data, refs, decode, runs, args.decode)
    lines += tests_section(data, refs, decode)
    if data:
        figure(data, refs, decode, args.fig)
        lines += ["## 6. Figure", "", f"![Gap vs. time](figs/{os.path.basename(args.fig)})", ""]
    lines += ["## Missing results", ""] + ([f"- `{m}`" for m in missing] if missing else ["None."])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved to {args.out} ({len(missing)} missing)")


if __name__ == "__main__":
    main()
