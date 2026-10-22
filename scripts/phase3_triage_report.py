"""Phase 3 triage report: per window, do nothing, use the model, or call LKH?

Reads results/phase3_triage/log/tagcn_k3_seed<s>/tsp<n>_test.npz (scripts/triage_log.py) and
results/phase3_triage/policies/{tagcn_k3_seed<s>,lkh}/tsp<n>_test.{json,npz} (scripts/triage_eval.py).
Writes results/phase3_triage_report.md and results/figs/phase3_triage_{pr,pareto}.{png,pdf}.
Missing results are reported as "missing". Hand-written notes (the "three lines") are read from
results/phase3_triage_notes.md.

Example:
    python scripts/phase3_triage_report.py
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
from scipy.stats import spearmanr

from tsp_gnn.triage import average_precision, pr_curve, signals, targets

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarize = _load("summarize")
beam = _load("phase2_beam_report")
INK, MUTED, ALPHA = summarize.INK, summarize.MUTED, summarize.ALPHA

SIZES = (200, 1000, 10000)
SEEDS = (0, 1, 2)
VALIDATION = 200
P2_TOP = (10, 20, 30, 50)
P3_TOP = (10, 20, 30)
POLICIES = ["P0", *[f"P2_top{q}" for q in P2_TOP], *[f"P3_top{q}" for q in P3_TOP]]     # per seed; P1 in lkh/
LABELS = {"P0": "P0: model everywhere", "P1": "P1: LKH everywhere",
          **{f"P2_top{q}": f"P2: LKH on top {q} % disagreement" for q in P2_TOP},
          **{f"P3_top{q}": f"P3: model if confident, LKH if top {q} % dis. and not confident" for q in P3_TOP}}
SIGNAL_COLORS = {"disagreement": "#2a78d6", "logp": "#eb6834", "margin": "#1baf7a", "model gain": "#eda100"}
POLICY_COLORS = {"P0": "#eb6834", "P1": "#1baf7a", "P2": "#2a78d6", "P3": "#eda100"}
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
    root = os.path.join(results, "phase3_triage")
    logs = {(s, n): _read(os.path.join(root, "log", f"tagcn_k3_seed{s}", ds(n))) for s in SEEDS for n in SIZES}
    pol = {(s, n): _read(os.path.join(root, "policies", f"tagcn_k3_seed{s}", ds(n))) for s in SEEDS for n in SIZES}
    pol.update({("lkh", n): _read(os.path.join(root, "policies", "lkh", ds(n))) for n in SIZES})
    return {"logs": logs, "pol": pol}


def logs_at(R, n):
    """Per-seed window logs at size n, or None if any seed is missing."""
    runs = [R["logs"][s, n] for s in SEEDS]
    return [r[1] for r in runs] if all(r is not None for r in runs) else None


def pooled(logs):
    return {k: np.concatenate([x[k] for x in logs]) for k in logs[0]}


def fmt(values, digits=2):
    if values is None:
        return "missing"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}" if len(values) > 1 \
        else f"{values[0]:.{digits}f}"


def policy_runs(R, p, n):
    """Per-instance arrays of policy p at size n, one dict per seed (P1: one, shared), or None."""
    if p == "P1":
        r = R["pol"]["lkh", n]
        return None if r is None else [r[1]]
    runs = [R["pol"][s, n] for s in SEEDS]
    return [r[1] for r in runs] if all(r is not None for r in runs) else None


def initial(R, n):
    for key in [("lkh", n), *[(s, n) for s in SEEDS]]:
        r = R["pol"].get(key)
        if r is not None:
            return r[1]
    return None


# ---------------------------------------------------------------- tables: signals

def table_base_rates(R):
    lines = ["| n | Windows per seed | LKH improves | Model (beam) improves | Model catches ≥ 80 % of LKH's gain "
             "(of the windows LKH improves) | Share of LKH's gain the model catches | ms per window: network / beam / LKH |",
             "|---|---|---|---|---|---|---|"]
    for n in SIZES:
        logs = logs_at(R, n)
        if logs is None:
            lines.append(f"| {n} | missing | | | | | |")
            continue
        imp, catch, mimp, share = [], [], [], []
        for x in logs:
            li, ca = targets(x)
            lkh_gain, model_gain = x["cur_len"] - x["lkh_len"], x["cur_len"] - x["model_len"]
            imp.append(100 * li.mean()); catch.append(100 * ca[li].mean()); mimp.append(100 * (model_gain > 1e-9).mean())
            share.append(100 * np.maximum(model_gain, 0).sum() / lkh_gain[li].sum())
        P = pooled(logs)
        ms = " / ".join(f"{P[f].mean():.2f}" for f in ("ms_net", "ms_beam", "ms_lkh"))
        lines.append(f"| {n} | {len(logs[0]['start'])} | {fmt(imp, 1)} % | {fmt(mimp, 1)} % | {fmt(catch, 1)} % | "
                     f"{fmt(share, 1)} % | {ms} |")
    return "\n".join(lines)


def table_signal_quality(R):
    """Average precision (and its base rate) and Spearman correlation of each signal, mean ± std over seeds."""
    lines = ["| Target | Signal (higher = more likely) | " + " | ".join(f"AP n = {n}" for n in SIZES) + " | "
             + " | ".join(f"ρ n = {n}" for n in SIZES) + " |", "|---|---|" + "---|" * (2 * len(SIZES))]
    for target, name in (("i", "(i) LKH improves"), ("ii", "(ii) model catches ≥ 80 %")):
        names = list(signals({k: np.ones(2) for k in ("cur_len", "model_len", "lkh_len", "disagreement", "logp",
                                                        "margin")})[target])
        rows = {s: [] for s in ["base rate", *names]}
        for n in SIZES:
            logs = logs_at(R, n)
            for s in rows:
                if logs is None:
                    rows[s].append(("missing", "missing"))
                    continue
                aps, rhos = [], []
                for x in logs:
                    li, ca = targets(x)
                    mask = li if target == "ii" else np.ones_like(li)
                    y = (li if target == "i" else ca)[mask]
                    if s == "base rate":
                        aps.append(y.mean())
                        continue
                    score = signals(x)[target][s][mask]
                    aps.append(average_precision(score, y))
                    # correlation with the continuous version: LKH's relative gain, or the share the model catches
                    lkh_gain = (x["cur_len"] - x["lkh_len"])[mask]
                    cont = lkh_gain / x["cur_len"][mask] if target == "i" else \
                        np.clip((x["cur_len"] - x["model_len"])[mask] / lkh_gain, -1, 1)
                    rhos.append(spearmanr(score, cont).statistic)
                rows[s].append((fmt(aps, 2), fmt(rhos, 2) if rhos else "—"))
        for s, cells in rows.items():
            lines.append(f"| {name} | {s} | " + " | ".join(c[0] for c in cells) + " | " + " | ".join(c[1] for c in cells) + " |")
    return "\n".join(lines)


def table_deciles(R):
    """Deciles of disagreement (of each seed's own windows at that size), pooled over seeds."""
    head = " | ".join(f"n = {n}: LKH improves / model catches / LKH gain share" for n in SIZES)
    lines = ["| Disagreement decile | " + head + " |", "|---|" + "---|" * len(SIZES)]
    cells = {d: [] for d in range(10)}
    for n in SIZES:
        logs = logs_at(R, n)
        for d in range(10):
            if logs is None:
                cells[d].append("missing")
                continue
            imp, catch, share = [], [], []
            for x in logs:
                li, ca = targets(x)
                edges = np.quantile(x["disagreement"], np.linspace(0, 1, 11))
                dec = np.clip(np.searchsorted(edges, x["disagreement"], side="right") - 1, 0, 9)
                sel = dec == d
                lkh_gain = np.where(li, x["cur_len"] - x["lkh_len"], 0)
                imp.append(100 * li[sel].mean())
                catch.append(100 * ca[sel & li].mean() if (sel & li).any() else np.nan)
                share.append(100 * lkh_gain[sel].sum() / lkh_gain.sum())
            cells[d].append(f"{np.mean(imp):.0f} % / {np.nanmean(catch):.0f} % / {np.mean(share):.0f} %")
    for d in range(10):
        label = "1 (lowest)" if d == 0 else "10 (highest)" if d == 9 else str(d + 1)
        lines.append(f"| {label} | " + " | ".join(cells[d]) + " |")
    return "\n".join(lines)


def table_confidence(R):
    """Quintiles of logp: where can the model replace LKH?"""
    head = " | ".join(f"n = {n}: catches / LKH gain left" for n in SIZES)
    lines = ["| logp quintile | " + head + " |", "|---|" + "---|" * len(SIZES)]
    cells = {q: [] for q in range(5)}
    for n in SIZES:
        logs = logs_at(R, n)
        for q in range(5):
            if logs is None:
                cells[q].append("missing")
                continue
            catch, left = [], []
            for x in logs:
                li, ca = targets(x)
                edges = np.quantile(x["logp"], np.linspace(0, 1, 6))
                qq = np.clip(np.searchsorted(edges, x["logp"], side="right") - 1, 0, 4)
                sel = qq == q
                lkh_gain = np.where(li, x["cur_len"] - x["lkh_len"], 0)
                model_gain = np.maximum(x["cur_len"] - x["model_len"], 0)
                catch.append(100 * ca[sel & li].mean() if (sel & li).any() else np.nan)
                left.append(100 * (lkh_gain[sel] - np.minimum(model_gain[sel], lkh_gain[sel])).sum() / lkh_gain.sum())
            cells[q].append(f"{np.nanmean(catch):.0f} % / {np.mean(left):.0f} %")
    for q in range(5):
        label = "1 (least confident)" if q == 0 else "5 (most confident)" if q == 4 else str(q + 1)
        lines.append(f"| {label} | " + " | ".join(cells[q]) + " |")
    return "\n".join(lines)


def table_thresholds(R):
    lines = ["| Seed | Disagreement top 10 / 20 / 30 / 50 % | Confidence signal (AP logp / margin) | Confidence threshold | "
             + " | ".join(f"Windows above top-20 % threshold, n = {n}" for n in SIZES) + " |",
             "|---|---|---|---|" + "---|" * len(SIZES)]
    for s in SEEDS:
        r = next((R["pol"][s, n] for n in SIZES if R["pol"][s, n] is not None), None)
        if r is None:
            lines.append(f"| {s} | missing | | |" + " |" * len(SIZES))
            continue
        th = r[0]["meta"]["thresholds"]
        dis = " / ".join(f"{th['dis'][str(q)]:.3f}" for q in P2_TOP)
        above = []
        for n in SIZES:
            x = R["logs"][s, n]
            above.append(f"{100 * (x[1]['disagreement'] >= th['dis']['20']).mean():.1f} %" if x else "missing")
        lines.append(f"| {s} | {dis} | {th['conf_signal']} ({th['conf_ap']['logp']:.2f} / {th['conf_ap']['margin']:.2f}) | "
                     f"{th['conf']:.3f} | " + " | ".join(above) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- tables: policies

def policy_point(R, p, n):
    """(gap, total seconds, LKH fraction, model fraction) per run, or None."""
    runs = policy_runs(R, p, n)
    if runs is None:
        return None
    return {"gap": [float(r[f"{p}__gap"].mean()) for r in runs],
            "s": [float((r["initial__ms"] + r[f"{p}__ms"]).mean()) / 1000 for r in runs],
            "lkh": [100 * r[f"{p}__to_lkh"].sum() / r[f"{p}__windows"].sum() for r in runs],
            "model": [100 * r[f"{p}__to_model"].sum() / r[f"{p}__windows"].sum() for r in runs],
            "gain": [float(r[f"{p}__gain"].mean()) for r in runs]}


def table_policies(R, n):
    ini = initial(R, n)
    if ini is None:
        return "missing"
    p1 = policy_point(R, "P1", n)
    lines = ["| Policy | Gap (%) | Gain (pp) | Share of P1's gain | Total time (s) | Time / P1 | Windows to LKH | Windows to model |",
             "|---|---|---|---|---|---|---|---|",
             f"| Greedy edge + 2-opt (initial tour) | {ini['initial__gap'].mean():.2f} | — | — | "
             f"{ini['initial__ms'].mean() / 1000:.2f} | — | — | — |"]
    for p in ["P1", *POLICIES]:
        pt = policy_point(R, p, n)
        if pt is None:
            lines.append(f"| {LABELS[p]} | missing | | | | | | |")
            continue
        rel = lambda key: f"{100 * np.mean(pt[key]) / np.mean(p1[key]):.0f} %" if p1 else "missing"
        lines.append(f"| {LABELS[p]} | {fmt(pt['gap'])} | {fmt(pt['gain'])} | {rel('gain')} | {fmt(pt['s'])} | "
                     f"{rel('s')} | {fmt(pt['lkh'], 1)} % | {fmt(pt['model'], 1)} % |")
    return "\n".join(lines)


def run_tests(R):
    """{(n, policy, reference): result}, final gaps paired by instance, n with >= 20 instances."""
    tests = {}
    for n in SIZES:
        ini = initial(R, n)
        if ini is None or len(ini["initial__gap"]) < MIN_TEST:
            continue
        family = []
        for ref in ("P0", "P1"):
            ref_runs = policy_runs(R, ref, n)
            for p in [q for q in POLICIES if q != ref]:
                runs = policy_runs(R, p, n)
                if runs is None or ref_runs is None:
                    continue
                refs = ref_runs * len(runs) if ref == "P1" else ref_runs
                family.append(((n, p, ref), [r[f"{p}__gap"] for r in runs], [r[f"{ref}__gap"] for r in refs]))
        for key, a, ref in family:
            tests[key] = {**beam.paired(a, ref, ALPHA / len(family)), "threshold": ALPHA / len(family),
                          "family": len(family)}
    return tests


def table_tests(R, tests):
    sizes = sorted({k[0] for k in tests})
    if not sizes:
        return "No size with at least 20 instances has complete results yet."
    cols = [(n, ref) for n in sizes for ref in ("P0", "P1")]
    lines = ["| Policy | " + " | ".join(f"n = {n} vs. {ref}" for n, ref in cols) + " |", "|---|" + "---|" * len(cols)]
    for p in POLICIES:
        cells = []
        for n, ref in cols:
            t = tests.get((n, p, ref))
            if t is None:
                cells.append("—")
                continue
            v = f"**{t['verdict']}**" if t["verdict"] != "n.s." else "n.s."
            cells.append(f"{v} {t['mean_diff']:+.2f} (p {t['p_max']:.1e})")
        lines.append(f"| {LABELS[p]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Each cell: verdict, mean gap difference policy − reference (pp; negative = policy shorter), largest p over "
                 "the 3 seeds. Bonferroni within each size: " +
                 "; ".join(f"n = {n}: 0.05/{t['family']} = {t['threshold']:.4f}"
                           for n, t in sorted({k[0]: v for k, v in tests.items()}.items())) + ".")
    return "\n".join(lines)


def table_small(R):
    sizes = [n for n in SIZES if (ini := initial(R, n)) is not None and len(ini["initial__gap"]) < MIN_TEST]
    if not sizes:
        return "No size with fewer than 20 instances has results yet."
    lines = ["| Policy | " + " | ".join(f"n = {n} vs. {ref}" for n in sizes for ref in ("P0", "P1")) + " |",
             "|---|" + "---|" * (2 * len(sizes))]
    for p in POLICIES:
        cells = []
        for n in sizes:
            runs = policy_runs(R, p, n)
            for ref in ("P0", "P1"):
                ref_runs = policy_runs(R, ref, n)
                if runs is None or ref_runs is None or p == ref:
                    cells.append("—" if p == ref else "missing")
                    continue
                refs = ref_runs * len(runs) if ref == "P1" else ref_runs
                diff = np.concatenate([r[f"{p}__gap"] - q[f"{ref}__gap"] for r, q in zip(runs, refs)])
                cells.append(f"{diff.mean():+.2f} ({int((diff < -1e-6).sum())}/{len(diff)} shorter)")
        lines.append(f"| {LABELS[p]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- figures

def plot_pr(R, figs_dir):
    fig, axes = plt.subplots(2, len(SIZES), figsize=(14, 7.5), sharex=True, sharey=True)
    for row, (target, title) in enumerate((("i", "(i) LKH improves the window"),
                                           ("ii", "(ii) model catches ≥ 80 % of LKH's gain (windows LKH improves)"))):
        for col, n in enumerate(SIZES):
            ax = axes[row, col]
            logs = logs_at(R, n)
            if logs is None:
                ax.text(0.5, 0.5, "missing", ha="center", color=MUTED)
                continue
            P = pooled(logs)
            li, ca = targets(P)
            mask = li if target == "ii" else np.ones_like(li)
            y = (li if target == "i" else ca)[mask]
            for s, score in signals(P)[target].items():
                color = SIGNAL_COLORS[s.lstrip("-")]
                precision, recall = pr_curve(score[mask], y)
                ax.plot(recall, precision, color=color, linewidth=2,
                        label=f"{s} (AP {average_precision(score[mask], y):.2f})")
            ax.axhline(y.mean(), color=MUTED, linestyle="--", linewidth=1)
            ax.text(0.99, y.mean() + 0.02, f"base rate {y.mean():.2f}", ha="right", color=MUTED, fontsize=7)
            summarize.style_axes(ax)
            ax.set_ylim(0, 1.02)
            ax.legend(frameon=False, fontsize=7, loc="upper right")
            ax.set_title(f"{title.split(' (')[0] if col else title}  ·  n = {n}", color=INK, fontsize=8, loc="left")
            if row == 1:
                ax.set_xlabel("Recall", color=INK)
            if col == 0:
                ax.set_ylabel("Precision", color=INK)
    fig.suptitle("Precision–recall of each window signal (3 seeds pooled, w = 20, first pass)", color=INK,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(figs_dir, exist_ok=True)
    summarize.save(fig, figs_dir, "phase3_triage_pr")


def plot_pareto(R, figs_dir):
    fig, axes = plt.subplots(1, len(SIZES), figsize=(15, 4.8))
    for ax, n in zip(axes, SIZES):
        ini = initial(R, n)
        if ini is None:
            ax.text(0.5, 0.5, "missing", ha="center", color=MUTED, transform=ax.transAxes)
            continue
        ax.plot(ini["initial__ms"].mean() / 1000, ini["initial__gap"].mean(), "s", color=INK, markersize=8,
                label="Greedy edge + 2-opt")
        for fam, members in (("P2", [f"P2_top{q}" for q in P2_TOP]), ("P3", [f"P3_top{q}" for q in P3_TOP]),
                             ("P0", ["P0"]), ("P1", ["P1"])):
            pts = [(q, policy_point(R, q, n)) for q in members]
            pts = [(q, pt) for q, pt in pts if pt is not None]
            if not pts:
                continue
            x = [np.mean(pt["s"]) for _, pt in pts]
            y = [np.mean(pt["gap"]) for _, pt in pts]
            label = {"P2": "P2: LKH on high disagreement", "P3": "P3: model if confident, else LKH if high dis.",
                     "P0": "P0: model everywhere", "P1": "P1: LKH everywhere"}[fam]
            ax.plot(x, y, "-o" if len(pts) > 1 else "o", color=POLICY_COLORS[fam], linewidth=2, markersize=8, label=label)
            for (q, _), xx, yy in zip(pts, x, y):
                if "_top" in q:
                    ax.annotate(f"{q.split('_top')[1]} %", (xx, yy), textcoords="offset points", xytext=(6, 4),
                                fontsize=7, color=INK)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        summarize.style_axes(ax)
        ax.set_title(f"n = {n}", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("Total time per instance (s, log scale)", color=INK)
    axes[0].set_ylabel("Gap to LKH (%)", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Gap vs. time of the triage policies (w = 20, ≤ 3 passes; mean over 3 seeds; labels = disagreement top %)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs(figs_dir, exist_ok=True)
    summarize.save(fig, figs_dir, "phase3_triage_pareto")


# ---------------------------------------------------------------- report

def build_report(results, figs_dir=None):
    R = load_all(results)
    figs_dir = figs_dir or os.path.join(results, "figs")
    tests = run_tests(R)
    plot_pr(R, figs_dir)
    plot_pareto(R, figs_dir)
    notes_path = os.path.join(results, "phase3_triage_notes.md")
    notes = open(notes_path).read().strip() if os.path.exists(notes_path) else "(to be written after reading the results)"
    missing = [f"phase3_triage/log/tagcn_k3_seed{s}/{ds(n)}" for (s, n), r in R["logs"].items() if r is None]
    missing += [f"phase3_triage/policies/{'lkh' if s == 'lkh' else f'tagcn_k3_seed{s}'}/{ds(n)}"
                for (s, n), r in R["pol"].items() if r is None]
    commits = sorted({r[0]["meta"]["git_commit"] for d in (R["logs"], R["pol"]) for r in d.values() if r is not None})
    lines = [
        "# Phase 3 — window triage: nothing, the model, or LKH?",
        "",
        f"Generated by `scripts/phase3_triage_report.py` on {datetime.date.today().isoformat()} "
        f"(results from commit(s) {', '.join(commits) or '—'}).",
        "",
        "## Three lines",
        "",
        notes,
        "",
        "## Setup",
        "",
        "- **Tour and windows:** greedy edge + 2-opt tour (the baseline of the scale experiment). Windows of "
        "w = 20 consecutive tour cities, ends fixed, a window replaced only if shorter (as in the window experiment).",
        "- **Model:** TAGCN K=3 (best.pt, seeds 0–2), beam B = 10 paths.",
        "- **Record (first pass):** every window of the initial tour at offsets 0 and w/2, each evaluated alone. "
        "Signals, all on the window rescaled to [0, 1]²:",
        "  - **disagreement** = 1 − mean model probability of the 19 edges of the current segment (an edge outside "
        "the model's k-NN graph counts as 0);",
        "  - **logp** = mean log-probability per edge of the returned beam path;",
        "  - **margin** = highest − second highest final beam score, per edge;",
        "  - **model gain** = relative shortening by the beam path (it needs the beam, so it is not free; for (i) it is trivially exact when positive, since LKH is at least as good, hence the flat start of its curve).",
        "- **Targets:** (i) LKH shortens the window; (ii) the beam path catches ≥ 80 % of LKH's gain, "
        "defined only on the windows of (i).",
        "- **Policies** (≤ 3 passes, each pass = offsets 0 and w/2):",
        "  - P0: beam everywhere; P1: LKH everywhere;",
        "  - P2: LKH on windows with disagreement above the tsp200 top-q % threshold, nothing elsewhere "
        "(needs only the network);",
        "  - P3: LKH where disagreement is above the top-q % threshold and confidence below its threshold; beam path "
        "where confidence is above it; nothing elsewhere. Confidence needs the beam on every window, so P3 pays the "
        "decoding everywhere, and keeping only the confident beam paths cannot be faster than P0.",
        "- **Thresholds** are chosen on the tsp200 record of each seed and used as fixed numbers on tsp1000 and "
        "tsp10000. The confidence signal is the one (logp or margin) with the higher average precision for (ii); "
        "its threshold maximises F1 for (ii). tsp200 is therefore a validation set: its policy results are in-sample.",
        "- **Time** = initial tour + refinement (network + beam decoding + LKH), per instance, with the network on "
        "the shared T4 GPU and everything else in single-thread Python/LKH.",
        "",
        "## 1. How often a window can be improved",
        "",
        table_base_rates(R),
        "",
        "## 2. Does a signal separate the windows?",
        "",
        "Average precision (AP; a random signal scores the base rate) and Spearman ρ with the continuous version of "
        "the target (LKH's relative gain for (i); the share of LKH's gain the model catches, clipped to [−1, 1], for "
        "(ii)). Mean ± std over the 3 seeds.",
        "",
        table_signal_quality(R),
        "",
        "![Precision-recall curves](figs/phase3_triage_pr.png)",
        "",
        "### By decile of disagreement",
        "",
        "Per decile: fraction of windows LKH improves / fraction of those the model catches (≥ 80 %) / share of "
        "all of LKH's gain that lies in this decile. Deciles of each seed's own windows, averaged over seeds.",
        "",
        table_deciles(R),
        "",
        "### By quintile of model confidence (logp)",
        "",
        "Per quintile: fraction of the windows LKH improves that the model catches / LKH gain the model leaves "
        "behind in this quintile, as a share of all of LKH's gain.",
        "",
        table_confidence(R),
        "",
        "## 3. Triage policies",
        "",
        "### Thresholds (from tsp200) and how they transfer",
        "",
        table_thresholds(R),
        "",
    ]
    for n in SIZES:
        lines += [f"### n = {n}", "", table_policies(R, n), ""]
    lines += [
        "Mean ± std over the 3 seeds (P1 and the initial tour have no seed). Share of P1's gain and time / P1 use "
        "the means.",
        "",
        "![Gap vs. time](figs/phase3_triage_pareto.png)",
        "",
        "### Paired tests (sizes with ≥ 20 instances)",
        "",
        table_tests(R, tests),
        "",
        "### Sizes with fewer than 20 instances (no test)",
        "",
        "Mean gap difference policy − reference (pp) and the instances where the policy is shorter, over 3 seeds.",
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
    parser = argparse.ArgumentParser(description="Phase 3 window triage report.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/phase3_triage_report.md")
    args = parser.parse_args()
    text = build_report(args.results)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
