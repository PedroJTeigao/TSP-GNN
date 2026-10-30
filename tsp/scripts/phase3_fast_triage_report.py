"""Report of the cheap window scoring for the triage policies (phase 3).

Reads results/phase3_fast_triage/: check/<set>.json (equivalence and scoring time on a sample),
global_features/ and global_models/ (version B), b_decision.json, e2/ (policies with the batched scores, run
together with random) and e2_b/ (version B policies, only if B passed the Spearman check); the per-window policies
and the oracle come from the learned-triage run (results/phase3_learned_triage/e2). Opening notes:
results/phase3_fast_triage_notes.md. Writes results/phase3_fast_triage_report.md and a copy in docs/.

Example:
    python scripts/phase3_fast_triage_report.py
"""

import datetime
import importlib.util
import json
import os
import shutil

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("summarize", os.path.join(_here, "summarize.py"))
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)

ROOT = "results/phase3_fast_triage"
OLD = "results/phase3_learned_triage/e2"
OUT = "results/phase3_fast_triage_report.md"
DOCS = "docs/phase3_fast_triage_report.md"
NOTES = "results/phase3_fast_triage_notes.md"
SETS = ("tsp1000_test", "tsp10000_test_concorde")
SEEDS = (0, 1, 2)
RANDOM_SEEDS = (0, 1, 2, 3, 4)
FRACS = (20, 50)
ALPHA = 0.05
# (label, root, folder pattern, policy key pattern)
POLICIES = {
    "random": ("Random (5 seeds)", ROOT + "/e2", "random", "random_s{s}_top{x}"),
    "fast_disagreement": ("Disagreement, batched A", ROOT + "/e2", "fast_disagreement_seed{s}",
                          "fast_disagreement_top{x}"),
    "fast_learned": ("Learned, batched A", ROOT + "/e2", "fast_learned_seed{s}", "fast_learned_top{x}"),
    "global_disagreement": ("Disagreement, global B", ROOT + "/e2_b", "global_disagreement_seed{s}",
                            "global_disagreement_top{x}"),
    "global_learned": ("Learned (retrained), global B", ROOT + "/e2_b", "global_learned_seed{s}",
                       "global_learned_top{x}"),
    "disagreement": ("Disagreement, per window (previous run)", OLD, "disagreement_seed{s}", "disagreement_top{x}"),
    "learned": ("Learned, per window (previous run)", OLD, "learned_seed{s}", "learned_top{x}"),
    "oracle": ("Oracle: real LKH gain (previous run)", OLD, "oracle", "oracle_top{x}"),
}


def npz(path):
    return dict(np.load(path)) if os.path.exists(path) else None


def runs_of(name, key, x):
    label, root, folder, pk = POLICIES[key]
    if key == "random":
        r = npz(os.path.join(root, folder, f"{name}.npz"))
        return [] if r is None else [(r, pk.format(s=s, x=x)) for s in RANDOM_SEEDS
                                     if pk.format(s=s, x=x) + "__gap" in r]
    if key == "oracle":
        r = npz(os.path.join(root, folder, f"{name}.npz"))
        return [] if r is None else [(r, pk.format(x=x))]
    runs = [(npz(os.path.join(root, folder.format(s=s), f"{name}.npz")), pk.format(x=x)) for s in SEEDS]
    return [(r, k) for r, k in runs if r is not None]


def policy(name, key, x):
    runs = runs_of(name, key, x)
    if not runs:
        return None
    gaps = np.array([r[f"{k}__gap"] for r, k in runs])
    return {"per_inst": gaps.mean(axis=0), "run_means": gaps.mean(axis=1), "per_run": gaps,
            "s": float(np.mean([((r["initial__ms"] + r[f"{k}__ms"]) / 1000).mean() for r, k in runs])),
            "score_s": float(np.mean([r[f"{k}__ms_score"].mean() / 1000 for r, k in runs])),
            "net_s": float(np.mean([r[f"{k}__ms_net"].mean() / 1000 for r, k in runs])),
            "lkh": float(np.mean([r[f"{k}__to_lkh"].sum() / r[f"{k}__windows"].sum() for r, k in runs])),
            "initial": float(runs[0][0]["initial__gap"].mean()), "runs": len(runs)}


def ms(v, d=3):
    return f"{np.mean(v):.{d}f}" if len(v) == 1 else f"{np.mean(v):.{d}f} ± {np.std(v):.{d}f}"


# ---------------------------------------------------------------------------------------------------- tables

def table_check(B):
    lines = ["| Set | Version | Spearman vs per-window (mean, min) | Top-20 % overlap (mean, min) | "
             "Largest |difference| | Passes to the time table |", "|---|---|---|---|---|---|"]
    for name in SETS:
        p = os.path.join(ROOT, "check", f"{name}.json")
        if not os.path.exists(p):
            lines.append(f"| {name} | missing | | | | |")
            continue
        s = json.load(open(p))["summary"]
        for v in ("A disagreement", "A learned", "B disagreement", "B learned"):
            if v not in s:
                continue
            ok = "yes" if v.startswith("A") else ("yes" if B.get(v, {}).get("pass") else "**no** (Spearman < 0.5)")
            lines.append(f"| {name} | {v} | {s[v]['spearman']['mean']:.4f}, {s[v]['spearman']['min']:.4f} | "
                         f"{100 * s[v]['top20_overlap']['mean']:.1f} %, {100 * s[v]['top20_overlap']['min']:.1f} % | "
                         f"{s[v]['max_abs_diff']['max']:.1e} | {ok} |")
    lines += ["", "Sample: the first 2 instances of each set, 3 TAGCN seeds, both half-passes of the first pass on the "
              "greedy + 2-opt tour. B learned is the logistic model retrained on global features (same protocol as "
              "the learned triage); it is compared with the current learned score."]
    return lines


def table_scoring_time():
    lines = ["| Set | Per window (disagreement + learned) | A disagreement | A learned | B disagreement | B learned | "
             "Tour edges outside the global graph |", "|---|---|---|---|---|---|---|"]
    for name in SETS:
        p = os.path.join(ROOT, "check", f"{name}.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))["summary"]
        t = s["times_s"]
        cell = lambda k: f"{t[k]:.3f}" if k in t else "—"
        lines.append(f"| {name} | {cell('per-window (both scores)')} | {cell('A disagreement')} | {cell('A learned')} | "
                     f"{cell('B disagreement')} | {cell('B learned')} | {100 * s['global_missing']:.2f} % |")
    lines += ["", "Seconds per instance to score every window of both half-passes of one pass (network included; one "
              "process alone, not under the 8-process load of the policy runs). The per-window column computes the "
              "heatmaps once for both scores (the learned features add ~10 %). B includes the global k-NN graph and "
              "the network pass."]
    return lines


def table_identity():
    lines = ["| Set | x | Batched A vs per-window | Instances × seeds with the same gap | Largest gap difference (pp) |",
             "|---|---|---|---|---|"]
    for name in SETS:
        for x in FRACS:
            for new, old in (("fast_disagreement", "disagreement"), ("fast_learned", "learned")):
                a, b = policy(name, new, x), policy(name, old, x)
                if a is None or b is None:
                    continue
                d = np.abs(a["per_run"] - b["per_run"])
                lines.append(f"| {name} | {x} % | {POLICIES[new][0]} | {int((d < 1e-9).sum())} / {d.size} | "
                             f"{d.max():.1e} |")
    return lines


def table_policy(keys):
    lines = ["| Set | x | Policy | Gap (%) | Share of oracle gain | Time (s) | of which scoring (s) | "
             "of which network (s) |", "|---|---|---|---|---|---|---|---|"]
    for name in SETS:
        for x in FRACS:
            orc = policy(name, "oracle", x)
            for key in keys:
                p = policy(name, key, x)
                if p is None:
                    continue
                gain = p["initial"] - p["per_inst"].mean()
                share = f"{100 * gain / (orc['initial'] - orc['per_inst'].mean()):.0f} %" if orc else "—"
                gap = f"{p['per_inst'].mean():.3f}" + (f" ± {np.std(p['run_means']):.3f}" if p["runs"] > 1 else "")
                star = " *" if key == "oracle" else ""
                lines.append(f"| {name} | {x} % | {POLICIES[key][0]} | {gap} | {share} | {p['s']:.2f}{star} | "
                             f"{p['score_s']:.2f} | {p['net_s']:.2f} |")
    lines += ["", "w = 20, ≤ 3 passes, top x % of the windows of each half-pass to LKH (E2 protocol). Random and the "
              "batched scores ran together (7 processes); the per-window and oracle rows are from the learned-triage "
              "run (8 processes, same machine, same protocol): their gaps are deterministic, their times carry that "
              "run's load. * the oracle runs LKH on every window: its time is not that of a usable policy."]
    return lines


def table_same_time(keys):
    lines = ["| Set | Policy (sorted by time) | Time (s) | Gap (%) |", "|---|---|---|---|"]
    for name in SETS:
        rows = []
        for key in keys:
            for x in FRACS:
                p = policy(name, key, x)
                if p is not None:
                    rows.append((p["s"], p["per_inst"].mean(), f"{POLICIES[key][0]}, top {x} %"))
        for sec, gap, label in sorted(rows):
            lines.append(f"| {name} | {label} | {sec:.2f} | {gap:.3f} |")
    return lines


def table_tests():
    pairs = [("fast_learned", 20, "random", 50), ("fast_learned", 50, "random", 50),
             ("fast_learned", 20, "random", 20), ("fast_learned", 20, "fast_disagreement", 20),
             ("fast_learned", 50, "fast_disagreement", 50)]
    lines = ["| Set | Policy | vs. | Better / equal / worse (% inst.) | Mean diff (pp) | p (Wilcoxon) | Verdict |",
             "|---|---|---|---|---|---|---|"]
    for name in SETS:
        for a_key, ax, b_key, bx in pairs:
            a, b = policy(name, a_key, ax), policy(name, b_key, bx)
            if a is None or b is None:
                continue
            t = summarize.paired_compare(a["per_inst"], b["per_inst"])
            thr = ALPHA / len(pairs)
            verdict = "n.s." if t["p"] >= thr else ("**better**" if t["median_diff"] < 0 or
                                                    (t["median_diff"] == 0 and t["mean_diff"] < 0) else "**worse**")
            lines.append(f"| {name} ({len(a['per_inst'])}) | {POLICIES[a_key][0]} top {ax} % | "
                         f"{POLICIES[b_key][0]} top {bx} % | {t['better']:.0f} / {t['equal']:.0f} / {t['worse']:.0f} | "
                         f"{t['mean_diff']:+.3f} | {t['p']:.1e} | {verdict} |")
    lines += ["", f"Paired two-sided Wilcoxon on the per-instance gaps; Bonferroni {ALPHA} / {len(pairs)} within each set."]
    return lines


def main():
    B = json.load(open(os.path.join(ROOT, "b_decision.json"))) if os.path.exists(os.path.join(ROOT, "b_decision.json")) else {}
    notes = open(NOTES).read().strip() if os.path.exists(NOTES) else f"(notes missing: {NOTES})"
    keys = ["random", "fast_disagreement", "fast_learned"] + [k for k in ("global_disagreement", "global_learned")
                                                            if B.get("B " + k.split("_")[1], {}).get("pass")]
    lines = ["# Phase 3 — cheap window scoring for the triage", "",
             f"Generated {datetime.date.today().isoformat()} by scripts/phase3_fast_triage_report.py. TAGCN K=3 (3 "
             "seeds), w = 20, LKH per window; tsp1000 (20 instances) and Fu-10k (16).", "", notes, "",
             "## Scoring time", "", *table_scoring_time(), "",
             "## Equivalence with the per-window scores", "", *table_check(B), "",
             "### Policy outcome: batched A against per-window", "", *table_identity(), "",
             "## Policies (E2 protocol)", "", *table_policy(keys + ["disagreement", "learned", "oracle"]), "",
             "### Same time", "", *table_same_time(keys), "",
             "### Paired tests", "", *table_tests(), ""]
    open(OUT, "w").write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(DOCS), exist_ok=True)
    shutil.copy(OUT, DOCS)
    print(f"wrote {OUT} and {DOCS}")


if __name__ == "__main__":
    main()
