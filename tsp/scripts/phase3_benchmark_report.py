"""Phase 3 benchmark report: our windows and triage against LKH-3 at equal time and the published numbers.

Reads (all under results/phase3_benchmark/):
    ref_lkh3.npz                                    E0, LKH-3 reference (scripts/lkh3_reference.py)
    e1/<set>.{npz,json}                             E1, LKH-3 at equal time (scripts/lkh3_equal_time.py)
    e3/policies/{tagcn_k3_seed<s>,lkh}/<set>.npz    E3, triage policies and P1 (scripts/triage_eval.py)
    e3/window/tagcn_k3_seed<s>/<set>.npz            E3, model windows w = 20 (scripts/window_eval.py)
    e2/<score>[_seed<s>]/<set>.npz                  E2, ranked triage ablation (scripts/triage_ablation.py)
and the hand-written three sentences of results/phase3_benchmark_notes.md. The published numbers (PUBLISHED)
are copied from docs/literature_review_tsp.md and are not reproduced.
Writes results/phase3_benchmark_report.md and results/figs/phase3_benchmark_pareto.{png,pdf}.
Missing results are reported as "missing".

Example:
    python scripts/phase3_benchmark_report.py
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
INK, MUTED, ALPHA = summarize.INK, summarize.MUTED, summarize.ALPHA

SEEDS = (0, 1, 2)
RANDOM_SEEDS = (0, 1, 2, 3, 4)
FU = {1000: "tsp1000_test_concorde", 10000: "tsp10000_test_concorde"}      # the standard sets (E3)
OWN = {1000: "tsp1000_test", 10000: "tsp10000_test"}                        # ours (phase 2-3)
REF_SETS = ("tsp1000_test", "tsp1000_test_concorde", "tsp10000_test", "tsp10000_test_concorde")
E2_SETS = ("tsp1000_test", "tsp1000_test_concorde", "tsp10000_test_concorde")
BUDGETS = {1000: (0.72, 1.25, 1.54, 1.94), 10000: (8.4, 14.2, 17.5, 21.5)}
MAIN_LKH = "quadrant5"                       # the main LKH-3 reference line (start = LKH's own tour)
OVER = 1.10                                  # "over budget": total time > 1.10 × budget
OVER_SHARE = 0.25                            # ... on more than 25 % of the instances
P2_TOP = (10, 20, 30, 50)
E2_FRACS = (20, 50)
HARDWARE = "Xeon E5-2620 v3 (8 cores, shared, nice 10) + T4 (< 5.5 GiB)"
REFERENCE = "LKH-3.0.6, E0 (1k: RUNS = 10; 10k: RUNS = 5, ≤ 696 s per run)"

COLORS = {"ours": "#2a78d6", "lkh": "#1baf7a", "published": "#eb6834", "hybrid": "#6250d6"}

VARIANT_LABELS = {"alpha": "ALPHA (default)", "popmusic": "POPMUSIC", "nn5": "NEAREST-NEIGHBOR 5",
                  "nn8": "NEAREST-NEIGHBOR 8", "alpha_nosub": "ALPHA, SUBGRADIENT = NO", "quadrant5": "QUADRANT 5",
                  "quadrant8": "QUADRANT 8", "heat5": "heatmap top 5", "heat8": "heatmap top 8",
                  "nnheat5": "NN 5 ∪ heatmap top 5"}


def variant_label(v):
    """alpha_p<k> = ALPHA with INITIAL_PERIOD = k (lkh3_e1_prep.variants: n/10 and n/100)."""
    return f"ALPHA, INITIAL_PERIOD = {v[len('alpha_p'):]}" if v.startswith("alpha_p") else VARIANT_LABELS.get(v, v)


# Published numbers (docs/literature_review_tsp.md, section A): not reproduced, other hardware and reference.
# gap / time: as reported (text); g1k, g10k, t1k, t10k: the point drawn in E4 (time per instance in s;
# a time reported for the whole test set is divided by its 128 / 16 instances); None = not drawn.
PUBLISHED = [
    {"name": "Fu et al. 2021 (Att-GCRN + MCTS)", "short": "Fu 2021", "target": "no (TSP-20/50 subgraphs)",
     "gap": ("3.22", "4.39"), "time": ("44 s net + 11.7 min MCTS", "4.2 min + 1.69 h"),
     "hw": "GTX 1080Ti + Xeon Gold 5118 (8 cores)", "ref": "Concorde / LKH-3",
     "g1k": 3.22, "t1k": 44 + 11.7 * 60, "g10k": 4.39, "t10k": 4.2 * 60 + 1.69 * 3600},
    {"name": "Select & Optimize (Cheng et al. 2023)", "short": "S&O 2023", "target": "yes (TSP-200/500/1000)",
     "gap": ("2.80", "3.52"), "time": ("2.9 s + 23 s", "40 s + 418 s"),
     "hw": "V100; LKH-3/MCTS on Xeon E5-2650", "ref": "LKH-3",
     "g1k": 2.80, "t1k": 2.9 + 23, "g10k": 3.52, "t10k": 40 + 418},
    {"name": "H-TSP 2023", "short": "H-TSP", "target": "yes (up to 10k)",
     "gap": ("6.62 (4.06 with LKH-3 lower level)", "7.32 (5.57 with LKH-3)"), "time": ("0.33 s", "3.3 s"),
     "hw": "V100 16 GB", "ref": "LKH-3", "g1k": 6.62, "t1k": 0.33, "g10k": 7.32, "t10k": 3.3},
    {"name": "GLOP 2024 (best reviser setting)", "short": "GLOP", "target": "no (revisers on small SHPP)",
     "gap": ("3.11 (3.85)", "4.90 (5.36)"), "time": ("3 min (set)", "1.8 min (set)"),
     "hw": "RTX 3090 + Xeon (12 cores)", "ref": "LKH-3",
     "g1k": 3.11, "t1k": 3 * 60 / 128, "g10k": 4.90, "t10k": 1.8 * 60 / 16},
    {"name": "DIFUSCO 2023, greedy + 2-opt (+ MCTS)", "short": "DIFUSCO", "target": "yes (per size)",
     "gap": ("1.90 (1.17)", "3.10 (2.58)"), "time": ("24 min (set)", "47 min (set)"),
     "hw": "8× V100 train; 1 GPU test", "ref": "LKH-3",
     "g1k": 1.90, "t1k": 24 * 60 / 128, "g10k": 3.10, "t10k": 47 * 60 / 16},
    {"name": "SoftDist + MCTS (Xia et al. 2024)", "short": "SoftDist", "target": "n/a (no learning)",
     "gap": ("2.20", "3.13"), "time": ("MCTS budget as Fu", "idem"),
     "hw": "EPYC 7V13 (64 threads); A100", "ref": "LKH-3", "g1k": None, "t1k": None, "g10k": None, "t10k": None},
    {"name": "INViT 2024", "short": "INViT", "target": "no (TSP-100, zero-shot)",
     "gap": ("5.99", "≈ 6–7.5"), "time": ("4.8 min (set)", "—"), "hw": "RTX 4090", "ref": "LKH-3",
     "g1k": 5.99, "t1k": 4.8 * 60 / 128, "g10k": None, "t10k": None},
    {"name": "UDC 2024 (×50)", "short": "UDC", "target": "yes (≤ 2k)", "gap": ("1.78", "—"),
     "time": ("8 min (set)", "—"), "hw": "V100S", "ref": "LKH-3",
     "g1k": 1.78, "t1k": 8 * 60 / 128, "g10k": None, "t10k": None},
]


# ---------------------------------------------------------------- loading

def _npz(path):
    if not os.path.exists(path):
        return None
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def load_all(results):
    root = os.path.join(results, "phase3_benchmark")
    R = {"ref": _npz(os.path.join(root, "ref_lkh3.npz")), "e1": {}, "e1_meta": {}, "pol": {}, "win": {}, "e2": {}}
    for name in REF_SETS:
        R["e1"][name] = _npz(os.path.join(root, "e1", f"{name}.npz"))
        meta = os.path.join(root, "e1", f"{name}.json")
        if os.path.exists(meta):
            with open(meta) as f:
                R["e1_meta"][name] = json.load(f)
        for folder in [f"tagcn_k3_seed{s}" for s in SEEDS] + ["lkh"]:
            R["pol"][(name, folder)] = _npz(os.path.join(root, "e3", "policies", folder, f"{name}.npz"))
        for s in SEEDS:
            R["win"][(name, s)] = _npz(os.path.join(root, "e3", "window", f"tagcn_k3_seed{s}", f"{name}.npz"))
        folders = ["random", "simple", "oracle"] + [f"disagreement_seed{s}" for s in SEEDS]
        for folder in folders:
            R["e2"][(name, folder)] = _npz(os.path.join(root, "e2", folder, f"{name}.npz"))
    notes = os.path.join(results, "phase3_benchmark_notes.md")
    R["notes"] = open(notes).read().strip() if os.path.exists(notes) else "(notes missing: results/phase3_benchmark_notes.md)"
    return R


def ref_lengths(R, name):
    return None if R["ref"] is None or f"{name}/lengths" not in R["ref"] else R["ref"][f"{name}/lengths"]


# ---------------------------------------------------------------- per-method summaries

def our_point(R, name, method):
    """{"gap": [mean per seed], "s": [mean total time per seed], "lkh": [...], "per_inst": (instances,) mean over
    seeds} of one of our methods on one set, or None. method: "initial", "P1", a triage policy (P0, P2_top20,
    ...), or "beam10_w20" / "greedy_w20" (window_eval). Total time = greedy + 2-opt + refinement."""
    if method in ("beam10_w20", "greedy_w20"):
        runs = [R["win"].get((name, s)) for s in SEEDS]
    elif method in ("P1", "initial"):
        runs = [R["pol"].get((name, "lkh"))]
    else:
        runs = [R["pol"].get((name, f"tagcn_k3_seed{s}")) for s in SEEDS]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None
    if method == "initial":
        r = runs[0]
        return {"gap": [float(r["initial__gap"].mean())], "s": [float(r["initial__ms"].mean()) / 1000],
                "lkh": [0.0], "per_inst": r["initial__gap"]}
    gap = [float(r[f"{method}__gap"].mean()) for r in runs]
    secs = [float((r["initial__ms"] + r[f"{method}__ms"]).mean()) / 1000 for r in runs]
    lkh = [float(r[f"{method}__to_lkh"].sum() / r[f"{method}__windows"].sum()) if f"{method}__to_lkh" in r else 0.0
           for r in runs]
    return {"gap": gap, "s": secs, "lkh": lkh, "per_inst": np.mean([r[f"{method}__gap"] for r in runs], axis=0)}


def e1_rows(R, name):
    """[(variant, ours, budget, stats or None)] of E1 on one set; stats has gap, gap1 (after one trial), s,
    trials, over (share of instances with total time > OVER × budget); None if the config did not fit."""
    e, ref = R["e1"].get(name), ref_lengths(R, name)
    if e is None or ref is None:
        return []
    ref = ref[:e["length"].shape[0]]
    rows = []
    for c in range(len(e["config_variant"])):
        v, ours, b = str(e["config_variant"][c]), bool(e["config_ours"][c]), float(e["config_budget"][c])
        if not e["config_fits"][c]:
            rows.append((v, ours, b, None))
            continue
        t = e["total_seconds"][:, c]
        rows.append((v, ours, b, {"gap": float(np.mean(100 * (e["length"][:, c] / ref - 1))),
                                  "gap1": float(np.nanmean(100 * (e["first_trial_length"][:, c] / ref - 1))),
                                  "s": float(t.mean()), "trials": float(np.nanmean(e["trials"][:, c])),
                                  "over": float(np.mean(t > OVER * b)),
                                  "per_inst": 100 * (e["length"][:, c] / ref - 1)}))
    return rows


def e1_get(R, name, variant, ours, budget):
    for v, o, b, st in e1_rows(R, name):
        if v == variant and o == ours and np.isclose(b, budget):
            return st
    return None


def e2_policy(R, name, score, x):
    """Per-instance gap (mean over seeds), mean time (s, greedy + refinement) and per-seed mean gaps."""
    if score == "random":
        r = R["e2"].get((name, "random"))
        if r is None:
            return None
        keys = [f"random_s{s}_top{x}" for s in RANDOM_SEEDS if f"random_s{s}_top{x}__gap" in r]
        runs = [(r, k) for k in keys]
    elif score == "disagreement":
        runs = [(R["e2"].get((name, f"disagreement_seed{s}")), f"disagreement_top{x}") for s in SEEDS]
        runs = [(r, k) for r, k in runs if r is not None]
    else:
        r = R["e2"].get((name, score))
        runs = [(r, f"{score}_top{x}")] if r is not None else []
    if not runs:
        return None
    gaps = np.array([r[f"{k}__gap"] for r, k in runs])
    secs = np.array([((r["initial__ms"] + r[f"{k}__ms"]) / 1000).mean() for r, k in runs])
    lkh = np.mean([r[f"{k}__to_lkh"].sum() / r[f"{k}__windows"].sum() for r, k in runs])
    return {"per_inst": gaps.mean(axis=0), "seed_means": gaps.mean(axis=1), "s": float(secs.mean()),
            "lkh": float(lkh), "initial": float(runs[0][0]["initial__gap"].mean()), "runs": len(runs)}


def wilcoxon_e2(R, name):
    """{(x, other): paired test of disagreement vs other} on the per-instance gaps (each averaged over its
    seeds), Bonferroni over the comparisons of the set."""
    pairs = [(x, other) for x in E2_FRACS for other in ("random", "simple")]
    out = {}
    for x, other in pairs:
        a, b = e2_policy(R, name, "disagreement", x), e2_policy(R, name, other, x)
        if a is None or b is None or len(a["per_inst"]) < 5:
            continue
        t = summarize.paired_compare(a["per_inst"], b["per_inst"])
        thr = ALPHA / len(pairs)
        verdict = "n.s." if t["p"] >= thr else ("better" if t["median_diff"] < 0 or
                                                 (t["median_diff"] == 0 and t["mean_diff"] < 0) else "worse")
        out[(x, other)] = {**t, "threshold": thr, "verdict": verdict, "n": len(a["per_inst"])}
    return out


# ---------------------------------------------------------------- tables

def f2(v, d=2):
    return "missing" if v is None else f"{v:.{d}f}"


def mean_std(values, d=2):
    if values is None:
        return "missing"
    return f"{np.mean(values):.{d}f}" if len(values) == 1 else f"{np.mean(values):.{d}f} ± {np.std(values):.{d}f}"


OUR_ROWS = [("initial", "Ours: greedy edge + 2-opt (no network)", "n/a (no learning)"),
            ("beam10_w20", "Ours: model windows, beam B = 10, w = 20 (≤ 5 passes)", "no (TSP-20 only)"),
            ("P2_top20", "Ours: triage, LKH on top 20 % disagreement", "no (TSP-20 only)"),
            ("P2_top50", "Ours: triage, LKH on top 50 % disagreement", "no (TSP-20 only)"),
            ("P1", "Ours: LKH on all windows (no network)", "n/a (no learning)")]


def table_main(R):
    head = ("| Method | Gap 1k (%) | Gap 10k (%) | Time 1k (s / inst.) | Time 10k (s / inst.) | Hardware | "
            "Reference | Trained at target size |")
    lines = [head, "|---|---|---|---|---|---|---|---|"]
    for method, label, target in OUR_ROWS:
        p = {n: our_point(R, FU[n], method) for n in FU}
        cells = [mean_std(p[n]["gap"]) if p[n] else "missing" for n in FU] + \
                [f2(np.mean(p[n]["s"])) if p[n] else "missing" for n in FU]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {HARDWARE} | {REFERENCE} | {target} |")
    for idx, what in ((0, "budget of triage top 20 %"), (3, "budget of LKH on all windows")):
        st = {n: e1_get(R, FU[n], MAIN_LKH, False, BUDGETS[n][idx]) for n in FU}
        cells = [f2(st[n]["gap"], 3) if st[n] else "missing" for n in FU] + \
                [f2(st[n]["s"]) if st[n] else "missing" for n in FU]
        lines.append(f"| **LKH-3 QUADRANT 5 on the whole instance, {what}** "
                     f"({BUDGETS[1000][idx]} / {BUDGETS[10000][idx]} s) | " + " | ".join(cells) +
                     f" | {HARDWARE.split(' + ')[0]}, CPU only | {REFERENCE} | n/a (no learning) |")
    for p in PUBLISHED:
        lines.append(f"| {p['name']} *(published, not reproduced)* | {p['gap'][0]} | {p['gap'][1]} | {p['time'][0]} | "
                     f"{p['time'][1]} | {p['hw']} | {p['ref']} | {p['target']} |")
    lines.append("")
    lines.append(f"Our rows and LKH-3: the standard sets of Fu et al. ({FU[1000]}, 128 instances; {FU[10000]}, 16), "
                 "gap against the E0 reference, mean over the 3 TAGCN K=3 seeds (± std across seeds), wall-clock per "
                 "instance including the greedy + 2-opt tour. Published rows: as reported by each paper "
                 "(docs/literature_review_tsp.md); \"(set)\" = time for the whole test set.")
    return "\n".join(lines)


def table_e0(R):
    if R["ref"] is None:
        return "missing"
    lines = ["| Set | Instances | LKH-3 parameters | Mean length (E0) | Mean stored length | Improvement (mean / max) | "
             "Mean wall-clock (s) | Stored tours |", "|---|---|---|---|---|---|---|---|"]
    for name in REF_SETS:
        if f"{name}/lengths" not in R["ref"]:
            lines.append(f"| {name} | missing |" + " |" * 6)
            continue
        g = lambda k: R["ref"][f"{name}/{k}"]
        has = int(g("has_reference"))
        imp = g("improvement_pct")
        stored = {"tsp1000_test": "elkai (LKH-3.0.8), RUNS = 10", "tsp10000_test": "elkai, 1 run",
                  "tsp1000_test_concorde": "Fu et al. file (\"concorde\")",
                  "tsp10000_test_concorde": "none (the file's tour is the identity)"}[name]
        lines.append(f"| {name} | {len(g('lengths'))} | {g('params')} | {g('lengths').mean():.4f} | "
                     f"{g('old_lengths').mean():.4f} | " +
                     (f"{imp.mean():+.3f} % / {imp.max():+.3f} %" if has else "— (no reference)") +
                     f" | {g('wall_seconds').mean():.0f} | {stored} |")
    lines.append("")
    lines.append("Improvement = 100 × (stored − E0) / stored (> 0: the E0 tour is shorter). LKH-3.0.6, GitHub mirror "
                 "cerebis/LKH3 (the official site is not reachable from this server). TIME_LIMIT is per run in "
                 "3.0.6 (checked between trials), so a 10k instance takes up to about 5 × 696 s + preprocessing.")
    return "\n".join(lines)


def e1_cell(st):
    if st is None:
        return "does not fit"
    flag = " †" if st["over"] > OVER_SHARE else ""
    return f"{st['gap']:.3f} ({st['s']:.2f} s, {st['trials']:.0f} tr.){flag}"


def table_e1(R, name):
    rows = e1_rows(R, name)
    if not rows:
        return "missing"
    n = 1000 if "1000_" in name or name.endswith("1000_test") else 10000
    budgets = BUDGETS[n]
    variants = list(dict.fromkeys(v for v, _, _, _ in rows))
    lines = ["| Candidate set | Start | " + " | ".join(f"B = {b} s" for b in budgets) + " | Gap after 1 trial (%) |",
             "|---|---|" + "---|" * (len(budgets) + 1)]
    for v in variants:
        for ours in (False, True):
            st = {b: next((s for vv, o, bb, s in rows if vv == v and o == ours and np.isclose(bb, b)), None)
                  for b in budgets}
            if all(s is None for s in st.values()):
                lines.append(f"| {variant_label(v)} | {'ours' if ours else 'LKH'} | " +
                             " | ".join("does not fit" for _ in budgets) + " | — |")
                continue
            gap1 = next(s["gap1"] for s in st.values() if s is not None)
            label = f"**{variant_label(v)}**" if v == MAIN_LKH and not ours else variant_label(v)
            lines.append(f"| {label} | {'ours' if ours else 'LKH'} | " + " | ".join(e1_cell(st[b]) for b in budgets) +
                         f" | {gap1:.3f} |")
    lines.append("")
    meta = R["e1_meta"].get(name, {})
    lines.append(f"{meta.get('num', '?')} instances. Each cell: mean gap (%) to the E0 reference (mean real total time "
                 "per instance, mean number of LKH trials). † = more than 25 % of the instances took more than "
                 "1.1 × the budget (TIME_LIMIT is checked only between trials; 8 runs shared the 8 cores). "
                 "\"does not fit\": the measured fixed cost + one trial already exceeds the budget. Start \"ours\" = "
                 "INITIAL_TOUR_FILE with our greedy + 2-opt tour, whose time is part of the total; the heatmap "
                 "variants also pay the graph + network time.")
    return "\n".join(lines)


E3_ROWS = [("initial", "Greedy edge + 2-opt"), ("greedy_w20", "Model windows, greedy path, w = 20"),
           ("beam10_w20", "Model windows, beam B = 10, w = 20"), ("P0", "P0: model (beam) everywhere, ≤ 3 passes"),
           *[(f"P2_top{q}", f"P2: LKH on top {q} % disagreement") for q in P2_TOP],
           *[(f"P3_top{q}", f"P3: model if confident, LKH if top {q} % dis. and not confident") for q in (10, 20, 30)],
           ("P1", "P1: LKH on all windows")]


def table_e3(R):
    lines = ["| Method | " + " | ".join(f"Gap n = {n} (%) | Time n = {n} (s) | Windows to LKH n = {n}" for n in FU) + " |",
             "|---|" + "---|" * (3 * len(FU))]
    for method, label in E3_ROWS:
        cells = []
        for n in FU:
            p = our_point(R, FU[n], method)
            cells += ["missing"] * 3 if p is None else [mean_std(p["gap"]), f2(np.mean(p["s"])),
                                                         f"{100 * np.mean(p['lkh']):.0f} %"]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Mean ± std over the 3 TAGCN K=3 seeds (no seed for the greedy + 2-opt tour and P1). Thresholds of "
                 "P2 / P3: from the tsp200 record of the triage experiment (fixed numbers, as before).")
    return "\n".join(lines)


E2_LABELS = {"random": "Random (5 seeds)", "simple": "Simple rule: length / Σ NN distance",
             "disagreement": "Disagreement (3 seeds)", "oracle": "Oracle: real LKH gain"}


def table_e2(R):
    lines = ["| Set | x | Policy | Gap (%) | Gain (pp) | Share of oracle gain | Time (s) | Windows to LKH |",
             "|---|---|---|---|---|---|---|---|"]
    for name in E2_SETS:
        for x in E2_FRACS:
            pol = {s: e2_policy(R, name, s, x) for s in E2_LABELS}
            orc = pol["oracle"]
            for s, label in E2_LABELS.items():
                p = pol[s]
                if p is None:
                    lines.append(f"| {name} | {x} % | {label} | missing |  |  |  |  |")
                    continue
                gain = p["initial"] - p["per_inst"].mean()
                share = f"{100 * gain / (orc['initial'] - orc['per_inst'].mean()):.0f} %" if orc else "—"
                gap = (f"{p['per_inst'].mean():.3f}" if len(p["seed_means"]) == 1 else
                       f"{p['per_inst'].mean():.3f} ± {np.std(p['seed_means']):.3f}")
                time_s = f"{p['s']:.2f}" + (" *" if s == "oracle" else "")
                lines.append(f"| {name} | {x} % | {label} | {gap} | {gain:.3f} | {share} | {time_s} | "
                             f"{100 * p['lkh']:.0f} % |")
    lines.append("")
    lines.append("w = 20, offsets 0 and w/2, ≤ 3 passes, from the greedy + 2-opt tour; in every half-pass exactly the "
                 "top x % of the windows by the policy's score go to LKH (unlike P2, whose fixed thresholds send "
                 "14–46 %). Gap: mean over instances of the per-instance mean over seeds (± std of the seed means). "
                 "* the oracle runs LKH on every window to know its gain: its time is not that of a usable policy.")
    return "\n".join(lines)


def table_e2_tests(R):
    lines = ["| Set | x | Disagreement vs. | Better / equal / worse (% inst.) | Mean diff (pp) | p (Wilcoxon) | Verdict |",
             "|---|---|---|---|---|---|---|"]
    any_test = False
    for name in E2_SETS:
        for (x, other), t in sorted(wilcoxon_e2(R, name).items()):
            any_test = True
            v = f"**{t['verdict']}**" if t["verdict"] != "n.s." else "n.s."
            lines.append(f"| {name} ({t['n']}) | {x} % | {E2_LABELS[other]} | {t['better']:.0f} / {t['equal']:.0f} / "
                         f"{t['worse']:.0f} | {t['mean_diff']:+.3f} | {t['p']:.1e} | {v} |")
    if not any_test:
        return "missing"
    lines.append("")
    lines.append("Paired two-sided Wilcoxon on the per-instance gaps (disagreement averaged over its 3 seeds, random "
                 "over its 5); negative difference = disagreement shorter. Bonferroni within each set: 0.05 / 4 = 0.0125. "
                 "With 16 instances (10k) the test has little power.")
    return "\n".join(lines)


# ---------------------------------------------------------------- figure (E4)

def plot_pareto(R, figs_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, n in zip(axes, (1000, 10000)):
        name = FU[n]
        # LKH-3 at equal time: QUADRANT 5 (main reference line) and the hybrid network candidates.
        for variant, ours, color, style, label in ((MAIN_LKH, False, COLORS["lkh"], "-o",
                                                    "LKH-3 QUADRANT 5, whole instance, equal time"),
                                                   ("nnheat5", True, COLORS["hybrid"], "--^",
                                                    "LKH-3 with NN 5 ∪ heatmap 5 candidates + our tour")):
            pts = [e1_get(R, name, variant, ours, b) for b in BUDGETS[n]]
            pts = [p for p in pts if p is not None]
            if pts:
                ax.plot([p["s"] for p in pts], [p["gap"] for p in pts], style, color=color,
                        linewidth=2.5 if variant == MAIN_LKH else 1.5, markersize=7, label=label, zorder=3)
        # Ours.
        ini = our_point(R, name, "initial")
        if ini:
            ax.plot(ini["s"], ini["gap"], "s", color=INK, markersize=8, label="Greedy edge + 2-opt", zorder=4)
        p2 = [our_point(R, name, f"P2_top{q}") for q in P2_TOP]
        if all(p is not None for p in p2):
            ax.plot([np.mean(p["s"]) for p in p2], [np.mean(p["gap"]) for p in p2], "-o", color=COLORS["ours"],
                    linewidth=2, markersize=7, label="Ours: triage (LKH on top 10–50 % disagreement)", zorder=4)
            for q, p in zip(P2_TOP, p2):
                ax.annotate(f"{q} %", (np.mean(p["s"]), np.mean(p["gap"])), textcoords="offset points",
                            xytext=(5, 5), fontsize=7, color=INK)
        for method, marker, label in (("beam10_w20", "P", "Ours: model windows, beam B = 10"),
                                      ("P1", "X", "Ours: LKH on all windows")):
            p = our_point(R, name, method)
            if p:
                ax.plot(np.mean(p["s"]), np.mean(p["gap"]), marker, color=COLORS["ours"], markersize=9, label=label,
                        zorder=4, markeredgecolor="white")
        # Published (not reproduced).
        key_g, key_t = ("g1k", "t1k") if n == 1000 else ("g10k", "t10k")
        pub = [p for p in PUBLISHED if p[key_g] is not None and p[key_t] is not None]
        ax.plot([p[key_t] for p in pub], [p[key_g] for p in pub], "D", color=COLORS["published"], markersize=7,
                linestyle="none", label="Published: not reproduced, different hardware and reference", zorder=3,
                markeredgecolor="white")
        for p in pub:
            ax.annotate(p["short"], (p[key_t], p[key_g]), textcoords="offset points", xytext=(6, -3), fontsize=7,
                        color=MUTED)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
            axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        summarize.style_axes(ax)
        ax.set_title(f"n = {n} ({name}, {'128' if n == 1000 else '16'} instances)", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("Time per instance (s, log scale)", color=INK)
    axes[0].set_ylabel("Gap (%, log scale)", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Gap vs. time: our windows and triage, LKH-3 at equal time, published methods "
                 "(ours and LKH-3: gap to the E0 LKH-3 reference)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(figs_dir, exist_ok=True)
    summarize.save(fig, figs_dir, "phase3_benchmark_pareto")


# ---------------------------------------------------------------- report

CAVEATS = """1. **Reference (E0).** Our gaps and LKH-3's are measured against our E0 reference: LKH-3.0.6 (GitHub mirror
   cerebis/LKH3, commit ba346db, since the official site is unreachable from this server), RUNS = 10 at 1k and
   RUNS = 5 at 10k. At 10k, TIME_LIMIT = 696 s is **per run, not total** (in 3.0.6 it is checked between trials), so an
   instance took ≈ 1 h. This is not Concorde: the published gaps are against Concorde or long LKH-3 runs, which can be a
   few hundredths of a percent shorter at 10k. On our own sets the E0 reference equals the old one (tsp1000_test:
   identical; tsp10000_test: 0.0003 % shorter), so the earlier phase 2–3 reports stay valid (they only get a note).
2. **Fu et al. files.** The 1k file's "concorde" tours are not optimal: E0 is 0.47 % shorter on average (0.27–0.68 %),
   so a gap computed against the file is ≈ 0.47 pp too optimistic (or too pessimistic for a method that beats it).
   The 10k file has **no reference tour** (its "tour" is the identity): every 10k gap here is against E0, and the
   published 10k gaps are against each paper's own LKH-3 runs.
3. **Joshi et al. 2021 use k = 0.2 n** for test-time sparsification; our k = min(0.5 n, 100) neighbours (2-opt
   lists and the network's k-NN graph) is our own extension. A 0.2 n ablation was not run.
4. **Hardware and timing.** Ours: a shared Intel Xeon E5-2620 v3 (8 cores) under `nice -n 10` and a shared Tesla T4
   capped at 5.5 GiB; published times use 1080Ti to 8× V100, A100 or a 64-thread EPYC, often the network on GPU and
   the search on many cores. Several papers report the time of the whole test set (marked "(set)"; divided by 128 / 16
   in the figure) or network and search separately. Our times are wall-clock per instance, one instance at a time
   per process, with 4 processes (E3) or 8 LKH runs (E1) sharing the CPU.
5. **LKH-3 variants that exceed their budget.** TIME_LIMIT is checked only between trials, and the fixed cost was
   measured on one instance with a single process, so some variants overshoot (marked † in the E1 tables: more than
   25 % of the instances above 1.1 × the budget). On the standard sets: {over}. The main line (QUADRANT 5, LKH's own
   start) {main_over}. The hybrid NN 5 ∪ heatmap 5 variant was not probed: its fixed cost was estimated from heat5
   and nn8 / heat8.
6. **Training size.** Our network is trained on TSP-20 only and applied to windows of 20 cities; DIFUSCO, UDC,
   H-TSP and Select & Optimize train on or near the target size.
7. **Instance counts.** Our earlier reports used 20 (1k) and 3 (10k) own instances; E3 uses Fu's 128 and 16. The
   10k paired tests have 16 instances.
8. **Published numbers are not reproduced:** they are copied from the papers (docs/literature_review_tsp.md) and differ in
   hardware, reference, time accounting and sometimes in the test set."""


def over_budget(R):
    """(text listing the (variant, start) that are † at some budget on the standard sets, text for the main line)."""
    items, main = [], []
    for n, name in FU.items():
        bad = {}
        for v, ours, b, st in e1_rows(R, name):
            if st is not None and st["over"] > OVER_SHARE:
                bad.setdefault((v, ours), []).append(b)
                if v == MAIN_LKH and not ours:
                    main.append(f"{b} s at n = {n}")
        if bad:
            items.append(f"n = {n}: " + ", ".join(f"{variant_label(v)} ({'our' if o else 'LKH'} start; "
                                                  f"{'/'.join(f'{b:g}' for b in bs)} s)" for (v, o), bs in bad.items()))
    return ("; ".join(items) or "none"), ("exceeds it at " + ", ".join(main) if main else "stays within its budgets")


def build_report(results, figs_dir=None):
    R = load_all(results)
    over, main_over = over_budget(R)
    if figs_dir:
        plot_pareto(R, figs_dir)
    meta = next(iter(R["e1_meta"].values()), {})
    parts = [
        "# Phase 3 benchmark: windows and triage against LKH-3 and the literature",
        "",
        f"Generated by `scripts/phase3_benchmark_report.py` on {datetime.date.today().isoformat()}.",
        "",
        R["notes"],
        "",
        "## Comparison table",
        "",
        table_main(R),
        "",
        "![Gap vs. time](figs/phase3_benchmark_pareto.png)",
        "",
        "**E4.** Gap against time per instance on the standard sets (both axes log). Green: LKH-3 with QUADRANT 5 "
        "candidates on the whole instance, at our four time budgets (E1). Blue: our triage curve, model windows and "
        "LKH on all windows (E3). Orange diamonds: published numbers, not reproduced (other hardware and reference; "
        "SoftDist has no time of its own and is not drawn).",
        "",
        "## E0: LKH-3 reference",
        "",
        table_e0(R),
        "",
        "## E1: LKH-3 on the whole instance at equal time",
        "",
        "Budgets = the times of our triage policies (1k: P2 top 20 % 0.72 s, top 50 % 1.25 s, P0 1.54 s, P1 1.94 s; "
        "10k: 8.4, 14.2, 17.5, 21.5 s). One LKH run per instance (RUNS = 1) with TIME_LIMIT = budget − fixed cost of "
        "the variant − our greedy / network time; the real total time is recorded.",
        "",
        f"### {FU[1000]} (Fu et al., 1k)",
        "",
        table_e1(R, FU[1000]),
        "",
        f"### {FU[10000]} (Fu et al., 10k)",
        "",
        table_e1(R, FU[10000]),
        "",
        "### Our own sets (tsp1000_test, 20 instances; tsp10000_test, 3)",
        "",
        table_e1(R, OWN[1000]),
        "",
        table_e1(R, OWN[10000]),
        "",
        "## E3: our methods on the standard sets",
        "",
        table_e3(R),
        "",
        "## E2: which windows should go to LKH?",
        "",
        table_e2(R),
        "",
        "### Paired tests",
        "",
        table_e2_tests(R),
        "",
        "## Comparability caveats",
        "",
        CAVEATS.format(over=over, main_over=main_over),
        "",
        "## Reproduction",
        "",
        "From `tsp/`: `bash scripts/build_lkh3.sh` (LKH-3.0.6, kept out of git for its licence), then "
        "`scripts/run_phase3_bench_e0.sh`, `run_phase3_bench_e1.sh` (after `scripts/lkh3_e1_prep.py`), "
        "`run_phase3_bench_e3.sh`, `run_phase3_bench_e2.sh` and `python scripts/phase3_benchmark_report.py`. "
        f"E1 network candidates: {meta.get('heat_run', 'tagcn_k3_seed0')} (best.pt).",
    ]
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Phase 3 benchmark report.")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    text = build_report(args.results, os.path.join(args.results, "figs"))
    out = os.path.join(args.results, "phase3_benchmark_report.md")
    with open(out, "w") as f:
        f.write(text)
    print(f"saved to: {out}")


if __name__ == "__main__":
    main()
