"""Report of the triage with larger windows (phase 3): does window selection pay where LKH is expensive per window?

Reads results/phase3_triage_wsize/: log/w<w>/ (LKH per window), models/w<w>/ (signal quality per w), par/w<w>/
(policies run in parallel: oracle, tsp1000), seq/w<w>/ (policies timed one process at a time on the first 8 Fu-10k
instances, with seq/load.tsv); w = 20 also reuses the fast-triage sequential run (results/phase3_fast_triage/seq)
and the learned-triage oracle (results/phase3_learned_triage/e2/oracle) on the same 8 instances. Opening notes:
results/phase3_triage_wsize_notes.md. Writes results/phase3_triage_wsize_report.md, the figure
results/figs/phase3_triage_wsize.png and copies of both in docs/.

Example:
    python scripts/phase3_triage_wsize_report.py
"""

import datetime
import importlib.util
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("summarize", os.path.join(_here, "summarize.py"))
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)

ROOT = "results/phase3_triage_wsize"
FAST = "results/phase3_fast_triage/seq"
LT_ORACLE = "results/phase3_learned_triage/e2/oracle"
OUT = "results/phase3_triage_wsize_report.md"
DOCS = "docs/phase3_triage_wsize_report.md"
FIG = "results/figs/phase3_triage_wsize.png"
NOTES = "results/phase3_triage_wsize_notes.md"
FU10K, T1K = "tsp10000_test_concorde", "tsp1000_test"
NUM_10K = 8
SEEDS = (0, 1, 2)
RANDOM_SEEDS = (0, 1, 2)
WIDTHS = (20, 50, 100)
ALPHA = 0.05
SET_ROLES = {"tsp200_test": "train / validation", "tsp1000_test_concorde": "train / validation (Fu-1k)",
             "tsp1000_test": "test", "tsp10000_test_concorde": "test (Fu-10k)"}
LABEL = {"random": "Random", "disagreement": "Disagreement", "learned": "Learned", "all": "LKH on every window",
         "oracle": "Oracle"}


def npz(path):
    return dict(np.load(path)) if os.path.exists(path) else None


def ms(v, d=3):
    v = [x for x in v if x == x]
    return "—" if not v else (f"{np.mean(v):.{d}f}" if len(v) == 1 else f"{np.mean(v):.{d}f} ± {np.std(v):.{d}f}")


# ---------------------------------------------------------------------------------------------------- policy data

def _runs(root, score, x):
    """[(npz, key)] of one policy in a folder tree laid out by triage_ablation.py."""
    if score == "random":
        out = []
        for folder in ("random", f"random_top{x}"):
            r = npz(os.path.join(root, folder, f"{NAME}.npz"))
            if r is not None:
                out += [(r, f"random_s{s}_top{x}") for s in range(5) if f"random_s{s}_top{x}__gap" in r]
        return out
    if score == "all":
        r = npz(os.path.join(root, "random_all", f"{NAME}.npz"))
        return [] if r is None else [(r, "random_s0_top100")]
    if score == "oracle":
        out = []
        for folder in ("oracle", f"oracle_top{x}"):
            r = npz(os.path.join(root, folder, f"{NAME}.npz"))
            if r is not None and f"oracle_top{x}__gap" in r:
                out.append((r, f"oracle_top{x}"))
        return out[:1]
    pre = {"disagreement": "fast_disagreement", "learned": "fast_learned"}[score]
    runs = [(npz(os.path.join(root, f"{pre}_seed{s}", f"{NAME}.npz")), f"{pre}_top{x}") for s in SEEDS]
    return [(r, k) for r, k in runs if r is not None and f"{k}__gap" in r]


NAME = FU10K


def policy(name, w, score, x, mode):
    """Per-instance gap (mean over runs), time per instance (s), ... ; mode 'seq' (Fu-10k, 8 instances, timed alone)
    or 'par' (gaps only; tsp1000 and the oracle)."""
    global NAME
    NAME = name
    roots = []
    if mode == "seq":
        roots.append(os.path.join(ROOT, "seq", f"w{w}"))
        if w == 20:
            roots.append(FAST)
    else:
        roots.append(os.path.join(ROOT, "par", f"w{w}"))
        if w == 20:
            roots.append(LT_ORACLE.rsplit("/", 1)[0] if score == "oracle" else "results/phase3_fast_triage/e2")
    runs = []
    for root in roots:
        runs += _runs(root, score, x)
        if runs:
            break
    if not runs:
        return None
    num = NUM_10K if name == FU10K else None
    gaps = np.array([r[f"{k}__gap"][:num] for r, k in runs])
    secs = np.array([((r["initial__ms"] + r[f"{k}__ms"]) / 1000)[:num].mean() for r, k in runs])
    return {"per_inst": gaps.mean(axis=0), "run_means": gaps.mean(axis=1), "s": float(secs.mean()),
            "initial": float(runs[0][0]["initial__gap"][:num].mean()), "runs": len(runs),
            "lkh_s": float(np.mean([r[f"{k}__ms_lkh"][:num].mean() / 1000 for r, k in runs])),
            "score_s": float(np.mean([r[f"{k}__ms_score"][:num].mean() / 1000 for r, k in runs]))}


# ---------------------------------------------------------------------------------------------------- tables

def table_windows():
    lines = ["| w | Set | Windows (first pass) | LKH improves (%) | Mean window gain (%) | LKH ms per window | "
             "… when it improves / when not |", "|---|---|---|---|---|---|---|"]
    for w in (50, 100):
        for name in SET_ROLES:
            p = os.path.join(ROOT, "log", f"w{w}", f"{name}.json")
            if not os.path.exists(p):
                lines.append(f"| {w} | {name} | missing | | | | |")
                continue
            d = json.load(open(p))
            fmt = lambda v: "—" if v is None else f"{v:.1f}"
            lines.append(f"| {w} | {name} | {d['windows']} | {100 * d['improves_frac']:.1f} | "
                         f"{d['mean_gain_rel_pct']:.2f} | {d['ms_lkh_per_window']:.1f} | "
                         f"{fmt(d['ms_lkh_per_window_improves'])} / {fmt(d['ms_lkh_per_window_not'])} |")
    lines += ["", "w = 20 (from the learned-triage logs): 21–30 % of the windows improve, ~7 ms of LKH per window. "
              "Times measured with other projects loading the machine (see the load table)."]
    return lines


def metrics(w):
    if w == 20:
        root = "results/phase3_learned_triage/models"
    else:
        root = os.path.join(ROOT, "models", f"w{w}")
    out = {}
    for s in SEEDS:
        p = os.path.join(root, f"tagcn_k3_seed{s}", "metrics.json")
        if os.path.exists(p):
            out[s] = json.load(open(p))
    return out


def table_ap():
    lines = ["| w | Set | Windows | Random (= positive rate) | Disagreement | Learned (chosen) | Oracle | "
             "Learned − disagreement |", "|---|---|---|---|---|---|---|---|"]
    for w in WIDTHS:
        M = metrics(w)
        if not M:
            lines.append(f"| {w} | missing | | | | | | |")
            continue
        for name in ("tsp200_test", "tsp1000_test_concorde", T1K, FU10K):
            test = name in (T1K, FU10K)
            v = lambda f: [f(m) for m in M.values()]
            try:
                rnd = v(lambda m: m["raw"][name]["random"])
                dis = v(lambda m: m["raw"][name]["disagreement"])
                lrn = v(lambda m: m["models"][m["chosen"]]["test_ap" if test else "val_ap"][name])
            except KeyError:
                continue
            windows = M[next(iter(M))]["raw"][name]["windows"]
            role = "test" if test else "validation half"
            lines.append(f"| {w} | {name} ({role}) | {windows} | {ms(rnd)} | {ms(dis)} | {ms(lrn)} | 1 | "
                         f"{np.mean(lrn) - np.mean(dis):+.3f} |")
    lines += ["", "AP of 'LKH improves the window'; mean ± std over the 3 TAGCN seeds; the learned model is retrained "
              "for each w (same protocol as the learned triage: train on half of tsp200 + Fu-1k, equal weight per "
              "set). w = 20 rows come from the learned-triage run."]
    return lines


def table_policies(name, mode):
    fracs = (10, 20, 50)
    lines = ["| w | x | Policy | Gap (%) | Share of oracle gain | Time (s) | of which LKH (s) | of which scoring (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for w in (20, 50):
        for x in fracs:
            orc = policy(name, w, "oracle", x, "par")
            for score in ("random", "disagreement", "learned"):
                p = policy(name, w, score, x, mode)
                if p is None:
                    lines.append(f"| {w} | {x} % | {LABEL[score]} | missing | | | | |")
                    continue
                gain = p["initial"] - p["per_inst"].mean()
                share = (f"{100 * gain / (orc['initial'] - orc['per_inst'].mean()):.0f} %" if orc else "—")
                gap = f"{p['per_inst'].mean():.3f}" + (f" ± {np.std(p['run_means']):.3f}" if p["runs"] > 1 else "")
                lines.append(f"| {w} | {x} % | {LABEL[score]} | {gap} | {share} | {p['s']:.2f} | {p['lkh_s']:.2f} | "
                             f"{p['score_s']:.2f} |")
            if orc:
                lines.append(f"| {w} | {x} % | Oracle (not a usable policy) | {orc['per_inst'].mean():.3f} | 100 % | "
                             "— | | |")
        p = policy(name, w, "all", 100, mode)
        if p is not None:
            lines.append(f"| {w} | 100 % | LKH on every window | {p['per_inst'].mean():.3f} | | {p['s']:.2f} | "
                         f"{p['lkh_s']:.2f} | 0.00 |")
    return lines


def table_tests(name, mode):
    lines = ["| w | x | Learned vs. | Better / equal / worse (% inst.) | Mean diff (pp) | p (Wilcoxon) | Verdict |",
             "|---|---|---|---|---|---|---|"]
    pairs = [(w, x, o) for w in (20, 50) for x in (10, 20, 50) for o in ("random", "disagreement")]
    for w, x, other in pairs:
        a, b = policy(name, w, "learned", x, mode), policy(name, w, other, x, mode)
        if a is None or b is None:
            continue
        t = summarize.paired_compare(a["per_inst"], b["per_inst"])
        thr = ALPHA / 6                                       # 3 fractions × 2 comparisons per w and set
        verdict = "n.s." if t["p"] >= thr else ("**better**" if t["median_diff"] < 0 or
                                                (t["median_diff"] == 0 and t["mean_diff"] < 0) else "**worse**")
        lines.append(f"| {w} | {x} % | {LABEL[other]} | {t['better']:.0f} / {t['equal']:.0f} / {t['worse']:.0f} | "
                     f"{t['mean_diff']:+.3f} | {t['p']:.1e} | {verdict} |")
    lines += ["", f"Paired two-sided Wilcoxon on the per-instance gaps (each policy averaged over its seeds); Bonferroni "
              f"{ALPHA} / 6 within each w and set. With 8 instances the smallest possible p is 0.0078."]
    return lines


def table_load():
    p = os.path.join(ROOT, "seq", "load.tsv")
    if not os.path.exists(p):
        return ["missing"]
    rows = [line.rstrip("\n").split("\t") for line in list(open(p))[1:]]
    runs = {}
    for when, run, l1, l5, l15, g in rows:
        runs.setdefault(run, {})[when] = (f"{l1} / {l5} / {l15}", g)
    lines = ["| Run | Load average before (1 / 5 / 15 min) | after | GPU used by other processes before / after (MiB) |",
             "|---|---|---|---|"]
    for run, d in runs.items():
        b, a = d.get("before", ("—", "—")), d.get("after", ("—", "—"))
        lines.append(f"| {run} | {b[0]} | {a[0]} | {b[1]} / {a[1]} |")
    return lines


def figure():
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = {20: "#2a6fdb", 50: "#d9822b"}
    markers = {"random": "o", "disagreement": "s", "learned": "D"}
    for w in (20, 50):
        for score in ("random", "disagreement", "learned"):
            pts = [policy(FU10K, w, score, x, "seq") for x in (10, 20, 50)]
            pts = [(p["s"], p["per_inst"].mean()) for p in pts if p is not None]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, marker=markers[score], color=colors[w], lw=1.5, ms=6,
                        ls="-" if score == "learned" else (":" if score == "random" else "--"),
                        label=f"w = {w}, {LABEL[score].lower()} (10 / 20 / 50 %)")
        p = policy(FU10K, w, "all", 100, "seq")
        if p is not None:
            ax.plot([p["s"]], [p["per_inst"].mean()], marker="*", ms=13, color=colors[w], ls="none",
                    label=f"w = {w}, LKH on every window")
    ax.set_xlabel("Time per instance (s), one process at a time")
    ax.set_ylabel("Gap to the LKH-3 reference (%)")
    ax.set_title(f"Fu-10k, first {NUM_10K} instances: gap vs time by window size")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def main():
    notes = open(NOTES).read().strip() if os.path.exists(NOTES) else f"(notes missing: {NOTES})"
    figure()
    lines = ["# Phase 3 — triage with larger windows (w = 20, 50, 100)", "",
             f"Generated {datetime.date.today().isoformat()} by scripts/phase3_triage_wsize_report.py. Greedy edge + "
             "2-opt tour, consecutive windows with fixed ends, offsets 0 and w/2, ≤ 3 passes, LKH on a window via a "
             "dummy city; TAGCN K=3 (3 seeds), batched scoring (fast triage, version A).", "", notes, "",
             "## Windows: LKH cost and how many improve", "", *table_windows(), "",
             "## Signal quality: average precision by w", "", *table_ap(), "",
             f"## Policies on Fu-10k, first {NUM_10K} instances, one process at a time", "",
             *table_policies(FU10K, "seq"), "",
             "Time: greedy + refinement per instance, each policy run alone (no other process of ours). w = 20 top 20 / "
             "50 % come from the fast-triage sequential run (same 8 instances, random with 5 seeds there); the oracle "
             "ran in parallel (its time is not that of a usable policy). w = 100 is not in the policy runs (see the "
             "summary).", "",
             "![gap vs time](figs/phase3_triage_wsize.png)", "",
             "### Paired tests (Fu-10k)", "", *table_tests(FU10K, "seq"), "",
             "### Machine load during the sequential runs", "", *table_load(), "",
             "## Policies on tsp1000 (20 instances, run in parallel: gaps only)", "",
             *table_policies(T1K, "par"), "",
             "### Paired tests (tsp1000)", "", *table_tests(T1K, "par"), ""]
    open(OUT, "w").write("\n".join(lines) + "\n")
    os.makedirs("docs", exist_ok=True)
    open(DOCS, "w").write("\n".join(lines).replace("](figs/", "](") + "\n")   # the figure sits next to it
    shutil.copy(FIG, os.path.join("docs", os.path.basename(FIG)))
    print(f"wrote {OUT}, {DOCS} and the figure")


if __name__ == "__main__":
    main()
