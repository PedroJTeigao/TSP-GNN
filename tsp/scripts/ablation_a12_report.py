"""Tables and figure of ablations A1 (window memo) and A2 (repeated timings), from results/ablations/ (run_ablation_a12.sh).

Time per instance = greedy edge + 2-opt tour + refinement, as in the phase 3 reports. Memo modes: none (phase 3), (a) cache,
(b) exclude. "(a) derived" = the run without memo minus the LKH calls its shadow memo marked as repeats (ms_lkh_repeat).

Equal time: a policy is compared with random at the same time on each instance, interpolating along that instance's random
curve (top 10 / 20 / 50 % and LKH on every window, same memo mode), as in the phase 3 triage report; Wilcoxon on the per-instance
differences. Fu-1k: the learned selector was trained on half of Fu-1k, so the main rows use the other 64 instances.

Writes results/ablations/a12.md, a12.json and tsp/docs/figs/ablation_a2_pareto.png.

    python scripts/ablation_a12_report.py
"""

import glob
import json
import os

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tsp_gnn.learned_triage import split_instances  # noqa: E402

ROOT = "results/ablations"
NAMES = {"fu1k": "tsp1000_test_concorde", "fu10k": "tsp10000_test_concorde"}
POLICIES = ("all", "random", "dis", "learned")
XS = (10, 20, 50)
LABEL = {"all": "LKH on every window", "random": "random", "dis": "disagreement", "learned": "learned"}
MEMO = {"none": "none", "cache": "(a) cache", "exclude": "(b) exclude", "derived": "(a) derived"}


def folder(policy, seed, memo, rep):
    base = {"all": "random_all", "random": f"random_s{seed}", "dis": f"fast_disagreement_seed{seed}",
            "learned": f"fast_learned_seed{seed}"}[policy]
    return base + (f"_memo{memo}" if memo != "none" else "") + (f"_rep{rep}" if rep else "")


def prefix(policy, seed):
    return {"all": "random_s0", "random": f"random_s{seed}", "dis": "fast_disagreement", "learned": "fast_learned"}[policy]


def load(w, s, policy, seed, memo, rep=0):
    """{x: {gap, time, calls, hits, net} per-instance arrays} or None if not run. memo 'derived' reads the run without memo."""
    real = "none" if memo == "derived" else ("cache" if (memo == "exclude" and policy == "all") else memo)
    path = f"{ROOT}/a1/w{w}/{folder(policy, seed, real, rep)}/{NAMES[s]}.npz"
    if not os.path.exists(path):
        return None
    d = np.load(path)
    meta = json.load(open(path[:-4] + ".json"))["meta"]
    out = {}
    for x in ((100,) if policy == "all" else XS):
        p = f"{prefix(policy, seed)}_top{x}"
        ms = d[f"{p}__ms"] - (d[f"{p}__ms_lkh_repeat"] if memo == "derived" else 0.0)
        calls = d[f"{p}__lkh_calls"] - (d[f"{p}__memo_hits"] if memo == "derived" else 0.0)
        out[x] = {"gap": d[f"{p}__gap"], "time": (d["initial__ms"] + ms) / 1000, "calls": calls,
                  "hits": d[f"{p}__memo_hits"], "net": d[f"{p}__ms_net"] / 1000, "load": meta["load_before"][0]}
    return out


def mask(s, rows, held_out):
    if s == "fu1k" and held_out:
        return ~split_instances(128, seed=0)
    return np.ones(rows, dtype=bool)


def curve(w, s, seed, memo, rep=0):
    """Random's points per instance: (times (P, I), gaps (P, I)) over 10 / 20 / 50 % and LKH on every window."""
    r, a = load(w, s, "random", seed, memo, rep), load(w, s, "all", 0, memo, rep)
    if r is None or a is None:
        return None
    pts = [r[x] for x in XS] + [a[100]]
    return np.array([p["time"] for p in pts]), np.array([p["gap"] for p in pts])


def equal_time(w, s, policy, seed, x, memo, held_out=True, rep=0):
    """Mean of (policy gap − random gap at the policy's time) per instance, and the Wilcoxon p."""
    pol, cur = load(w, s, policy, seed, memo, rep), curve(w, s, seed if policy == "random" else 0, memo, rep)
    if pol is None or cur is None:
        return None
    t, g = cur
    m = mask(s, t.shape[1], held_out)
    diffs = []
    for i in np.flatnonzero(m):
        order = np.argsort(t[:, i])
        diffs.append(pol[x]["gap"][i] - np.interp(pol[x]["time"][i], t[order, i], g[order, i]))
    diffs = np.array(diffs)
    return {"diff": float(diffs.mean()), "p": float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0,
            "out_of_range": int(np.sum([pol[x]["time"][i] > t[:, i].max() or pol[x]["time"][i] < t[:, i].min()
                                        for i in np.flatnonzero(m)]))}


def table_a1(w, s, seed=0, held_out=True):
    lines = [f"| policy | x % | memo | gap % | time s / inst. | LKH calls / inst. | calls saved by the memo | gap − random at equal time (pp) | Wilcoxon p |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for policy in POLICIES:
        for memo in ("none", "derived", "cache", "exclude"):
            if policy == "all" and memo == "exclude":
                continue
            d = load(w, s, policy, seed, memo)
            if d is None:
                continue
            for x, v in d.items():
                m = mask(s, len(v["gap"]), held_out)
                none = load(w, s, policy, seed, "none")
                saved = (1 - v["calls"][m].mean() / none[x]["calls"][m].mean()) * 100 if none else float("nan")
                eq = equal_time(w, s, policy, seed, x, memo, held_out) if policy in ("dis", "learned") else None
                row = {"policy": policy, "x": x, "memo": memo, "gap": float(v["gap"][m].mean()), "time": float(v["time"][m].mean()),
                       "calls": float(v["calls"][m].mean()), "saved": float(saved), "eq": eq}
                rows.append(row)
                lines.append(f"| {LABEL[policy]} | {x} | {MEMO[memo]} | {row['gap']:.3f} | {row['time']:.2f} | {row['calls']:.0f} | "
                             f"{saved:.0f} % | " + (f"{eq['diff']:+.3f} | {eq['p']:.3g} |" if eq else "— | — |"))
    return "\n".join(lines), rows


def table_seeds(w, s, held_out=True):
    lines = ["| policy | seed | x % | memo | gap % | time s / inst. | gap − random at equal time (pp) | Wilcoxon p |", "|---|---:|---:|---|---:|---:|---:|---:|"]
    for policy in ("random", "dis", "learned"):
        for seed in (0, 1, 2):
            for memo in ("none", "derived", "exclude"):
                d = load(w, s, policy, seed, memo)
                if d is None:
                    continue
                for x, v in d.items():
                    m = mask(s, len(v["gap"]), held_out)
                    eq = equal_time(w, s, policy, seed, x, memo, held_out) if policy != "random" else None
                    lines.append(f"| {LABEL[policy]} | {seed} | {x} | {MEMO[memo]} | {v['gap'][m].mean():.3f} | {v['time'][m].mean():.2f} | "
                                 + (f"{eq['diff']:+.3f} | {eq['p']:.3g} |" if eq else "— | — |"))
    return "\n".join(lines)


def reps(w, s, policy, memo, x):
    """Mean time per instance in each repetition (rep 0 is block A)."""
    out = []
    for r in range(5):
        d = load(w, s, policy, 0, memo, r)
        if d is not None:
            out.append((float(d[x]["time"].mean()), float(d[x]["gap"].mean()), d[x]["load"]))
    return out


def table_a2():
    pts = [("all", "none", 100), ("all", "cache", 100)] + [(p, "none", x) for p in ("random", "learned") for x in XS]
    lines = ["| point (Fu-10k, w = 20, seed 0) | reps | median time s / inst. | IQR | min–max | gap % (all reps) | load before each rep |",
             "|---|---:|---:|---:|---:|---:|---|"]
    stats = {}
    for p, memo, x in pts:
        r = reps(20, "fu10k", p, memo, x)
        if not r:
            continue
        t = np.array([a for a, _, _ in r])
        q1, med, q3 = np.percentile(t, [25, 50, 75])
        gaps = sorted({round(g, 6) for _, g, _ in r})
        stats[(p, memo, x)] = (q1, med, q3, t)
        lines.append(f"| {LABEL[p]} {x} % {MEMO[memo]} | {len(r)} | {med:.2f} | {q1:.2f}–{q3:.2f} | {t.min():.2f}–{t.max():.2f} | "
                     f"{', '.join(f'{g:.3f}' for g in gaps)} | {', '.join(f'{l:.1f}' for _, _, l in r)} |")
    # orderings by time that the IQRs separate or not
    order = ["| ordering by time | separated by the IQR? |", "|---|---|"]
    keys = list(stats)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            qa, qb = stats[a], stats[b]
            fast, slow = (a, b) if qa[1] < qb[1] else (b, a)
            sep = stats[fast][2] < stats[slow][0]
            if abs(qa[1] - qb[1]) / max(qa[1], qb[1]) < 0.35:            # only pairs close enough to be in doubt
                order.append(f"| {LABEL[fast[0]]} {fast[2]} % {MEMO[fast[1]]} < {LABEL[slow[0]]} {slow[2]} % {MEMO[slow[1]]} | "
                             f"{'yes' if sep else 'no'} |")
    # the equal-time verdict in each repetition
    eq = ["| learned at x % against random, equal time, per repetition (pp) | rep 0 | rep 1 | rep 2 | rep 3 | rep 4 | all of one sign? |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for x in XS:
        vals = [equal_time(20, "fu10k", "learned", 0, x, "none", rep=r) for r in range(5)]
        vals = [v["diff"] if v else np.nan for v in vals]
        ok = [v for v in vals if not np.isnan(v)]
        eq.append(f"| top {x} % | " + " | ".join("—" if np.isnan(v) else f"{v:+.3f}" for v in vals) +
                  f" | {'yes' if ok and (all(v < 0 for v in ok) or all(v > 0 for v in ok)) else 'no'} |")
    return "\n".join(lines), "\n".join(order), "\n".join(eq), stats


def anchors():
    out = {}
    for s, name in NAMES.items():
        path = f"{ROOT}/a2/e1_rep0/{name}.npz"
        if not os.path.exists(path):
            continue
        d = np.load(path)
        ref = np.load(f"results/phase3_benchmark/data/{name}.npz")["lengths"][: d["length"].shape[0]]
        rows = []
        for c, b in enumerate(d["config_budget"]):
            gap = 100 * (d["length"][:, c] / ref - 1)
            rows.append((float(b), float(np.nanmean(d["total_seconds"][:, c])), float(np.nanmean(gap))))
        out[s] = rows
    return out


def table_anchors(an):
    old = {"fu1k": {0.72: (0.75, 0.079), 1.25: (1.28, 0.054), 1.54: (1.57, 0.042), 1.94: (1.98, 0.035)},
           "fu10k": {8.4: (8.35, 0.150), 14.2: (14.09, 0.105), 17.5: (17.46, 0.093), 21.5: (21.43, 0.081)}}
    lines = ["| set | budget s | 8 processes (phase 3): time s / gap % | 1 process, now: time s / gap % |", "|---|---:|---:|---:|"]
    for s, rows in an.items():
        for b, t, g in rows:
            ot, og = old[s].get(round(b, 2), (np.nan, np.nan))
            lines.append(f"| {s} | {b} | {ot:.2f} / {og:.3f} | {t:.2f} / {g:.3f} |")
    return "\n".join(lines)


def figure(stats, an):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"none": "#2b6cb0", "cache": "#c05621", "exclude": "#2f855a"}
    for ax, s in zip(axes, ("fu1k", "fu10k")):
        for memo in ("none", "cache", "exclude"):
            for policy, mk in (("random", "o"), ("dis", "s"), ("learned", "^"), ("all", "*")):
                d = load(20, s, policy, 0, memo)
                if d is None:
                    continue
                m = mask(s, len(next(iter(d.values()))["gap"]), True)
                xs = sorted(d)
                t = [d[x]["time"][m].mean() for x in xs]
                g = [d[x]["gap"][m].mean() for x in xs]
                ax.plot(t, g, marker=mk, color=colors[memo], ls="-" if policy != "all" else "none", ms=8 if policy == "all" else 5,
                        label=f"{LABEL[policy]}, {MEMO[memo]}" if s == "fu1k" else None, alpha=0.9)
                if s == "fu10k":
                    for x, tt, gg in zip(xs, t, g):
                        key = (policy, memo, x)
                        if key in stats:
                            q1, med, q3, _ = stats[key]
                            ax.errorbar([med], [gg], xerr=[[med - q1], [q3 - med]], color="black", capsize=3, lw=1)
        if s in an:
            ax.plot([t for _, t, _ in an[s]], [g for _, _, g in an[s]], "D-", color="#6b46c1",
                    label="LKH-3 QUADRANT 5, whole instance" if s == "fu1k" else None)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("time per instance (s), 1 process, CPU"); ax.set_ylabel("gap to the E0 reference (%)")
        ax.set_title({"fu1k": "Fu-1k (64 held-out instances)", "fu10k": "Fu-10k (16); black bars: IQR of 5 runs"}[s])
        ax.grid(alpha=0.3, which="both")
    fig.legend(loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    os.makedirs("docs/figs", exist_ok=True)
    fig.savefig("docs/figs/ablation_a2_pareto.png", dpi=150)


def load_table():
    path = f"{ROOT}/a12_load.tsv"
    rows = [l.rstrip("\n").split("\t") for l in open(path)][1:] if os.path.exists(path) else []
    loaded = [r for r in rows if r[4] == "1"]
    l1 = [float(r[2]) for r in rows]
    return (f"{len(rows)} measurements; load before: median {np.median(l1):.2f}, max {max(l1):.2f}; "
            f"{len(loaded)} measured above 4 after waiting 30 min: {', '.join(r[1] for r in loaded) or 'none'}.") if rows else "no load table"


def main():
    parts = ["# Ablations A1 / A2: tables (generated by scripts/ablation_a12_report.py)", "", "## Load", "", load_table(), ""]
    for s in ("fu10k", "fu1k"):
        t, _ = table_a1(20, s)
        parts += [f"## A1, {s}, w = 20, seed 0" + (" (64 instances not used to train the learned selector)" if s == "fu1k" else ""), "", t, ""]
    t, _ = table_a1(20, "fu1k", held_out=False)
    parts += ["## A1, fu1k, w = 20, seed 0, all 128 instances", "", t, ""]
    t, _ = table_a1(50, "fu10k")
    parts += ["## A1, fu10k, w = 50, seed 0", "", t, ""]
    for s in ("fu10k", "fu1k"):
        parts += [f"## A1, {s}, w = 20, seeds 0-2", "", table_seeds(20, s), ""]
    rep, order, eq, stats = table_a2()
    an = anchors()
    parts += ["## A2, repetitions", "", rep, "", order, "", eq, "", "## A2, LKH-3 anchors", "", table_anchors(an), ""]
    figure(stats, an)
    open(f"{ROOT}/a12.md", "w").write("\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
