"""Report of phase A: docs/phaseA_report.md and its figures, from results/warehouse/phaseA/.

The hand-written parts are read from docs/phaseA_notes.md (one "## " section per part of the report); every
number and table is computed here.

Example:
    python scripts/phaseA_report.py
"""

import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student
from scipy.stats import ttest_rel

from warehouse import conflicts as C
from warehouse import env as W
from warehouse import viz

OUT = "results/warehouse/phaseA"
FIG = "docs/figures"
SIZES = tuple(W.SIZES)
METHODS = ("nn", "2opt", "lkh")
PLANNERS = ("pibt", "prioritized", "cbs")
RATES = ("loose", "medium", "x0.9", "x1.0", "x1.1", "saturated", "x1.3", "x1.4", "x1.5", "x1.7", "x2.0")
STEPS = 2000


def ci(v):
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(student.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def fmt(v, d=0):
    m, h = ci(v)
    return f"{m:.{d}f} ± {h:.{d}f}"


def load(sub):
    return [json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/{sub}/*.json"))]


def table_layouts():
    lines = ["| size | rows x cols | corridor | free cells | shelves | stations | robots | K (E1) | products |", "|---|---|---|---|---|---|---|---|---|"]
    for s in SIZES:
        lay = W.SIZES[s]
        wh = W.Warehouse(lay)
        lines.append(f"| {s} | {lay.height} x {lay.width} | {lay.corridor} | {len(wh.free)} | {len(wh.shelves)} | {lay.stations} | "
                     f"{lay.robots} | {W.BATCH[s]} | {2 * len(wh.shelves)} |")
    return "\n".join(lines)


def table_seq():
    lines = ["| size | orders | stops, mean (min–max) | NN gap %, mean / p95 | 2-opt gap %, mean / p95 | NN optimal % | 2-opt optimal % | "
             "NN µs | 2-opt µs | LKH-3 ms, mean / p95 |", "|---|---|---|---|---|---|---|---|---|---|"]
    hist = []
    for s in SIZES:
        rows = json.load(open(f"{OUT}/seq_{s}.json"))
        stops = np.array([r["stops"] for r in rows])
        lk = np.array([r["lkh"] for r in rows], dtype=np.float64)
        nn = 100 * (np.array([r["nn"] for r in rows]) / lk - 1)
        op = 100 * (np.array([r["2opt"] for r in rows]) / lk - 1)
        ms = 1e3 * np.array([r["lkh_s"] for r in rows])
        lines.append(f"| {s} | {len(rows)} | {stops.mean():.1f} ({stops.min()}–{stops.max()}) | {nn.mean():.1f} / {np.percentile(nn, 95):.1f} | "
                     f"{op.mean():.2f} / {np.percentile(op, 95):.1f} | {100 * (nn == 0).mean():.0f} | {100 * (op == 0).mean():.0f} | "
                     f"{1e6 * np.mean([r['nn_s'] for r in rows]):.0f} | {1e6 * np.mean([r['2opt_s'] for r in rows]):.0f} | "
                     f"{ms.mean():.1f} / {np.percentile(ms, 95):.1f} |")
        hist.append((s, np.bincount(stops, minlength=13)))
    lines += ["", "Stops per order (distinct shelves):", "", "| size | " + " | ".join(str(k) for k in range(1, 13)) + " |",
              "|---|" + "---|" * 12]
    lines += [f"| {s} | " + " | ".join(str(int(h[k])) for k in range(1, 13)) + " |" for s, h in hist]
    return "\n".join(lines)


def table_e1(rows):
    lines = ["| size | sequence | planner | delivered | makespan | sum of completion (k) | mean delay | orders/h | step ms, mean | p50 | p95 | max | steps under PIBT % | CBS time-outs |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SIZES:
        for q in PLANNERS:
            for m in METHODS:
                rs = [r for r in rows if (r["size"], r["method"], r["planner"]) == (s, m, q)]
                if not rs:
                    continue
                fb = [100 * r["planner_stats"]["fallbacks"] / r["steps"] for r in rs]
                lines.append(f"| {s} | {m} | {q} | {np.mean([r['delivered'] for r in rs]):.0f}/{W.BATCH[s]} | {fmt([r['makespan'] for r in rs])} | "
                             f"{fmt([r['sum_completion'] / 1e3 for r in rs], 1)} | {fmt([r['mean_delay'] for r in rs])} | "
                             f"{fmt([r['orders_per_hour'] for r in rs])} | {fmt([r['step_ms_mean'] for r in rs], 2)} | "
                             f"{np.mean([r['step_ms_p50'] for r in rs]):.2f} | {np.mean([r['step_ms_p95'] for r in rs]):.1f} | "
                             f"{np.max([r['step_ms_max'] for r in rs]):.0f} | {np.mean(fb):.1f} | {np.mean([r['planner_stats']['timeouts'] for r in rs]):.1f} |")
    return "\n".join(lines)


def table_paired(rows):
    """Paired by seed, since a seed fixes the orders: mean delay of b against a."""
    get = lambda s, m, q: np.array([r["mean_delay"] for r in sorted(rows, key=lambda r: r["seed"]) if (r["size"], r["method"], r["planner"]) == (s, m, q)])
    pairs = [("nn + pibt", "lkh + pibt"), ("2opt + pibt", "lkh + pibt"), ("2opt + prioritized", "lkh + prioritized"),
             ("lkh + pibt", "lkh + prioritized"), ("lkh + prioritized", "lkh + cbs")]
    lines = ["| size | from | to | delay change % | p (paired t, 5 seeds) |", "|---|---|---|---|---|"]
    for s in SIZES:
        for a, b in pairs:
            x, y = get(s, *a.split(" + ")), get(s, *b.split(" + "))
            if len(x) and len(y):
                lines.append(f"| {s} | {a} | {b} | {100 * (y.mean() / x.mean() - 1):+.1f} | {ttest_rel(x, y).pvalue:.3g} |")
    return "\n".join(lines)


def fig_e1(rows):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=130)
    marks = {"nn": "o", "2opt": "s", "lkh": "^"}
    cols = {"pibt": "#2563eb", "prioritized": "#ea580c", "cbs": "#059669"}
    for ax, s in zip(axes, SIZES):
        for q in PLANNERS:
            for m in METHODS:
                rs = [r for r in rows if (r["size"], r["method"], r["planner"]) == (s, m, q)]
                if rs:
                    mean, half = ci([r["mean_delay"] for r in rs])
                    ax.errorbar(np.mean([r["step_ms_mean"] for r in rs]), mean, yerr=half, marker=marks[m], color=cols[q], capsize=2, ms=5,
                                label=f"{m} + {q}")
        ax.set_xscale("log")
        ax.set_title(s, fontsize=10)
        ax.set_xlabel("computation per step, mean (ms)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean delay per order (steps)")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{FIG}/phaseA_e1_delay_vs_time.png")
    plt.close(fig)


def backlog(r):
    """Orders arrived and not delivered, at each step."""
    a = np.bincount(np.minimum(r["arrivals"], STEPS), minlength=STEPS + 1).cumsum()
    d = np.bincount(np.minimum(r["deliveries"], STEPS), minlength=STEPS + 1).cumsum()
    return a - d


def slope(r):
    b = backlog(r)[STEPS // 2:]
    return float(np.polyfit(np.arange(len(b)), b, 1)[0])


def keeps_up(rs):
    """Keeps up if the backlog grows, over the second half, at less than 5 % of the arrival rate."""
    return np.mean([slope(r) for r in rs]) < 0.05 * rs[0]["rate"]


def throughput(r):
    """Orders delivered per step in the second half of the episode."""
    return sum(d >= STEPS // 2 for d in r["deliveries"]) / (STEPS - STEPS // 2)


def table_e2(rows):
    lines = ["| size | rate name | arrivals per step | planner | arrived | delivered | orders/h | backlog at the end | backlog slope, % of the rate | mean delay of the delivered | step ms, mean | p95 | max | keeps up |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SIZES:
        for name in RATES:
            for q in ("pibt", "prioritized"):
                rs = [r for r in rows if (r["size"], r["rate_name"], r["planner"]) == (s, name, q)]
                if not rs:
                    continue
                lines.append(f"| {s} | {name} | {rs[0]['rate']:.3f} | {q} | {np.mean([r['orders'] for r in rs]):.0f} | {fmt([r['delivered'] for r in rs])} | "
                             f"{fmt([r['orders_per_hour'] for r in rs])} | {fmt([backlog(r)[-1] for r in rs])} | "
                             f"{100 * np.mean([slope(r) for r in rs]) / rs[0]['rate']:.0f} | {fmt([r['mean_delay'] for r in rs])} | "
                             f"{fmt([r['step_ms_mean'] for r in rs], 2)} | {np.mean([r['step_ms_p95'] for r in rs]):.1f} | "
                             f"{np.max([r['step_ms_max'] for r in rs]):.0f} | {'yes' if keeps_up(rs) else 'no'} |")
    return "\n".join(lines)


def table_limits(rows):
    lines = ["| size | capacity of E1 (orders per step) | planner | failed at | stops keeping up at | throughput at 2.0 x, second half (orders per step) | same, orders/h |",
             "|---|---|---|---|---|---|---|"]
    for s in SIZES:
        cap = next(r["rate"] for r in rows if r["size"] == s and r["rate_name"] == "x1.0")
        for q in ("pibt", "prioritized"):
            runs = sorted(((rs[0]["rate"], rs) for rs in ([x for x in rows if (x["size"], x["rate_name"], x["planner"]) == (s, n, q)] for n in RATES) if rs),
                          key=lambda c: c[0])
            ok = [keeps_up(rs) for _, rs in runs]
            # the first rate that fails together with every higher one: one failure below that is noise of the arrivals
            first = next((k for k in range(len(ok)) if not any(ok[k:])), None)
            stop = "not reached" if first is None else f"{runs[first][0] / cap:.1f} x ({runs[first][0]:.3f})"
            fails = ", ".join(f"{a / cap:.1f}" for (a, _), good in zip(runs, ok) if not good) or "none"
            sat = [throughput(r) for r in runs[-1][1]]
            lines.append(f"| {s} | {cap:.3f} | {q} | {fails} | {stop} | {fmt(sat, 3)} | {3600 * np.mean(sat):.0f} |")
    return "\n".join(lines)


def fig_e2(rows):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), dpi=130, sharex=True)
    style = {"pibt": "-", "prioritized": "--"}
    cols = {"loose": "#16a34a", "medium": "#ca8a04", "saturated": "#dc2626", "x2.0": "#7c3aed"}
    x = np.arange(STEPS + 1)
    for k, s in enumerate(SIZES):
        for name in cols:
            for q in ("pibt", "prioritized"):
                rs = [r for r in rows if (r["size"], r["rate_name"], r["planner"]) == (s, name, q)]
                if not rs:
                    continue
                d = np.mean([np.bincount(np.minimum(r["deliveries"], STEPS), minlength=STEPS + 1).cumsum() for r in rs], axis=0)
                axes[0, k].plot(x, d, style[q], color=cols[name], label=f"{name}, {q}", lw=1.2)
                axes[1, k].plot(np.arange(STEPS), np.mean([np.load(f"{OUT}/e2b/{s}_{name}_{q}_{r['seed']}.npz")["step_seconds"][:STEPS] for r in rs], axis=0).cumsum(),
                                style[q], color=cols[name], lw=1.2)
        axes[0, k].set_title(s, fontsize=10)
        axes[1, k].set_xlabel("step")
        axes[0, k].grid(alpha=0.3)
        axes[1, k].grid(alpha=0.3)
    axes[0, 0].set_ylabel("orders delivered")
    axes[1, 0].set_ylabel("computation, cumulative (s)")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{FIG}/phaseA_e2_curves.png")
    plt.close(fig)


def e3(rows):
    lines = ["| size | planner | runs | robot-steps lost % | waits | detours | cells with a wait % | waits in the hottest 10 % of cells, per episode | same, cells chosen on all runs together | same, cells chosen on the other seeds | busiest 10 % by visits, chosen on the other seeds | visits in the hottest 10 %, per episode | waits with a blocker % | waits in the hottest 1 % of (cell, 50 steps) boxes | boxes with a wait % |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary, maps = {}, {}
    for s in SIZES:
        wh = W.Warehouse(W.SIZES[s])
        free = wh.grid != W.SHELF
        for q in PLANNERS:
            rs = [r for r in rows if r["size"] == s and r["planner"] == q]
            if not rs:
                continue
            data = [(r, np.load(f"{OUT}/conflicts_{s}_{r['method']}_{q}_{r['seed']}.npz")) for r in rs]
            w = [d["waits"] for _, d in data]
            hot = [C.hot_share(x, free) for x in w]
            vis = [C.hot_share(d["visits"], free) for _, d in data]
            cross, traffic = [], []
            for seed in range(5):
                other = sum(d["waits"] for r, d in data if r["seed"] != seed)
                mask = C.hot_cells(other, free)
                busy = C.hot_cells(sum(d["visits"] for r, d in data if r["seed"] != seed), free)
                for r, d in data:
                    if r["seed"] == seed:
                        cross.append(d["waits"][mask].sum() / max(d["waits"].sum(), 1))
                        traffic.append(d["waits"][busy].sum() / max(d["waits"].sum(), 1))
            blocked, tops, boxes = [], [], []
            for r, d in data:
                ev = d["events"]
                x = ev[ev[:, 3] == 0]
                blocked.append(100 * (x[:, 4] >= 0).mean() if len(x) else 0.0)
                a, b = C.cell_time_share(ev, len(wh.free), r["steps"])
                tops.append(a)
                boxes.append(100 * b)
            lost = [100 * int(d["lost"]) / max(int(d["moving"]), 1) for _, d in data]
            summary[(s, q)] = float(np.mean(hot))
            maps[(s, q)] = sum(w)
            lines.append(f"| {s} | {q} | {len(rs)} | {np.mean(lost):.1f} | {np.mean([x.sum() for x in w]):.0f} | {np.mean([int(d['detours'].sum()) for _, d in data]):.0f} | "
                         f"{100 * np.mean([(x[free] > 0).mean() for x in w]):.1f} | {100 * np.mean(hot):.1f} % | {100 * C.hot_share(sum(w), free):.1f} % | "
                         f"{100 * np.mean(cross):.1f} % | {100 * np.mean(traffic):.1f} % | {100 * np.mean(vis):.1f} % | "
                         f"{np.mean(blocked):.0f} | {100 * np.mean(tops):.1f} % | {np.mean(boxes):.1f} |")
        viz.heatmap(wh, maps[(s, "pibt")], f"Waits per cell, {s}, PIBT, all E1 runs", f"{FIG}/phaseA_e3_heat_{s}.png")
    return "\n".join(lines), summary


def e2b_wall():
    parse = lambda x: datetime.strptime(x.replace(" EST", ""), "%a %b %d %H:%M:%S %Y")
    a, b = re.findall(r"^=== .*?(\w{3} \w{3} +\d+ [\d:]+ EST \d{4})$", open(f"{OUT}/run_e2b.log").read(), flags=re.M)
    return parse(b) - parse(a)


def time_spent():
    stamps = re.findall(r"^=== (\w+)\s+(.+)$", open(f"{OUT}/run.log").read(), flags=re.M)
    lines, last = ["| stage | wall clock |", "|---|---|"], None
    parse = lambda x: datetime.strptime(x.replace(" EST", ""), "%a %b %d %H:%M:%S %Y")
    marks = [(a, parse(b.strip())) for a, b in stamps if a != "ALL"]
    end = re.search(r"^=== ALL DONE (.+)$", open(f"{OUT}/run.log").read(), flags=re.M)
    finish = parse(end.group(1).strip()) if end else None
    for (a, t), nxt in zip(marks, marks[1:] + [(None, finish)]):
        if nxt[1]:
            lines.append(f"| {a} | {nxt[1] - t} |")
    if finish:
        lines.append(f"| total of the first pass | {finish - marks[0][1]} |")
    lines += [f"| e2, repeated with more rates ({len(load('e2b'))} episodes) | {e2b_wall()} |"]
    cpu = sum(r["elapsed"] for r in load("e1") + load("e2") + load("e2b"))
    lines += ["", f"Compute time of the E1 and E2 episodes, both E2 runs included, added over the processes: {cpu / 3600:.2f} h."]
    return "\n".join(lines)


def section(notes, name):
    parts = notes.split(f"## {name}\n", 1)
    return parts[1].split("\n## ", 1)[0].strip() if len(parts) == 2 else f"(notes missing: {name})"


def main():
    e1, e2 = load("e1"), load("e2b")
    fig_e1(e1)
    fig_e2(e2)
    t3, hot = e3(e1)
    notes = open("docs/phaseA_notes.md").read()
    bad = [r for r in e1 + e2 if not r["ok"]]
    done = sum(r["delivered"] == r["orders"] for r in e1)
    out = ["# Warehouse, phase A: simulator and classic baselines", "", "## Conclusion", "", section(notes, "Conclusion"), "",
           "## Design", "", section(notes, "Design"), "", table_layouts(), "", "## Layer 1: visit order", "", section(notes, "Layer 1"), "",
           table_seq(), "", "## Verifier", "", f"All {len(e1) + len(e2)} episodes were judged by `verify.py` from the saved trajectory: "
           f"{len(e1) + len(e2) - len(bad)} passed, {len(bad)} failed. Collisions (vertex and edge): "
           f"{sum(r['problems']['vertex_conflicts'] + r['problems']['edge_conflicts'] for r in e1 + e2)}. E1 episodes with every order delivered: {done} of {len(e1)}. "
           f"E2 stops at step {STEPS}, so a started order may be unfinished there; those were {sum(r['open'] for r in e2)} over the E2 episodes, and their picks were judged too.", "",
           "## E1: offline lots", "", section(notes, "E1"), "", table_e1(e1), "", table_paired(e1), "", "![Delay against computation](figures/phaseA_e1_delay_vs_time.png)", "",
           "## E2: lifelong", "", section(notes, "E2"), "", table_e2(e2), "", table_limits(e2), "", "![Orders over time](figures/phaseA_e2_curves.png)", "",
           "## E3: where the conflicts are", "", section(notes, "E3"), "", t3, ""]
    out += [f"![Waits, {s}](figures/phaseA_e3_heat_{s}.png)\n" for s in SIZES]
    out += ["## Episode", "", "![A small episode with PIBT](figures/phaseA_small_pibt.gif)", "", "## Time spent", "", time_spent(), "",
            "## Limitations", "", section(notes, "Limitations"), ""]
    open("docs/phaseA_report.md", "w").write("\n".join(out))
    print("saved docs/phaseA_report.md")


if __name__ == "__main__":
    main()
