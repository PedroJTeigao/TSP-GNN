"""Report of phase B0: docs/phaseB0_margins.md and its figure, from results/warehouse/phaseB0/.

The hand-written parts are read from docs/phaseB0_notes.md (one "## " section per part of the report); every number and
table is computed here. A run that fails the verifier or leaves an order undelivered stops the report.

Example:
    python scripts/phaseB0_report.py
"""

import glob
import json
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student
from scipy.stats import wilcoxon

OUT = "results/warehouse/phaseB0"
FIG = "docs/figures/phaseB0_margins.png"
SIZES = ("small", "medium", "large")
ROWS = [("o1a", "O1a minimum-cost assignment over the queue"), ("o1a_fifo", "O1a control: same, oldest orders only"),
        ("o1b", "O1b same, with reassignment"), ("o1c", "O1c a posteriori search, 200 swaps"),
        ("o2a", "O2a traffic in hindsight, best of 3"), ("o2b", "O2b fixed traffic map"),
        ("o3a", "O3a two orders per trip, shelf overlap"), ("o3b", "O3b control: two orders, least added walking"),
        ("o4cbs", "O4 CBS, 3 x time limit"), ("o4pibt5", "O4 PIBT, best of 5 rules")]
OFFLINE = {"o1c", "o2a", "o4pibt5"}


def load():
    runs = {}
    for p in glob.glob(f"{OUT}/*.json"):
        r = json.load(open(p))
        if r["arm"] == "o4pibt":                     # the first version, where a rule that livelocked could win
            continue
        runs[(r["arm"], r["size"], r["seed"])] = r
        assert r["arm"] == "bound" or (r["ok"] and r["delivered"] == r["orders"]), f"bad run: {p}"
    return runs


def series(runs, arm, size, key):
    return np.array([runs[(arm, size, s)][key] for s in range(5)], dtype=np.float64)


def gain(runs, arm, size, key):
    """(% gain per seed against the baseline, its mean, half of the 95 % interval, Wilcoxon p)."""
    base, x = series(runs, "base", size, key), series(runs, arm, size, key)
    g = 100 * (base - x) / base
    half = float(student.ppf(0.975, 4) * g.std(ddof=1) / np.sqrt(5))
    p = 1.0 if not np.any(base - x) else float(wilcoxon(base, x, method="exact" if np.all(base != x) else "auto").pvalue)
    return g, float(g.mean()), half, p


def cell(g):
    return f"{g[1]:+.1f} ± {g[2]:.1f} (p={g[3]:.3g})"


def cost(runs, arm, size):
    if arm in OFFLINE:
        return f"{series(runs, arm, size, 'elapsed').mean():.0f} s per episode, offline"
    return f"{series(runs, arm, size, 'step_ms_mean').mean():.1f} ms"


def table(runs):
    lines = ["| oracle | size | makespan gain % | mean delay gain % | orders/h gain % | computation per step (baseline in the first row of each size) |",
             "|---|---|---|---|---|---|"]
    for s in SIZES:
        lines.append(f"| baseline | {s} | {series(runs, 'base', s, 'makespan').mean():.0f} steps | {series(runs, 'base', s, 'mean_delay').mean():.0f} steps | "
                     f"{series(runs, 'base', s, 'orders_per_hour').mean():.0f} | {cost(runs, 'base', s)} |")
        for arm, name in ROWS:
            if (arm, s, 0) not in runs:
                continue
            o = gain(runs, arm, s, "orders_per_hour")
            lines.append(f"| {name} | {s} | {cell(gain(runs, arm, s, 'makespan'))} | {cell(gain(runs, arm, s, 'mean_delay'))} | {-o[1]:+.1f} | {cost(runs, arm, s)} |")
    return "\n".join(lines)


def table_bound(runs):
    lines = ["| size | baseline makespan | bound | largest possible gain % | best one-order arm % |", "|---|---|---|---|---|"]
    for s in ("medium", "large"):
        base = series(runs, "base", s, "makespan")
        b = series(runs, "bound", s, "makespan_bound")
        best = max(gain(runs, a, s, "makespan")[1] for a in ("o1a", "o1b", "o1c"))
        lines.append(f"| {s} | {base.mean():.0f} | {b.mean():.0f} | {100 * np.mean((base - b) / base):.1f} | {best:.1f} |")
    return "\n".join(lines)


def table_o1c(runs):
    lines = ["| size | seed | start | end | swaps kept of 200 |", "|---|---|---|---|---|"]
    for s in ("medium", "large"):
        for k in range(5):
            r = runs[("o1c", s, k)]
            lines.append(f"| {s} | {k} | {r['history'][0]:.0f} | {r['makespan']:.0f} | {r['kept']} |")
    return "\n".join(lines)


def table_o2(runs):
    lines = ["| size | iteration | makespan | mean delay | mean penalty of a cell with losses (steps per pass) |", "|---|---|---|---|---|"]
    for s in ("medium", "large"):
        for it in range(3):
            rs = [runs[("o2a", s, k)]["iterations"][it] for k in range(5)]
            lines.append(f"| {s} | {it + 1} | {np.mean([r['makespan'] for r in rs]):.0f} | {np.mean([r['mean_delay'] for r in rs]):.0f} | {np.mean([r['mean_penalty'] for r in rs]):.2f} |")
    return "\n".join(lines)


def table_rules(runs):
    rules = ("age", "far", "near", "load", "fixed")
    lines = ["| size | " + " | ".join(rules) + " | best of 5 |", "|---|---|---|---|---|---|---|"]
    for s in ("small", "medium"):
        rows = [runs[("o4pibt5", s, k)]["all_rules"] for k in range(5)]
        cols = [np.mean([next(x["makespan"] for x in row if x["rule"] == q) for row in rows]) for q in rules]
        lines.append(f"| {s} | " + " | ".join(f"{c:.0f}" for c in cols) + f" | {series(runs, 'o4pibt5', s, 'makespan').mean():.0f} |")
    return "\n".join(lines)


def table_cbs(runs):
    lines = ["| size | steps under PIBT % | time-outs per episode | seconds in CBS per episode | episodes at the budget |", "|---|---|---|---|---|"]
    for s in ("small", "medium"):
        rs = [runs[("o4cbs", s, k)] for k in range(5)]
        st = [r["planner_stats"] for r in rs]
        lines.append(f"| {s} | {np.mean([100 * x['fallbacks'] / r['steps'] for x, r in zip(st, rs)]):.1f} | {np.mean([x['timeouts'] for x in st]):.1f} | "
                     f"{np.mean([x['spent'] for x in st]):.0f} | {sum(x['spent'] >= 2699 for x in st)} of 5 |")
    return "\n".join(lines)


def figure(runs):
    arms = [(a, n.split(",")[0]) for a, n in ROWS if a != "o1a_fifo"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130, sharey=True)
    y = np.arange(len(arms))[::-1]
    for ax, key, title in zip(axes, ("makespan", "mean_delay"), ("makespan", "mean delay")):
        for k, (s, colour) in enumerate(zip(SIZES, ("#94a3b8", "#2563eb", "#ea580c"))):
            for j, (a, _) in enumerate(arms):
                if (a, s, 0) in runs:
                    g = gain(runs, a, s, key)
                    ax.barh(y[j] + 0.27 * (1 - k), g[1], 0.26, xerr=g[2], color=colour, capsize=2, error_kw={"lw": 0.8},
                            label=s if j == next(i for i, (b, _) in enumerate(arms) if (b, s, 0) in runs) else None)
        ax.axvline(0, color="k", lw=0.7)
        ax.set_title(f"{title}: gain against LKH-3 + prioritized", fontsize=10)
        ax.set_xlabel("gain %")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [n for _, n in arms], fontsize=8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)


def time_spent(runs):
    text = open("results/warehouse/phaseB0.run.log").read()
    parse = lambda x: datetime.strptime(x.replace(" EST", ""), "%a %b %d %H:%M:%S %Y")
    marks = [(a, parse(b.strip())) for a, b in re.findall(r"^=== (\w+)\s+(.+)$", text, flags=re.M) if a != "ALL"]
    marks.append(("end", parse(re.search(r"^=== ALL DONE (.+)$", text, flags=re.M).group(1).strip())))
    lines = ["| stage | wall clock |", "|---|---|"] + [f"| {a} | {y - x} |" for (a, x), (_, y) in zip(marks, marks[1:])]
    lines += [f"| total | {marks[-1][1] - marks[0][1]} |", "", f"Compute time of all episodes, added over the processes: {sum(r.get('elapsed', 0.0) for r in runs.values()) / 3600:.1f} h."]
    return "\n".join(lines)


def section(notes, name):
    parts = notes.split(f"## {name}\n", 1)
    return parts[1].split("\n## ", 1)[0].strip() if len(parts) == 2 else f"(notes missing: {name})"


def main():
    runs = load()
    figure(runs)
    notes = open("docs/phaseB0_notes.md").read()
    n = sum(r["arm"] != "bound" for r in runs.values())
    out = ["# Warehouse, phase B0: where the margin is", "", "## Recommendation", "", section(notes, "Recommendation"), "",
           "## Margins", "", section(notes, "Margins"), "", table(runs), "", "![Gain per oracle](figures/phaseB0_margins.png)", "",
           f"All {n} reported episodes passed the verifier with every order delivered.", "",
           "## How each oracle was built", "", section(notes, "Oracles"), "",
           "## O1: assignment", "", section(notes, "O1"), "", table_bound(runs), "", table_o1c(runs), "",
           "## O2: traffic", "", section(notes, "O2"), "", table_o2(runs), "",
           "## O3: two orders per trip", "", section(notes, "O3"), "",
           "## O4: the path planner", "", section(notes, "O4"), "", table_cbs(runs), "", table_rules(runs), "",
           "## Time spent", "", time_spent(runs), "", "## Limitations", "", section(notes, "Limitations"), ""]
    open("docs/phaseB0_margins.md", "w").write("\n".join(out))
    print("saved docs/phaseB0_margins.md")


if __name__ == "__main__":
    main()
