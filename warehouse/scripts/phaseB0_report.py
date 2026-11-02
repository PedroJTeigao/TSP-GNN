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
ROWS = [("O3", "o3a", "(b) seed, shelf overlap, 2 per trip"), ("O3", "o3b", "(b) control, least added walking"),
        ("O3", "o3c", "(c) Clarke & Wright, 2 per trip"), ("O3", "o3c3", "(c) Clarke & Wright, 3 per trip"), ("O3", "o3d", "(d) k-means regions"),
        ("O3", "o3e", "(e) savings on the traffic map"), ("O3", "o3x", "oracle: exact pairs of the whole lot"),
        ("O1", "o1a", "Hungarian over the queue"), ("O1", "o1a_fifo", "control: same, oldest orders only"),
        ("O1", "o1b", "(b) Hungarian + reassignment"), ("O1", "o1b0", "control: same, without the order length"),
        ("O1", "o1it", "(c) greedy + iterative reassignment"), ("O1", "o1dw", "(d) Hungarian on decayed edge waits"),
        ("O1", "o1dg", "(d) Hungarian, delay near first goal x gamma"), ("O1", "o1e", "(e) oracle: MCA + LNS"), ("O1", "o1c", "oracle: 200 simulated swaps"),
        ("O2-A", "o2ah1", "(i) crisscross, wrong way +1"), ("O2-A", "o2ah2", "(i) crisscross, wrong way +2"), ("O2-A", "o2ag", "(ii) GGO-lite, CMA-ES"),
        ("O2-A", "o2af1", "(iii) flow costs 0.1 / 0.5"), ("O2-A", "o2af2", "(iii) flow costs 0.3 / 1.0"),
        ("O2-B", "o2b", "first version: static cell map"), ("O2-B", "o2a", "first version: hindsight per cell"),
        ("O2-B", "o2b1", "(i) static edge map, on o1b"), ("O2-B", "o2b2", "(ii) same x gamma"), ("O2-B", "o2b3", "(iii) oracle: hindsight per edge"),
        ("O4", "o4cbs", "windowed CBS, 3 x limits"), ("O4", "o4pibt5", "PIBT, best of 5 rules"), ("O4", "o4w5", "window 5"), ("O4", "o4w10", "window 10")]
OFFLINE = {"o1c", "o1e", "o2a", "o2b3", "o4pibt5"}
NO_FF = {"o4pibt5"}                      # its free flow ran before the arm existed, so it repeats the baseline's plan
HINDSIGHT_RIVALS = ("o1a", "o1b", "o1it", "o1dw", "o1dg", "o2b1", "o2b2")


def load():
    runs = {}
    for p in glob.glob(f"{OUT}/*.json"):
        r = json.load(open(p))
        if "arm" not in r or r["arm"] == "o4pibt":   # ggo files, and the first version, where a rule that livelocked could win
            continue
        runs[(r["arm"], r["size"], r["seed"])] = r
        assert "ok" not in r or (r["ok"] and r["delivered"] == r["orders"]), f"bad run: {p}"
    return runs


def series(runs, arm, size, key):
    return np.array([runs[(arm, size, s)][key] for s in range(5)], dtype=np.float64)


def has(runs, arm, size):
    return all((arm, size, s) in runs for s in range(5))


def paired(base, x):
    """(% gain per seed of x against base, its mean, half of the 95 % interval, Wilcoxon p)."""
    g = 100 * (base - x) / base
    half = float(student.ppf(0.975, 4) * g.std(ddof=1) / np.sqrt(5))
    p = 1.0 if not np.any(base - x) else float(wilcoxon(base, x, method="exact" if np.all(base != x) else "auto").pvalue)
    return g, float(g.mean()), half, p


def gain(runs, arm, size, key):
    return paired(series(runs, "base", size, key), series(runs, arm, size, key))


def cell(g):
    return f"{g[1]:+.1f} ± {g[2]:.1f} (p={g[3]:.3g})"


def cost(runs, arm, size):
    ms = series(runs, arm, size, "step_ms_mean").mean() if (arm, size, 0) in runs and "step_ms_mean" in runs[(arm, size, 0)] else 0.0
    if arm in OFFLINE:
        return f"{ms:.1f} ({series(runs, arm, size, 'elapsed').mean():.0f} s offline)"
    return f"{ms:.1f}"


def of_ff(runs, arm, size, ref):
    return f"{100 * np.mean(series(runs, arm, size, 'makespan') / series(runs, f'ff_{ref}', size, 'makespan')):.0f}"


def table(runs):
    lines = ["| size | component | arm | makespan | % own free flow | % baseline free flow | x best of component | ms/step | makespan gain % | delay gain % |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SIZES:
        lines.append(f"| {s} | baseline | LKH-3 + prioritized, nearest robot, no batching | {series(runs, 'base', s, 'makespan').mean():.0f} | {of_ff(runs, 'base', s, 'base')} | {of_ff(runs, 'base', s, 'base')} | | {cost(runs, 'base', s)} | | |")
        for comp in dict.fromkeys(c for c, _, _ in ROWS):
            arms = [(a, n) for c, a, n in ROWS if c == comp and has(runs, a, s)]
            if not arms:
                continue
            means = {a: series(runs, a, s, "makespan").mean() for a, _ in arms}
            best = min(means.values())
            for a, n in arms:
                own = of_ff(runs, a, s, a) if has(runs, f"ff_{a}", s) and a not in NO_FF else "-"
                lines.append(f"| {s} | {comp} | {a}: {n} | {means[a]:.0f} | {own} | {of_ff(runs, a, s, 'base')} | {means[a] / best:.3f} | {cost(runs, a, s)} | {cell(gain(runs, a, s, 'makespan'))} | {cell(gain(runs, a, s, 'mean_delay'))} |")
            lines.append(f"| {s} | {comp} | **worst / best** | | | | **{max(means.values()) / best:.3f}** | | | |")
    return "\n".join(lines)


def best_rival(runs, s, key):
    return min(HINDSIGHT_RIVALS, key=lambda a: series(runs, a, s, key).mean())


def table_hindsight(runs):
    lines = ["| size | metric | hindsight | same dispatcher without traffic (o1b) | gain % | best not learned | gain % |", "|---|---|---|---|---|---|---|"]
    for s in ("medium", "large"):
        for key, name in (("makespan", "makespan"), ("mean_delay", "mean delay")):
            h, o = series(runs, "o2b3", s, key), series(runs, "o1b", s, key)
            r = best_rival(runs, s, key)
            x = series(runs, r, s, key)
            lines.append(f"| {s} | {name} | {h.mean():.1f} | {o.mean():.1f} | {cell(paired(o, h))} | {r}: {x.mean():.1f} | {cell(paired(x, h))} |")
    return "\n".join(lines)


def table_ceiling(runs):
    lines = ["| size | arm | makespan | free flow | steps lost to traffic % | delay | free-flow delay | delay lost % |", "|---|---|---|---|---|---|---|---|"]
    for s in SIZES:
        for a in ("base", "o1b", "o2b3", "o3c", "o3c3"):
            if has(runs, a, s):
                m, f = series(runs, a, s, "makespan"), series(runs, f"ff_{a}", s, "makespan")
                d, g = series(runs, a, s, "mean_delay"), series(runs, f"ff_{a}", s, "mean_delay")
                lines.append(f"| {s} | {a} | {m.mean():.0f} | {f.mean():.0f} | {100 * np.mean((m - f) / m):.1f} | {d.mean():.0f} | {g.mean():.0f} | {100 * np.mean((d - g) / d):.1f} |")
    return "\n".join(lines)


def table_bound(runs):
    lines = ["| size | baseline makespan | bound | largest possible gain % | best one-order arm % |", "|---|---|---|---|---|"]
    for s in ("medium", "large"):
        base = series(runs, "base", s, "makespan")
        b = series(runs, "bound", s, "makespan_bound")
        one = [a for c, a, _ in ROWS if c in ("O1", "O2-B") and has(runs, a, s)]
        lines.append(f"| {s} | {base.mean():.0f} | {b.mean():.0f} | {100 * np.mean((base - b) / base):.1f} | {max(gain(runs, a, s, 'makespan')[1] for a in one):.1f} |")
    return "\n".join(lines)


def table_o3(runs):
    lines = ["| size | subsets of 20 orders | savings over exact % (mean, worst) | LKH-3 CVRP = exact | whole lot: exact walk | savings over exact % |", "|---|---|---|---|---|---|"]
    for s in ("medium", "large"):
        rows = [x for k in range(5) for x in runs[("o3f", s, k)]["subsets"]]
        over = [100 * (x["savings"] / x["exact"] - 1) for x in rows]
        same = sum(abs(x["cvrp"] - x["exact"]) < 1e-6 for x in rows)
        whole = series(runs, "o3x", s, "walk")
        sav = series(runs, "o3x", s, "walk_savings")
        lines.append(f"| {s} | {len(rows)} | {np.mean(over):.2f}, {max(over):.2f} | {same} of {len(rows)} | {whole.mean():.0f} | {100 * np.mean(sav / whole - 1):.2f} |")
    return "\n".join(lines)


def table_oracles(runs):
    lines = ["| size | seed | o1c start | o1c end | swaps kept of 200 | o1e start | o1e end |", "|---|---|---|---|---|---|---|"]
    for s in ("medium", "large"):
        for k in range(5):
            r = runs[("o1c", s, k)]
            e = runs.get(("o1e", s, k))
            lines.append(f"| {s} | {k} | {r['history'][0]:.0f} | {r['makespan']:.0f} | {r['kept']} | {e['start'] if e else '-'} | {e['makespan'] if e else '-'} |")
    return "\n".join(lines)


def table_iterations(runs):
    lines = ["| size | iteration | makespan | mean delay |", "|---|---|---|---|"]
    for s in ("medium", "large"):
        for it in range(3):
            rs = [runs[("o2b3", s, k)]["iterations"][it] for k in range(5)]
            lines.append(f"| {s} | {it + 1} | {np.mean([r['makespan'] for r in rs]):.0f} | {np.mean([r['mean_delay'] for r in rs]):.0f} |")
    return "\n".join(lines)


def table_ggo(runs):
    g, b = json.load(open(f"{OUT}/ggo.json")), json.load(open(f"{OUT}/ggo_base.json"))
    return (f"The search ran {len(g['log'])} generations in {g['log'][-1]['seconds'] / 60:.0f} min. On its own two seeds the baseline makes {np.mean(b['makespan']):.1f} steps and the best weights "
            f"{g['train_makespan']:.1f}; on the five seeds of the report they make {series(runs, 'o2ag', 'medium', 'makespan').mean():.0f} against {series(runs, 'base', 'medium', 'makespan').mean():.0f}. "
            f"Best wrong-way weights: {', '.join(f'{x:.2f}' for x in g['best'])}.")


def table_rules(runs):
    rules = ("age", "far", "near", "load", "fixed")
    lines = ["| size | " + " | ".join(rules) + " | best of 5 | window 20 prioritized |", "|---|---|---|---|---|---|---|---|"]
    for s in ("small", "medium"):
        rows = [runs[("o4pibt5", s, k)]["all_rules"] for k in range(5)]
        cols = [np.mean([next(x["makespan"] for x in row if x["rule"] == q) for row in rows]) for q in rules]
        lines.append(f"| {s} | " + " | ".join(f"{c:.0f}" for c in cols) + f" | {series(runs, 'o4pibt5', s, 'makespan').mean():.0f} | {series(runs, 'base', s, 'makespan').mean():.0f} |")
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
    arms = [(a, f"{c} {a}") for c, a, _ in ROWS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), dpi=130, sharey=True)
    y = np.arange(len(arms))[::-1]
    for ax, key, title in zip(axes, ("makespan", "mean_delay"), ("makespan", "mean delay")):
        for k, (s, colour) in enumerate(zip(SIZES, ("#94a3b8", "#2563eb", "#ea580c"))):
            for j, (a, _) in enumerate(arms):
                if has(runs, a, s):
                    g = gain(runs, a, s, key)
                    ax.barh(y[j] + 0.27 * (1 - k), g[1], 0.26, xerr=g[2], color=colour, capsize=2, error_kw={"lw": 0.8},
                            label=s if j == next(i for i, (b, _) in enumerate(arms) if has(runs, b, s)) else None)
            if s != "small":
                b, f = series(runs, "base", s, key), series(runs, "ff_base", s, key)
                ax.axvline(100 * np.mean((b - f) / b), color=colour, lw=0.9, ls="--")
        ax.axvline(0, color="k", lw=0.7)
        ax.set_title(f"{title}: gain against LKH-3 + prioritized", fontsize=10)
        ax.set_xlabel("gain %, with 95 % interval", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [n for _, n in arms], fontsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle("Dashed: gain if all traffic of the baseline were removed (free flow)", fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)


def time_spent(runs):
    text = open("results/warehouse/phaseB0.revised.log").read()
    parse = lambda x: datetime.strptime(x.replace(" EST", ""), "%a %b %d %H:%M:%S %Y")
    marks = [(a, parse(b.strip())) for a, b in re.findall(r"^=== (\w+)\s+(.+)$", text, flags=re.M) if a not in ("ALL", "NEW")]
    marks.append(("end", parse(re.search(r"^=== ALL DONE (.+)$", text, flags=re.M).group(1).strip())))
    lines = ["| stage (at most 4 processes) | wall clock |", "|---|---|"] + [f"| {a} | {y - x} |" for (a, x), (_, y) in zip(marks, marks[1:])]
    lines += [f"| total | {marks[-1][1] - marks[0][1]} |"]
    return "\n".join(lines)


def section(notes, name):
    parts = notes.split(f"## {name}\n", 1)
    return parts[1].split("\n## ", 1)[0].strip() if len(parts) == 2 else f"(notes missing: {name})"


def main():
    runs = load()
    figure(runs)
    notes = open("docs/phaseB0_notes.md").read()
    n = sum("ok" in r for r in runs.values())
    out = ["# Warehouse, phase B0: margins per component", "", "## Recommendation", "", section(notes, "Recommendation"), "",
           "## Margins", "", section(notes, "Margins"), "", table(runs), "", "![Gain per arm](figures/phaseB0_margins.png)", "",
           f"All {n} reported episodes passed the verifier with every order delivered; a free-flow run skips only the collision check.", "",
           "## Protocol and decisions", "", section(notes, "Protocol"), "",
           "## O3: batching", "", section(notes, "O3"), "", table_o3(runs), "",
           "## O1: assignment", "", section(notes, "O1"), "", table_bound(runs), "", table_oracles(runs), "",
           "## O2: traffic", "", section(notes, "O2"), "", table_ceiling(runs), "", table_hindsight(runs), "", table_iterations(runs), "", table_ggo(runs), "",
           "## O4: the path planner", "", section(notes, "O4"), "", table_cbs(runs), "", table_rules(runs), "",
           "## Budget and cuts", "", section(notes, "Budget"), "", time_spent(runs), "", "## Limitations", "", section(notes, "Limitations"), ""]
    open("docs/phaseB0_margins.md", "w").write("\n".join(out))
    print("saved docs/phaseB0_margins.md")


if __name__ == "__main__":
    main()
