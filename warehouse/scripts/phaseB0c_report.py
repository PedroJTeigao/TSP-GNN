"""Report of phase B0c: docs/phaseB0c_realism.md and its figure, from results/warehouse/phaseB0c/.

The hand-written parts are read from docs/phaseB0c_notes.md (one "## " section per part of the report); every number and table is
computed here. A run that fails the verifier never reaches the results: the run stops first.

Example:
    python scripts/phaseB0c_report.py
"""

import glob
import json
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student
from scipy.stats import wilcoxon

from warehouse import env as W

OUT = "results/warehouse/phaseB0c"
LOG = "results/warehouse/phaseB0c2.log"
FIG = "docs/figures/phaseB0c_gap.png"
SEEDS = range(5)
CELLS = {"m30": ("medium", 30), "l60": ("large", 60), "m90": ("medium", 90)}
ARMS = [("a", "(a) blind"), ("b", "(b) turns known"), ("c", "(c) static map"), ("d", "(d) map x gamma"), ("e", "(e) oracle: function known"),
        ("e0", "control: (e) without the robot type"), ("f", "(f) oracle: hindsight, best of 3")]
RULES = ("b", "c", "d")
PLANNERS = ("pibt", "prioritized")
REGIMES = ("r1", "r2")


def paired(base, x):
    """(% gain per seed of x against base, its mean, half of the 95 % interval, Wilcoxon p)."""
    base, x = np.asarray(base, dtype=np.float64), np.asarray(x, dtype=np.float64)
    g = 100 * (base - x) / base
    half = float(student.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g)))
    p = 1.0 if not np.any(base - x) else float(wilcoxon(base, x, method="exact" if np.all(base != x) else "auto").pvalue)
    return g, float(g.mean()), half, p


def show(g):
    return f"{g[1]:+.1f} ± {g[2]:.1f} (p={g[3]:.3g})"


def load(cell, regime, planner, arm):
    """The 5 seeds of an arm, or None if any is missing."""
    rows = [f"{OUT}/{cell}_{regime}_{planner}_{arm}_{k}.json" for k in SEEDS]
    return [json.load(open(p)) for p in rows] if all(glob.glob(p) for p in rows) else None


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=np.float64)


def density(cell):
    size, robots = CELLS[cell]
    return 100 * robots / len(W.Warehouse(W.SIZES[size]).free)


def name(cell):
    size, robots = CELLS[cell]
    return f"{size}, {robots}"


def table_arms():
    lines = ["| planner | regime | warehouse, robots | arm | makespan | % free flow | mean delay | ms/step | own delay, steps per move | turns per move |", "|---|---|---|---|---|---|---|---|---|---|"]
    for q in PLANNERS:
        for r in REGIMES:
            for c in CELLS:
                for arm, label in ARMS:
                    R = load(c, r, q, arm)
                    if R is None:
                        continue
                    lost = (col(R, "slips") + col(R, "stops")).sum() / col(R, "moves").sum()
                    lines.append(f"| {q} | {r.upper()} | {name(c)} | {label} | {col(R, 'makespan').mean():.0f} | {100 * np.mean(col(R, 'makespan') / col(R, 'free_flow')):.0f} | "
                                 f"{col(R, 'mean_delay').mean():.0f} | {col(R, 'step_ms').mean():.1f} | {lost:.3f} | {col(R, 'turns').sum() / col(R, 'moves').sum():.3f} |")
    return "\n".join(lines)


def gaps(c, r, q, key="makespan"):
    """The deciding numbers of one cell, or None. The best of b, c, d is the lowest mean, so it leans towards the rules, as best of 3 leans towards f."""
    data = {a: load(c, r, q, a) for a in ("a", "b", "c", "d", "e", "f")}
    if any(v is None for v in data.values()):
        return None
    rival = min(RULES, key=lambda a: col(data[a], key).mean())
    best = col(data[rival], key)
    third = col(data["f"], key)
    if key == "makespan":
        third3 = np.mean([x["iterations"] for x in data["f"]], axis=1)
    else:
        third3 = np.mean([x["iteration_delays"] for x in data["f"]], axis=1)
    out = {"rival": rival, "best": best.mean(), "e": paired(best, col(data["e"], key)), "f": paired(best, third), "f3": paired(best, third3), "b": paired(col(data["a"], key), col(data["b"], key))}
    e0 = load(c, r, q, "e0")
    out["e0"] = paired(best, col(e0, key)) if e0 else None
    return out


def table_gaps(key="makespan"):
    lines = ["| planner | regime | warehouse, robots | best of b, c, d | its value | (e) on it % | (f) best of 3 on it % | (f) mean of 3 on it % | (b) on (a) % |", "|---|---|---|---|---|---|---|---|---|"]
    for q in PLANNERS:
        for r in REGIMES:
            for c in CELLS:
                g = gaps(c, r, q, key)
                if g:
                    lines.append(f"| {q} | {r.upper()} | {name(c)} | ({g['rival']}) | {g['best']:.0f} | {show(g['e'])} | {show(g['f'])} | {show(g['f3'])} | {show(g['b'])} |")
    return "\n".join(lines)


def table_types():
    lines = ["| planner | warehouse, robots | (e) with types on best rule % | (e) without types on best rule % | types alone: (e) on the control % |", "|---|---|---|---|---|"]
    for q in PLANNERS:
        for c in CELLS:
            g, x, y = gaps(c, "r2", q), load(c, "r2", q, "e"), load(c, "r2", q, "e0")
            if g and y:
                lines.append(f"| {q} | {name(c)} | {show(g['e'])} | {show(g['e0'])} | {show(paired(col(y, 'makespan'), col(x, 'makespan')))} |")
    return "\n".join(lines)


def table_costs():
    """What rotation and delays cost the blind arm, and whether both regimes lost the same."""
    lines = ["| planner | warehouse, robots | free flow with turns | R1 % free flow | R2 % free flow | own delay R1 | own delay R2 | slips R1 / R2 | stops R1 / R2 | s | q_bad |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for q in PLANNERS:
        for c in CELLS:
            one, two = load(c, "r1", q, "a"), load(c, "r2", q, "a")
            if not one or not two:
                continue
            cal = json.load(open(f"{OUT}/cal_{c}.json"))
            rate = lambda R, k: col(R, k).sum() / col(R, "moves").sum()
            pct = lambda R: 100 * np.mean(col(R, "makespan") / col(R, "free_flow"))
            lines.append(f"| {q} | {name(c)} | {col(one, 'free_flow').mean():.0f} | {pct(one):.0f} | {pct(two):.0f} | {rate(one, 'slips') + rate(one, 'stops'):.3f} | {rate(two, 'slips') + rate(two, 'stops'):.3f} | "
                         f"{rate(one, 'slips'):.3f} / {rate(two, 'slips'):.3f} | {rate(one, 'stops'):.3f} / {rate(two, 'stops'):.3f} | {cal['s']:.4f} | {cal['q_bad']:.4f} |")
    return "\n".join(lines)


def table_gamma():
    lines = ["| planner | regime | warehouse, robots | gamma |", "|---|---|---|---|"]
    for q in PLANNERS:
        for r in REGIMES:
            for c in CELLS:
                R = load(c, r, q, "d")
                if R:
                    lines.append(f"| {q} | {r.upper()} | {name(c)} | {col(R, 'gamma').mean():.2f} |")
    return "\n".join(lines)


def figure():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=130, sharey=True)
    order = sorted(CELLS, key=density)
    for ax, r in zip(axes, REGIMES):
        pts = [(density(c), gaps(c, r, "pibt")) for c in order]
        pts = [(x, g) for x, g in pts if g]
        x = np.array([p[0] for p in pts])
        ax.errorbar(x, [p[1]["e"][1] for p in pts], yerr=[p[1]["e"][2] for p in pts], marker="o", capsize=3, color="#2563eb", label="(e) function known")
        ax.errorbar(x * 1.03, [p[1]["f"][1] for p in pts], yerr=[p[1]["f"][2] for p in pts], marker="s", capsize=3, color="#059669", label="(f) hindsight, best of 3")
        ax.errorbar(x * 1.06, [p[1]["f3"][1] for p in pts], yerr=[p[1]["f3"][2] for p in pts], marker="^", capsize=3, color="#94a3b8", ls="--", label="(f) hindsight, mean of 3")
        ax.axhline(3, color="#ea580c", lw=0.9, ls=":", label="3 %")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(f"{r.upper()}: {'unpredictable' if r == 'r1' else 'from the state'}")
        ax.set_xlabel("robots per 100 free cells")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("makespan gain on best of b, c, d, %")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)


def time_spent():
    text = open(LOG).read()
    parse = lambda x: datetime.strptime(x.replace(" EST", ""), "%a %b %d %H:%M:%S %Y")
    marks = [(a.strip(), parse(b.strip())) for a, b in re.findall(r"^=== (.+?)  (\w{3} \w{3} +\d+ .+)$", text, flags=re.M)]
    last = re.search(r"^=== ALL DONE (.+)$", text, flags=re.M)
    if last:
        marks.append(("end", parse(last.group(1).strip())))
    seconds = sum(json.load(open(p))["elapsed"] for p in glob.glob(f"{OUT}/*_[ab-f]*_*.json") if "trips" not in p and "cal_" not in p)
    table = ["| stage (at most 4 processes) | wall clock |", "|---|---|"] + [f"| {a} | {y - x} |" for (a, x), (_, y) in zip(marks, marks[1:])]
    return f"Episodes of the report: {seconds / 3600:.1f} h of CPU.\n\n" + "\n".join(table)


def section(notes, name):
    parts = notes.split(f"## {name}\n", 1)
    return parts[1].split("\n## ", 1)[0].strip() if len(parts) == 2 else f"(notes missing: {name})"


def main():
    figure()
    notes = open("docs/phaseB0c_notes.md").read()
    out = ["# Warehouse, phase B0c: imperfect execution", "", "## Recommendation", "", section(notes, "Recommendation"), "",
           "## The column that decides", "", section(notes, "Decision"), "", table_gaps(), "", section(notes, "Decision delay"), "", table_gaps("mean_delay"), "",
           "![Gap against density](figures/phaseB0c_gap.png)", "", "## Rotation alone", "", section(notes, "Rotation"), "",
           "## Every arm", "", section(notes, "Arms"), "", table_arms(), "", "## What delays cost, and the two regimes", "", section(notes, "Regimes"), "", table_costs(), "", section(notes, "Types"), "", table_types(), "",
           "γ fitted per cell:", "", table_gamma(), "", "## Model and protocol", "", section(notes, "Protocol"), "",
           "## Budget and changes", "", section(notes, "Budget"), "", time_spent(), "", "## Limitations", "", section(notes, "Limitations"), ""]
    open("docs/phaseB0c_realism.md", "w").write("\n".join(out))
    print("saved docs/phaseB0c_realism.md")


if __name__ == "__main__":
    main()
