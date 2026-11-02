"""Report of phase B0b: docs/phaseB0b_margins.md and its figure, from results/warehouse/phaseB0b/.

The hand-written parts are read from docs/phaseB0b_notes.md (one "## " section per part of the report); every number and
table is computed here. A run that fails the verifier stops the report.

Example:
    python scripts/phaseB0b_report.py
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

OUT = "results/warehouse/phaseB0b"
FIG = "docs/figures/phaseB0b_density.png"
SEEDS = range(5)
ARMS_T2 = [("one", "1 order per trip, at once"), ("a", "(a) trips of 3, at once"), ("b20", "(b) wait 20"), ("b50", "(b) wait 50"),
           ("c100", "(c) known 100 steps ahead"), ("cinf", "(c) all arrivals known")]
LEARNED_FREE = ("base", "o1b", "o2b1", "o2b2")


def paired(base, x):
    """(% gain per seed of x against base, its mean, half of the 95 % interval, Wilcoxon p)."""
    base, x = np.asarray(base, dtype=np.float64), np.asarray(x, dtype=np.float64)
    g = 100 * (base - x) / base
    half = float(student.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g)))
    p = 1.0 if not np.any(base - x) else float(wilcoxon(base, x, method="exact" if np.all(base != x) else "auto").pvalue)
    return g, float(g.mean()), half, p


def cell(g):
    return f"{g[1]:+.1f} ± {g[2]:.1f} (p={g[3]:.3g})"


def load(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/{pattern}"))]


def table_t1():
    lines = ["| size | cap | walk C&W | walk LKH-3 CVRP | walk best | HiGHS bound over candidates | best gain on C&W % | makespan C&W | LKH-3 | best | best gain on C&W % | % free flow, C&W / best | ms/step, C&W / best |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, c in (("medium", 3), ("large", 3), ("medium", 4)):
        R = load(f"t1_{s}_{c}_*.json")
        f = lambda a, k: np.array([r[a][k] for r in R], dtype=np.float64)
        ff = lambda a: 100 * np.mean(f(a, "makespan") / f(a, "free_flow"))
        lines.append(f"| {s} | {c} | {f('cw', 'walk').mean():.0f} | {f('lkh', 'walk').mean():.0f} | {f('best', 'walk').mean():.0f} | {np.mean([r['bound'] for r in R]):.0f} | "
                     f"{cell(paired(f('cw', 'walk'), f('best', 'walk')))} | {f('cw', 'makespan').mean():.0f} | {f('lkh', 'makespan').mean():.0f} | {f('best', 'makespan').mean():.0f} | "
                     f"{cell(paired(f('cw', 'makespan'), f('best', 'makespan')))} | {ff('cw'):.0f} / {ff('best'):.0f} | {f('cw', 'step_ms').mean():.1f} / {f('best', 'step_ms').mean():.1f} |")
    return "\n".join(lines)


def table_t1all():
    lines = ["| seed | every trip of 3 | lower bound over all of them | walk C&W | walk best of T1 | C&W over the bound % | best of T1 over the bound % | seconds |", "|---|---|---|---|---|---|---|---|"]
    for r in load("t1all_*.json"):
        b = json.load(open(f"{OUT}/t1_medium_3_{r['seed']}.json"))["best"]["walk"]
        lines.append(f"| {r['seed']} | {r['candidates']} | {r['bound']:.0f} | {r['cw']:.0f} | {b:.0f} | {100 * (r['cw'] / r['bound'] - 1):.2f} | {100 * (b / r['bound'] - 1):.2f} | {r['elapsed']:.0f} |")
    return "\n".join(lines)


def t2_series(size, rate, arm, key):
    if arm in ("c100", "cinf"):
        names = sorted({r["arm"] for r in load(f"t2_{size}_{rate}_{arm}_*.json")})
        rows = {n: [json.load(open(f"{OUT}/t2_{size}_{rate}_{n}_{k}.json"))[key] for k in SEEDS] for n in names}
        sign = -1 if key.startswith("orders") else 1
        return np.array(min(rows.values(), key=lambda v: sign * np.mean(v)), dtype=np.float64)
    return np.array([json.load(open(f"{OUT}/t2_{size}_{rate}_{arm}_{k}.json"))[key] for k in SEEDS], dtype=np.float64)


def table_t2():
    lines = ["| size | rate | arm | orders/h | orders/h, second half | mean delay, open orders included | backlog, second half | ms/step |", "|---|---|---|---|---|---|---|---|"]
    for s in ("medium", "large"):
        for rate in ("medium", "x2.0"):
            for a, n in ARMS_T2:
                lines.append(f"| {s} | {rate} | {n} | {t2_series(s, rate, a, 'orders_per_hour').mean():.0f} | {t2_series(s, rate, a, 'orders_per_hour_late').mean():.0f} | "
                             f"{t2_series(s, rate, a, 'delay_all').mean():.0f} | {t2_series(s, rate, a, 'backlog').mean():.0f} | {t2_series(s, rate, a, 'step_ms').mean():.1f} |")
    return "\n".join(lines)


def table_t2_value():
    lines = ["| size | rate | metric | best of (a, b) | best oracle (c) | value of the arrivals % |", "|---|---|---|---|---|---|"]
    for s in ("medium", "large"):
        for rate in ("medium", "x2.0"):
            for key, name, up in (("orders_per_hour", "orders/h", True), ("delay_all", "delay", False), ("backlog", "backlog", False)):
                pick = max if up else min
                on = pick((t2_series(s, rate, a, key) for a in ("a", "b20", "b50")), key=np.mean)
                off = pick((t2_series(s, rate, a, key) for a in ("c100", "cinf")), key=np.mean)
                g = paired(on, off)
                g = (g[0], -g[1], g[2], g[3]) if up else g
                lines.append(f"| {s} | {rate} | {name} | {on.mean():.0f} | {off.mean():.0f} | {cell(g)} |")
    return "\n".join(lines)


def t3_rows():
    R = [json.load(open(p)) for p in glob.glob(f"{OUT}/t3_*.json")]
    for r in R:
        assert r["ok"], r
    xs = sorted({r["x"] for r in R})
    return {x: {a: {r["seed"]: r for r in R if r["x"] == x and r["arm"] == a} for a in (*LEARNED_FREE, "o2b3")} for x in xs}


def density(row):
    """(free-flow % of the baseline, seeds, best not learned, gain of hindsight on it, gain on the Hungarian) per density."""
    seeds = sorted(row["o2b3"])
    mk = lambda a: np.array([row[a][k]["makespan"] for k in seeds], dtype=np.float64)
    ff = 100 * np.mean(mk("base") / np.array([row["base"][k]["free_flow"] for k in seeds]))
    rival = min(LEARNED_FREE, key=lambda a: mk(a).mean())
    return ff, seeds, rival, paired(mk(rival), mk("o2b3")), paired(mk("o1b"), mk("o2b3")), mk


def table_t3(rows):
    lines = ["| times robots | robots | orders | baseline, % free flow | steps under PIBT % | baseline | Hungarian | static map | map x gamma | hindsight | best not learned | hindsight gain on it % | gain on Hungarian % | seeds |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x, row in rows.items():
        ff, seeds, rival, g, h, mk = density(row)
        one = row["base"][seeds[0]]
        pibt = 100 * np.mean([row["base"][k]["fallbacks"] / row["base"][k]["calls"] for k in seeds])
        lines.append(f"| {x:g} | {one['robots']} | {one['orders']} | {ff:.0f} | {pibt:.0f} | " + " | ".join(f"{mk(a).mean():.0f}" for a in (*LEARNED_FREE, "o2b3")) + f" | {rival} | {cell(g)} | {cell(h)} | {len(seeds)} |")
    return "\n".join(lines)


def figure(rows):
    pts = [density(r) for r in rows.values()]
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=130)
    x = [p[0] for p in pts]
    ax.errorbar(x, [p[3][1] for p in pts], yerr=[p[3][2] for p in pts], marker="o", capsize=3, color="#2563eb", label="hindsight on the best rule not learned")
    ax.errorbar(x, [p[4][1] for p in pts], yerr=[p[4][2] for p in pts], marker="s", capsize=3, color="#94a3b8", ls="--", label="hindsight on the Hungarian")
    ax.axhline(3, color="#ea580c", lw=0.9, ls=":", label="3 %")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("baseline makespan, % of free flow")
    ax.set_ylabel("makespan gain %, with 95 % interval")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)


def time_spent():
    text = open("results/warehouse/phaseB0b.log").read()
    parse = lambda x: datetime.strptime(x.replace(" EST", ""), "%a %b %d %H:%M:%S %Y")
    marks = [(a.strip(), parse(b.strip())) for a, b in re.findall(r"^=== (.+?)  (\w{3} \w{3} +\d+ .+)$", text, flags=re.M)]
    marks.append(("end", parse(re.search(r"^=== ALL DONE (.+)$", text, flags=re.M).group(1).strip())))
    return "\n".join(["| stage (at most 4 processes) | wall clock |", "|---|---|"] + [f"| {a} | {y - x} |" for (a, x), (_, y) in zip(marks, marks[1:])])


def section(notes, name):
    parts = notes.split(f"## {name}\n", 1)
    return parts[1].split("\n## ", 1)[0].strip() if len(parts) == 2 else f"(notes missing: {name})"


def main():
    rows = t3_rows()
    figure(rows)
    notes = open("docs/phaseB0b_notes.md").read()
    out = ["# Warehouse, phase B0b: capacity, arrivals, density", "", "## Recommendation", "", section(notes, "Recommendation"), "",
           "## T1: capacity", "", section(notes, "T1"), "", table_t1(), "", section(notes, "T1 check"), "", table_t1all(), "",
           "## T2: online against clairvoyant", "", section(notes, "T2"), "", table_t2(), "", table_t2_value(), "",
           "## T3: density", "", section(notes, "T3"), "", table_t3(rows), "", "![Gap against free flow](figures/phaseB0b_density.png)", "",
           "## Budget and changes", "", section(notes, "Budget"), "", time_spent(), "", "## Limitations", "", section(notes, "Limitations"), ""]
    open("docs/phaseB0b_margins.md", "w").write("\n".join(out))
    print("saved docs/phaseB0b_margins.md")


if __name__ == "__main__":
    main()
