"""Ablation A3 (2026-11-03), TSP part: the "no translation" baseline of phase 5, step 3 (table of step 2C).

With an empty rule set the pipeline returns the pure route (scripts/open_pipeline.py: "nothing left after the check").
So the baseline needs no solver: for every instance the pure route is judged against the label with rules.check_all,
exactly as the arms' routes were. Each arm is then compared with the baseline on the same instances: valid routes,
validity above the baseline (pp) and the exact McNemar test on the discordant instances. Arms: the step 2C rows of
results/phase5/open_language/pipeline_tsp<n>.jsonl and the hybrids h1 / h2 of hybrid.json (h2 where its route is known).

Writes results/ablations/a3_tsp.json and a3_tsp.md (all texts, then without the L2 texts the audit found wrong).

    python scripts/ablation_a3_tsp.py
"""

import json
import os

import numpy as np
from scipy.stats import binomtest

from tsp_gnn import open_eval as E
from tsp_gnn import rules as R

OUT = "results/phase5/open_language"
DEST = "results/ablations"
NS = (200, 1000)
ARMS = ("oracle", "p0", "p1", "q7", "q14", "q14t", "g12")


def wilson(k, m, z=1.96):
    p = k / m
    d = 1 + z * z / m
    c = (p + z * z / (2 * m)) / d
    h = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
    return 100 * (c - h), 100 * (c + h)


def baseline():
    """{(n, level, idx): pure route valid against the label}."""
    out = {}
    for n in NS:
        for lv in E.LEVELS:
            for inst in E.load_level(n, lv):
                out[(n, lv, inst["idx"])] = bool(R.check_all(E.label_prims(inst), inst["coords"], inst["pure"]))
    return out


def arm_routes():
    """{(n, level, arm): {idx: valid}} from the pipeline records and the hybrids."""
    out = {}
    for n in NS:
        for r in map(json.loads, open(f"{OUT}/pipeline_tsp{n}.jsonl")):
            out.setdefault((n, r["level"], r["arm"]), {})[r["idx"]] = bool(r["valid"])
    for r in json.load(open(f"{OUT}/hybrid.json")):
        for arm in ("h1", "h2"):
            if r["valid"].get(arm) is not None:
                out.setdefault((r["n"], r["level"], arm), {})[r["idx"]] = bool(r["valid"][arm])
    return out


def compare(base, routes, excluded=frozenset()):
    rows = []
    for (n, lv, arm), valid in sorted(routes.items(), key=lambda x: (x[0][0], E.LEVELS.index(x[0][1]),
                                                                    (ARMS + ("h1", "h2")).index(x[0][2]))):
        idx = [i for i in valid if (n, lv, i) in base and not (lv == "L2" and (n, i) in excluded)]
        if not idx:
            continue
        a = np.array([valid[i] for i in idx])
        b = np.array([base[(n, lv, i)] for i in idx])
        x, y = int((a & ~b).sum()), int((b & ~a).sum())
        rows.append({"n": n, "level": lv, "arm": arm, "routes": len(idx), "valid": int(a.sum()), "baseline": int(b.sum()),
                     "above_pp": 100 * (a.mean() - b.mean()), "arm_only": x, "baseline_only": y,
                     "p": binomtest(x, x + y, 0.5).pvalue if x + y else 1.0})
    return rows


def table(rows):
    lines = ["| n | level | arm | routes | no translation: valid % [95 % CI] | arm: valid % [95 % CI] | validity above "
             "the baseline (pp) | arm valid, baseline not | baseline valid, arm not | exact McNemar p |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["routes"]
        bl, bh = wilson(r["baseline"], m)
        al, ah = wilson(r["valid"], m)
        lines.append(f"| {r['n']} | {r['level']} | {r['arm']} | {m} | {100 * r['baseline'] / m:.1f} [{bl:.1f}, {bh:.1f}] | "
                     f"{100 * r['valid'] / m:.1f} [{al:.1f}, {ah:.1f}] | {r['above_pp']:+.1f} | {r['arm_only']} | "
                     f"{r['baseline_only']} | {r['p']:.3g} |")
    return "\n".join(lines)


def main():
    base = baseline()
    routes = arm_routes()
    excluded = E.audit_excluded()
    out = {"all": compare(base, routes), "without_audited_errors": compare(base, routes, excluded),
           "baseline_by_level": {f"{n} {lv}": [sum(v for (m, l, _), v in base.items() if (m, l) == (n, lv)),
                                              sum(1 for (m, l, _) in base if (m, l) == (n, lv))]
                                 for n in NS for lv in E.LEVELS}}
    os.makedirs(DEST, exist_ok=True)
    json.dump(out, open(f"{DEST}/a3_tsp.json", "w"), indent=1)
    md = ["All texts:", "", table(out["all"]), "", "Without the L2 texts the audit found wrong:", "",
          table(out["without_audited_errors"])]
    open(f"{DEST}/a3_tsp.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
