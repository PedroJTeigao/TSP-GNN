"""Ablation A5 (2026-11-03): the three warehouse comparisons that decide, with 15 seeds instead of 5.

    t1   capacity (B0b T1, medium, capacity 3): Clarke & Wright against set partitioning (walking and makespan)
    t3   density (B0b T3, 3 x the robots = 90): hindsight per edge (o2b3, best of 3 runs) against the static map x gamma (o2b2)
    b0c  imperfect execution (B0c, regime R1, blind arm a): prioritized under action dependencies against PIBT, 30 and 90 robots

Seeds 0-4 run again, into a folder of their own, with the code of today: that is the reproduction check against the stored
results of B0b and B0c. Seeds 5-14 are new. The map and gamma of o2b2 come from the other seeds among 0-4, as in B0b (for a new
seed that is all five); hindsight and the B0c runs use their own seed only. Outputs in results/warehouse/ablation_a5/; a finished
run is skipped.

    python scripts/ablation_a5.py run --workers 4
    python scripts/ablation_a5.py table        # ../tsp/results/ablations/a5.{json,md}
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import t as student
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phaseB0b_margins as B0B
import phaseB0c_realism as B0C

OLD_B0B, OLD_B0C = B0B.OUT, B0C.OUT            # the stored results; the edge maps of o2b2 stay read from OLD_B0B (phaseB0_margins.OUT)
OUT = "results/warehouse/ablation_a5"
B0B.OUT, B0C.OUT = f"{OUT}/b0b", f"{OUT}/b0c"
SEEDS = range(15)
OLD = range(5)
DEST = "../tsp/results/ablations"


def job(j):
    kind, args = j
    if kind == "t1":
        return B0B.t1(("medium", 3, args))
    if kind == "t3":
        return B0B.t3((3.0, *args))
    if kind == "trips":
        return B0C.trips_job(args)
    return B0C.run(args)


def run(workers):
    os.makedirs(B0B.OUT, exist_ok=True)
    os.makedirs(B0C.OUT, exist_ok=True)
    start = time.perf_counter()
    with Pool(workers) as pool:
        trips = [("trips", (c, "plain", s)) for c in ("m30", "m90") for s in SEEDS]
        list(pool.imap_unordered(job, trips))
        # the longest first: hindsight (3 runs + 1), then partition, then the B0c episodes
        jobs = [("t3", ("o2b3", s)) for s in SEEDS] + [("t1", s) for s in SEEDS] + [("t3", ("o2b2", s)) for s in SEEDS] + \
               [("b0c", (c, "r1", q, "a", s)) for c in ("m90", "m30") for q in B0C.PLANNERS for s in SEEDS]
        for k, r in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"{time.strftime('%H:%M:%S')} {k}/{len(jobs)} {r} ({time.perf_counter() - start:.0f} s)", flush=True)
    print("ALL DONE a5 run")


def load(path):
    return json.load(open(path))


def values():
    """{comparison: {seed: (rival, ours)}} in the unit of each comparison, both new (all 15 seeds) and stored (0-4)."""
    out = {"new": {}, "old": {}}
    for tag, b, c, seeds in (("new", B0B.OUT, B0C.OUT, SEEDS), ("old", OLD_B0B, OLD_B0C, OLD)):
        d = out[tag]
        for s in seeds:
            t1 = load(f"{b}/t1_medium_3_{s}.json")
            d.setdefault("t1 walk", {})[s] = (t1["cw"]["walk"], t1["best"]["walk"])
            d.setdefault("t1 makespan", {})[s] = (t1["cw"]["makespan"], t1["best"]["makespan"])
            d.setdefault("t3 makespan", {})[s] = (load(f"{b}/t3_3.0_o2b2_{s}.json")["makespan"], load(f"{b}/t3_3.0_o2b3_{s}.json")["makespan"])
            for cell in ("m30", "m90"):
                pibt, prio = (load(f"{c}/{cell}_r1_{q}_a_{s}.json") for q in ("pibt", "prioritized"))
                d.setdefault(f"b0c {cell} makespan", {})[s] = (pibt["makespan"], prio["makespan"])
                d.setdefault(f"b0c {cell} delay", {})[s] = (pibt["mean_delay"], prio["mean_delay"])
    return out


def gain_stats(pairs):
    """Gain of the second over the first in % of the first (positive = the second is better), mean ± 95 % t interval, exact
    Wilcoxon signed-rank p on the paired values."""
    g = np.array([100 * (a - b) / a for a, b in pairs])
    half = student.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g))
    diff = np.array([a - b for a, b in pairs])
    p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
    return {"n": len(g), "mean": float(g.mean()), "lo": float(g.mean() - half), "hi": float(g.mean() + half), "p": float(p),
            "median": float(np.median(g)), "positive": int((g > 0).sum())}


LABELS = {"t1 walk": ("T1 medium, cap. 3: walking", "Clarke & Wright", "set partitioning"),
          "t1 makespan": ("T1 medium, cap. 3: makespan", "Clarke & Wright", "set partitioning"),
          "t3 makespan": ("T3, 90 robots: makespan", "map x gamma (o2b2)", "hindsight (o2b3)"),
          "b0c m30 makespan": ("B0c R1, 30 robots: makespan", "PIBT", "prioritized + dependencies"),
          "b0c m90 makespan": ("B0c R1, 90 robots: makespan", "PIBT", "prioritized + dependencies"),
          "b0c m30 delay": ("B0c R1, 30 robots: mean delay", "PIBT", "prioritized + dependencies"),
          "b0c m90 delay": ("B0c R1, 90 robots: mean delay", "PIBT", "prioritized + dependencies")}


def make_table():
    v = values()
    repro = {k: [s for s in OLD if v["new"][k][s] != v["old"][k][s]] for k in v["old"]}
    rows = []
    for k, (name, rival, ours) in LABELS.items():
        old, all15 = gain_stats([v["old"][k][s] for s in OLD]), gain_stats([v["new"][k][s] for s in SEEDS])
        rows.append({"key": k, "name": name, "rival": rival, "ours": ours, "old5": old, "new15": all15, "not_reproduced": repro[k]})
    os.makedirs(DEST, exist_ok=True)
    json.dump({"rows": rows, "values": {t: {k: {str(s): x for s, x in d.items()} for k, d in v[t].items()} for t in v}},
              open(f"{DEST}/a5.json", "w"), indent=1)
    fmt = lambda r: f"{r['mean']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}]"
    lines = ["| comparison | second against first | 5 seeds, stored: gain % [95 % CI] | Wilcoxon p | 15 seeds: gain % [95 % CI] | "
             "Wilcoxon p | seeds where the second wins | seeds 0-4 not reproduced |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['ours']} vs {r['rival']} | {fmt(r['old5'])} | {r['old5']['p']:.3g} | {fmt(r['new15'])} | "
                     f"{r['new15']['p']:.3g} | {r['new15']['positive']}/15 | {', '.join(map(str, r['not_reproduced'])) or 'none'} |")
    md = "\n".join(lines)
    open(f"{DEST}/a5.md", "w").write(md + "\n")
    print(md)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("run", "table"))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    run(args.workers) if args.what == "run" else make_table()


if __name__ == "__main__":
    main()
