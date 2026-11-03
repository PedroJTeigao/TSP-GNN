"""Ablation A3 (2026-11-03), warehouse part: the "no translation" baseline of phase C (table C).

Every test request runs once with the EMPTY set of accepted rules: the same world, seed and planner as the arms
(phaseC_language.c_job), judged against the true rules. Its episode goes to the cache of stage c under the key of the
empty set, results/warehouse/phaseC/c/<idx>_<key>.json, where 8 of them already are (arms whose accepted set was empty).
The table then adds, per level and arm, the baseline on the same requests, the validity above it (pp) and the exact
McNemar test on the discordant requests (valid = 0 true rules broken, as in table C).

    python scripts/ablation_a3_warehouse.py run --workers 4
    python scripts/ablation_a3_warehouse.py table          # results/ablations/a3_warehouse.{json,md}
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phaseC_language as Q
from warehouse import constraints as C
from warehouse import dsl as L
from warehouse import episode as P
from warehouse import world as Y

DEST = "../tsp/results/ablations"


def empty_key(ctx):
    return Q.accepted({"answer": {"rules": []}}, ctx)[3]


def job(r):
    w = Y.make(r["seed"])
    ctx = L.Context(w)
    key = empty_key(ctx)
    out = f"{Q.OUT}/c/{r['idx']}_{key}.json"
    if os.path.exists(out):
        return out, False
    truth = [p for x in Q.labels(r, ctx) for p in x["prims"]]
    t0 = time.perf_counter()
    rep, _ = P.run(w, [], C.to_checks(w, truth), r["seed"], P.load_memo(Q.memo_path(r["idx"])))
    rep["elapsed"] = time.perf_counter() - t0
    assert rep["structural"], (r["idx"], rep["problems"])
    tmp = out + ".tmp"
    json.dump(rep, open(tmp, "w"))
    os.replace(tmp, out)
    return out, True


def run(workers):
    reqs = Q.requests("test")
    start = time.perf_counter()
    new = 0
    with Pool(workers) as pool:
        for k, (_, ran) in enumerate(pool.imap_unordered(job, reqs), 1):
            new += ran
            if k % 20 == 0:
                print(f"{k}/{len(reqs)} ({new} run, {time.perf_counter() - start:.0f} s)", flush=True)
    print(f"ALL DONE a3 warehouse run ({new} new episodes)")


def wilson(k, m, z=1.96):
    p = k / m
    d = 1 + z * z / m
    c = (p + z * z / (2 * m)) / d
    h = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
    return 100 * (c - h), 100 * (c + h)


def compare(rows, base, skip=frozenset()):
    import phaseC_report as REP
    out = []
    for lv in Q.LEVELS:
        for arm in REP.ARMS[1:]:
            s = [r for r in REP.pick(rows, lv, arm, skip) if r["ep"] is not None]
            if not s:
                continue
            a = np.array([r["ep"]["broken"] == 0 for r in s])
            b = np.array([base[r["idx"]]["broken"] == 0 for r in s])
            x, y = int((a & ~b).sum()), int((b & ~a).sum())
            mk = [100 * (base[r["idx"]]["makespan"] - r["oracle"]["makespan"]) / r["oracle"]["makespan"] for r in s]
            out.append({"level": lv, "arm": arm, "episodes": len(s), "valid": int(a.sum()), "baseline": int(b.sum()),
                        "above_pp": 100 * (a.mean() - b.mean()), "arm_only": x, "baseline_only": y,
                        "p": binomtest(x, x + y, 0.5).pvalue if x + y else 1.0, "baseline_makespan_vs_oracle": float(np.mean(mk))})
    return out


def table(rows):
    lines = ["| level | arm | episodes | no translation: 0 violations [Wilson 95 %] | arm: 0 violations [Wilson 95 %] | "
             "validity above the baseline (pp) | arm valid, baseline not | baseline valid, arm not | exact McNemar p |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        m = r["episodes"]
        bl, bh = wilson(r["baseline"], m)
        al, ah = wilson(r["valid"], m)
        lines.append(f"| {r['level']} | {r['arm']} | {m} | {r['baseline']}/{m} = {100 * r['baseline'] / m:.1f} % [{bl:.1f}, {bh:.1f}] | "
                     f"{r['valid']}/{m} = {100 * r['valid'] / m:.1f} % [{al:.1f}, {ah:.1f}] | {r['above_pp']:+.1f} | {r['arm_only']} | "
                     f"{r['baseline_only']} | {r['p']:.3g} |")
    return "\n".join(lines)


def make_table():
    import phaseC_report as REP
    rows = REP.load()
    base = {}
    for r in Q.requests("test"):
        ep = REP.episode(r["idx"], empty_key(L.Context(Y.make(r["seed"]))))
        assert ep is not None, f"request {r['idx']}: no empty-set episode, run the 'run' stage first"
        base[r["idx"]] = ep
    out = {"all": compare(rows, base), "without_audited_errors": compare(rows, base, REP.excluded()),
           "baseline_valid": sum(e["broken"] == 0 for e in base.values()), "requests": len(base),
           "baseline_broken_per_episode": float(np.mean([e["broken"] for e in base.values()]))}
    os.makedirs(DEST, exist_ok=True)
    json.dump(out, open(f"{DEST}/a3_warehouse.json", "w"), indent=1)
    md = ["All texts:", "", table(out["all"]), "", "Without the L2 texts the audit found wrong:", "", table(out["without_audited_errors"])]
    open(f"{DEST}/a3_warehouse.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("run", "table"))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    run(args.workers) if args.what == "run" else make_table()


if __name__ == "__main__":
    main()
