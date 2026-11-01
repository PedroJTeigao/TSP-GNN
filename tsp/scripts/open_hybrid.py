"""Phase 5, step 3: the hybrid arms h1 and h2 (tsp_gnn.open_hybrid), offline, from the logs of open_formalize.py and
open_pipeline.py. No model and no LKH-3 run here.

Routes: h1 takes the route of the arm it copies. h2 reuses the reference if its rule set equals the label, the pure route if
no rule is left, and a logged solve if one has the same rule set; the other h2 sets would need a new LKH-3 run, so they are
only counted, and the rate is shown over the routes that exist with the bounds if all the others failed or held.

Writes results/phase5/open_language/hybrid.json (one record per text) and hybrid_tables.md (read by the report).

Example:
    python scripts/open_hybrid.py
"""

import importlib.util
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from tsp_gnn import open_dsl as D
from tsp_gnn import open_eval as E
from tsp_gnn import open_hybrid as H
from tsp_gnn import rules as R

OUT = "results/phase5/open_language"
NS = (200, 1000)
ARMS = ("p1", "q14", "h1", "h2")
_here = os.path.dirname(os.path.abspath(__file__))


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REPORT = load_module("phase5_open_language_report")
PIPE = load_module("open_pipeline")


def records():
    out = []
    for n in NS:
        logs = {(e["idx"], e["level"], e["arm"]): e for e in REPORT.jl(f"{OUT}/formalize_tsp{n}.jsonl")}
        routes = {(e["idx"], e["level"], e["arm"]): e for e in REPORT.jl(f"{OUT}/pipeline_tsp{n}.jsonl")}
        solves = {(e["idx"], e["key"]): e for e in REPORT.jl(f"{OUT}/pipeline_solves_tsp{n}.jsonl")}
        for lv in E.LEVELS:
            for inst in E.load_level(n, lv):
                idx = inst["idx"]
                p, q = logs.get((idx, lv, "p1")), logs.get((idx, lv, "q14"))
                if p is None or q is None:
                    continue
                cov = H.coverage(inst, n)
                label = E.label_prims(inst)
                sp = D.score(inst["label"], [x for r in p["pred"] for x in r["prims"]])
                sq = D.score(inst["label"], [x for r in q["pred"] for x in r["prims"]])
                prims, rejected = H.union(p["pred"], q["pred"], inst["coords"])
                su = D.score(inst["label"], prims)
                rec = {"n": n, "level": lv, "idx": idx, **cov, "q14_seconds": q["seconds"], "rejected_h2": len(rejected),
                       "correct": {"p1": sp["correct"], "q14": sq["correct"], "h1": (sq if cov["send"] else sp)["correct"],
                                   "h2": su["correct"]},
                       "exact": {"p1": sp["exact"], "q14": sq["exact"], "h1": (sq if cov["send"] else sp)["exact"],
                                 "h2": su["exact"]}, "valid": {}, "lkh": {}}
                rp, rq = routes.get((idx, lv, "p1")), routes.get((idx, lv, "q14"))
                if rp is not None and rq is not None:
                    pick = rq if cov["send"] else rp
                    for arm, r in (("p1", rp), ("q14", rq), ("h1", pick)):
                        rec["valid"][arm], rec["lkh"][arm] = r["valid"], r["lkh_seconds"]
                    key = PIPE.set_key(prims)
                    if key == PIPE.set_key(label):
                        rec["valid"]["h2"], rec["lkh"]["h2"] = True, 0.0
                    elif not prims:
                        rec["valid"]["h2"], rec["lkh"]["h2"] = bool(R.check_all(label, inst["coords"], inst["pure"])), 0.0
                    elif (idx, key) in solves:
                        s = solves[(idx, key)]
                        rec["valid"]["h2"] = bool(R.check_all(label, inst["coords"], np.array(s["tour"])))
                        rec["lkh"]["h2"] = s["seconds"]
                    else:
                        rec["valid"]["h2"] = None        # would need a new LKH-3 run
                out.append(rec)
    return out


def table(rows, excluded=frozenset(), levels=E.LEVELS):
    alone = REPORT.alone()
    lines = ["| n | level | arm | texts | rules exact % [95 % CI] | instances exact % [95 % CI] | valid routes % [95 % CI] (routes) | "
             "sent to the LLM % | LLM s, alone | LLM s, run | LLM + LKH s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in NS:
        for lv in levels:
            rs = [r for r in rows if r["n"] == n and r["level"] == lv and (n, r["idx"]) not in (excluded if lv == "L2" else ())]
            lat = float(np.mean(alone[(n, lv, "q14")]))
            for arm in ARMS:
                share = {"p1": 0.0, "q14": 1.0, "h2": 1.0, "h1": float(np.mean([r["send"] for r in rs]))}[arm]
                k = sum(sum(r["correct"][arm]) for r in rs)
                m = sum(len(r["correct"][arm]) for r in rs)
                ex = sum(r["exact"][arm] for r in rs)
                have = [r for r in rs if arm in r["valid"]]
                known = [r for r in have if r["valid"][arm] is not None]
                ok = sum(r["valid"][arm] for r in known)
                cell = f"{REPORT.pc(ok, len(known))} ({len(known)})"
                if len(known) < len(have):
                    cell += f"; bounds {100 * ok / len(have):.1f}–{100 * (ok + len(have) - len(known)) / len(have):.1f} over {len(have)}"
                run = 0.0 if arm == "p1" else float(np.mean([r["q14_seconds"] * (arm != "h1" or r["send"]) for r in rs]))
                lkh = float(np.mean([r["lkh"][arm] for r in known]))
                lines.append(f"| {n} | {lv} | {arm} | {len(rs)} | {REPORT.pc(k, m)} | {REPORT.pc(ex, len(rs))} | {cell} | "
                             f"{100 * share:.1f} | {share * lat:.1f} | {run:.1f} | {share * lat + lkh:.1f} |")
    return "\n".join(lines)


def mcnemar(rows, a, b, excluded=frozenset(), levels=E.LEVELS):
    lines = [f"| n | level | rules: {a} only / {b} only | p | routes: {a} only / {b} only | p |", "|---|---|---|---|---|---|"]
    for n in NS:
        for lv in levels:
            rs = [r for r in rows if r["n"] == n and r["level"] == lv and (n, r["idx"]) not in (excluded if lv == "L2" else ())]
            x = sum(u and not v for r in rs for u, v in zip(r["correct"][a], r["correct"][b]))
            y = sum(v and not u for r in rs for u, v in zip(r["correct"][a], r["correct"][b]))
            routed = [r for r in rs if a in r["valid"] and b in r["valid"]]
            s = sum(r["valid"][a] and not r["valid"][b] for r in routed)
            t = sum(r["valid"][b] and not r["valid"][a] for r in routed)
            px = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
            ps = binomtest(s, s + t, 0.5).pvalue if s + t else 1.0
            lines.append(f"| {n} | {lv} | {x} / {y} | {px:.3g} | {s} / {t} | {ps:.3g} |")
    return "\n".join(lines)


def trigger(rows):
    lines = ["| n | level | texts | sent | p1 exact among the sent | q14 exact among the sent | p1 exact among the kept | p1 wrong and kept |",
             "|---|---|---|---|---|---|---|---|"]
    for n in NS:
        for lv in E.LEVELS:
            rs = [r for r in rows if r["n"] == n and r["level"] == lv]
            sent = [r for r in rs if r["send"]]
            kept = [r for r in rs if not r["send"]]
            lines.append(f"| {n} | {lv} | {len(rs)} | {len(sent)} | {sum(r['exact']['p1'] for r in sent)} | "
                         f"{sum(r['exact']['q14'] for r in sent)} | {sum(r['exact']['p1'] for r in kept)} | "
                         f"{sum(not r['exact']['p1'] for r in kept)} |")
    return "\n".join(lines)


def main():
    rows = records()
    excluded = E.audit_excluded()
    json.dump(rows, open(f"{OUT}/hybrid.json", "w"))
    missing = defaultdict(int)
    for r in rows:
        if r["valid"].get("h2", 0) is None:
            missing[(r["n"], r["level"])] += 1
    miss = "; ".join(f"n = {n} {lv}: {v}" for (n, lv), v in sorted(missing.items())) or "none"
    txt = ["All texts:", "", table(rows), "", "Without the 17 L2 texts the audit found wrong (L2 only):", "",
           table(rows, excluded, ("L2",)), "", "McNemar, h1 against q14:", "", mcnemar(rows, "h1", "q14"), "",
           "McNemar, h1 against p1:", "", mcnemar(rows, "h1", "p1"), "",
           "McNemar without the 17 texts, L2, h1 against q14:", "", mcnemar(rows, "h1", "q14", excluded, ("L2",)), "",
           "McNemar without the 17 texts, L2, h1 against p1:", "", mcnemar(rows, "h1", "p1", excluded, ("L2",)), "",
           "What the trigger did (exact instances):", "", trigger(rows), "",
           f"h2 rule sets with no route in the logs, which would need a new LKH-3 run (not run): {sum(missing.values())} ({miss})."]
    open(f"{OUT}/hybrid_tables.md", "w").write("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
