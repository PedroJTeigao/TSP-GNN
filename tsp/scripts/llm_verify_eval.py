"""Phase 5, step 2, part D: the local 7B as a route verifier (tsp_gnn.llm_verify), n = 200 only.

At n = 1000 a route written as visits with times is ~10k tokens and does not fit the 8192-token context, so D is
run at n = 200 only (recorded in the report).

Routes (text_v2b instances that are complete and have no label error): the oracle (reference) route, which breaks no
rule, and the same route with one injected error of a type drawn among the instance's hard rule types (verified in
code to break exactly that rule). Each route is shown to the 7B twice: plain (text + route) and with the code
checker's report. The truth is the code checker's verdict for every hard rule of the label.

Writes data/phase5/llm_verify_tsp200.jsonl (one record per route and variant: truth, the LLM's claims, latency, raw
answer); a rerun skips the records already there.

Example:
    python scripts/llm_verify_eval.py
"""

import argparse
import json
import os
import time

import numpy as np

from tsp_gnn import llm
from tsp_gnn import llm_verify as V
from tsp_gnn import rules as R
from tsp_gnn.utils import git_info

SEED = 20261029


def main():
    parser = argparse.ArgumentParser(description="The local 7B as a verifier of routes against text rules.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="default data/phase5/llm_verify_tsp<n>.jsonl")
    args = parser.parse_args()
    n = args.n
    out = args.out or f"data/phase5/llm_verify_tsp{n}.jsonl"
    rows = [json.loads(line) for line in open(f"data/phase5/text_rules_v2b_tsp{n}.jsonl")]
    errors_path = "data/phase5/text_v2b_label_errors.json"
    excluded = set(json.load(open(errors_path)).get(str(n), [])) if os.path.exists(errors_path) else set()
    rows = [r for r in rows if r["v2_complete"] and r["idx"] not in excluded]
    if args.limit:
        rows = rows[:args.limit]
    arr = np.load(f"data/phase5/text_rules_tsp{n}.npz")
    done = set()
    if os.path.exists(out):
        done = {(e["idx"], e["route"], e["variant"]) for e in map(json.loads, open(out))}
    commit, dirty = git_info()
    rng = np.random.default_rng([SEED, n])
    start, calls = time.perf_counter(), 0
    with open(out, "a") as f:
        for row in rows:
            idx = row["idx"]
            coords, ref = arr["coords"][idx], arr["tours"][idx]
            label = [{"type": r["type"], "formal": r["formal"]} for r in row["rules"]]
            routes = [("oracle", ref, None, None)]
            kinds = sorted({r["type"] for r in label} & set(V.INJECTABLE))
            if kinds:
                kind = str(rng.choice(kinds))
                inj = V.inject_error(rng, coords, ref, label, kind)
                if inj is not None:
                    routes.append(("injected", inj[0], kind, inj[1]))
            for name, tour, kind, k in routes:
                truth = R.violated_rules(label, coords, tour)
                assert truth == ([] if k is None else [k])
                route = V.route_text(coords, tour, label)
                for variant in ("plain", "report"):
                    if (idx, name, variant) in done:
                        continue
                    report = V.code_report(label, coords, tour) if variant == "report" else None
                    system = V.SYSTEM if variant == "plain" else V.SYSTEM_REPORT
                    content, sec, raw = llm.chat(system, V.user_prompt(row["text_v2"], route, report),
                                                 temperature=0.0, seed=0, num_predict=512)
                    calls += 1
                    obj = V.parse(content)
                    claimed, unmatched = V.match_claims(obj, label)
                    f.write(json.dumps({
                        "idx": idx, "n": n, "route": name, "variant": variant, "injected_type": kind,
                        "rules": [{"type": r["type"], "binding": r.get("binding", True)} for r in row["rules"]],
                        "truth": truth, "claimed": sorted(claimed) if claimed is not None else None,
                        "unmatched_claims": unmatched, "valid_json": obj is not None, "seconds": sec,
                        "prompt_tokens": raw.get("prompt_eval_count"), "raw": content, "git_commit": commit}) + "\n")
                    f.flush()
            if calls and calls % 40 == 0:
                print(f"  {calls} LLM calls  ({time.perf_counter() - start:.0f} s)", flush=True)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  {calls} calls  |  saved to {out}")


if __name__ == "__main__":
    main()
