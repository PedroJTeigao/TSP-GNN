"""Phase 5, step 3, part 3: qwen3:14b on the older tasks (only the new column; the 7B numbers are read from the old runs).

    planner   V0 and V3 of the planner probes (8 cases x 5 repeats = 40 calls each), same prompts and scoring;
    formalise arm b2 of step 2 (few-shot, primitive rules) on text_v2b, instances drawn until --rules rules at each n;
    verifier  part D of step 2: --routes routes drawn from the old run, each plain and with the code report.
The 7B figures come from the same subsets of the old records, so the two columns are paired.

Writes results/phase5/open_language/step3_<part>.jsonl; a rerun skips what is already there.

Example:
    python scripts/open_step3.py --think off
"""

import argparse
import importlib.util
import json
import os

import numpy as np

from tsp_gnn import formalize as F
from tsp_gnn import llm
from tsp_gnn import llm_verify as V
from tsp_gnn import rules as R
from tsp_gnn.utils import git_info

OUT = "results/phase5/open_language"
MODEL, MIB = "qwen3:14b", 11000
SEED = 20261031
_here = os.path.dirname(os.path.abspath(__file__))


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def done_keys(path):
    return {tuple(e["key"]) for e in map(json.loads, open(path))} if os.path.exists(path) else set()


def planner(think, guard, commit):
    v2 = load_module("planner_probe_v2")
    probe = v2.probe
    path = f"{OUT}/step3_planner.jsonl"
    done = done_keys(path)
    with open(path, "a") as f:
        for variant in ("V0", "V3"):
            system = probe.SYSTEM if variant == "V0" else v2.SYSTEM_V3
            for c in v2.CASES:
                user = probe.report_text(c) if variant == "V0" else v2.report_v3(c)
                for rep in range(5):
                    key = [variant, c["id"], rep]
                    if tuple(key) in done:
                        continue
                    guard.check(MODEL, MIB, label="step3 planner")
                    # V0 ran with Ollama's default length and context 4096; V3 with 512 and 8192
                    content, sec, raw = llm.chat(system, user, num_predict=2600 if think else (-1 if variant == "V0" else 512),
                                                 model=MODEL, think=think, num_ctx=4096 if variant == "V0" else 8192)
                    r = probe.score(c, v2.extract_json(content or "")) if content else {"ok": False, "error": "aborted"}
                    f.write(json.dumps({"key": key, "variant": variant, "case": c["id"], "rep": rep, "ok": bool(r["ok"]),
                                        "error": r.get("error"), "seconds": sec, "lkh_running": llm.lkh_running(), "content": content,
                                        "prompt_tokens": raw.get("prompt_eval_count"),
                                        "eval_tokens": raw.get("eval_count"), "git_commit": commit}) + "\n")
                    f.flush()


def formal_subset(n, rules=150):
    """Instances of the step-2 evaluation set, drawn until the given rules at this n."""
    rows = [json.loads(line) for line in open(f"data/phase5/text_rules_v2b_tsp{n}.jsonl")]
    bad = set(json.load(open("data/phase5/text_v2b_label_errors.json")).get(str(n), []))
    rows = [r for r in rows if r["v2_complete"] and r["idx"] not in bad]
    out, count = [], 0
    for k in np.random.default_rng([SEED, n]).permutation(len(rows)):
        out.append(rows[int(k)])
        count += len(rows[int(k)]["rules"])
        if count >= rules:
            break
    return out


def formal(think, guard, commit, rules=150):
    path = f"{OUT}/step3_formal.jsonl"
    done = done_keys(path)
    system = F.system_prompt("b2")
    with open(path, "a") as f:
        for n in (200, 1000):
            for row in formal_subset(n, rules):
                key = [n, row["idx"]]
                if tuple(key) in done:
                    continue
                guard.check(MODEL, MIB, label="step3 formal")
                content, sec, raw = llm.chat(system, F.user_prompt(row["text_v2"], n), num_predict=2600 if think else 512,
                                             model=MODEL, think=think)
                obj = llm.extract_json(content)
                pred, errors = F.validate(obj, n)
                f.write(json.dumps({"key": key, "n": n, "idx": row["idx"], "pred": pred, "errors": errors,
                                    "valid_json": obj is not None, "seconds": sec, "lkh_running": llm.lkh_running(), "raw": content,
                                    "git_commit": commit}) + "\n")
                f.flush()


def replay_routes():
    """The routes of part D, rebuilt in the order of scripts/llm_verify_eval.py (the old records keep only the truth)."""
    n = 200
    rows = [json.loads(line) for line in open(f"data/phase5/text_rules_v2b_tsp{n}.jsonl")]
    excluded = set(json.load(open("data/phase5/text_v2b_label_errors.json")).get(str(n), []))
    rows = [r for r in rows if r["v2_complete"] and r["idx"] not in excluded]
    arr = np.load(f"data/phase5/text_rules_tsp{n}.npz")
    rng = np.random.default_rng([20261029, n])
    out = {}
    for row in rows:
        idx = row["idx"]
        coords, ref = arr["coords"][idx], arr["tours"][idx]
        label = [{"type": r["type"], "formal": r["formal"]} for r in row["rules"]]
        out[(idx, "oracle")] = (row, label, coords, ref)
        kinds = sorted({r["type"] for r in label} & set(V.INJECTABLE))
        if kinds:
            kind = str(rng.choice(kinds))
            inj = V.inject_error(rng, coords, ref, label, kind)
            if inj is not None:
                out[(idx, "injected")] = (row, label, coords, inj[0])
    return out


def verifier(think, guard, commit, num_routes=100):
    path = f"{OUT}/step3_verify.jsonl"
    done = done_keys(path)
    old = {(e["idx"], e["route"], e["variant"]): e for e in map(json.loads, open("data/phase5/llm_verify_tsp200.jsonl"))}
    routes = replay_routes()
    keys = sorted({(i, r) for i, r, _ in old})
    pick = np.random.default_rng([SEED, 4]).choice(len(keys), num_routes, replace=False)
    with open(path, "a") as f:
        for k in sorted(int(x) for x in pick):
            idx, name = keys[k]
            row, label, coords, tour = routes[(idx, name)]
            truth = R.violated_rules(label, coords, tour)
            assert truth == old[(idx, name, "plain")]["truth"]         # the replay gives the same route
            route = V.route_text(coords, tour, label)
            for variant in ("plain", "report"):
                key = [idx, name, variant]
                if tuple(key) in done:
                    continue
                report = V.code_report(label, coords, tour) if variant == "report" else None
                system = V.SYSTEM if variant == "plain" else V.SYSTEM_REPORT
                guard.check(MODEL, MIB, label="step3 verify")
                content, sec, raw = llm.chat(system, V.user_prompt(row["text_v2"], route, report),
                                             num_predict=2600 if think else 512, model=MODEL, think=think)
                obj = V.parse(content)
                claimed, unmatched = V.match_claims(obj, label)
                f.write(json.dumps({"key": key, "idx": idx, "route": name, "variant": variant,
                                    "injected_type": old[(idx, name, variant)]["injected_type"],
                                    "rules": old[(idx, name, variant)]["rules"], "truth": truth,
                                    "claimed": sorted(claimed) if claimed is not None else None,
                                    "unmatched_claims": unmatched, "valid_json": obj is not None, "seconds": sec, "lkh_running": llm.lkh_running(),
                                    "prompt_tokens": raw.get("prompt_eval_count"), "raw": content,
                                    "git_commit": commit}) + "\n")
                f.flush()


def main():
    parser = argparse.ArgumentParser(description="qwen3:14b on the planner, formalisation and verifier tasks.")
    parser.add_argument("--think", choices=("on", "off"), default="off")
    parser.add_argument("--parts", nargs="+", default=["planner", "formal", "verify"])
    parser.add_argument("--rules", type=int, default=150, help="formalisation: text_v2b rules per n")
    parser.add_argument("--routes", type=int, default=100, help="verifier: routes (2 variants each)")
    args = parser.parse_args()
    think = args.think == "on"
    guard = llm.GpuGuard(f"{OUT}/gpu_guard.jsonl")
    commit, _ = git_info()
    parts = {"planner": planner, "formal": lambda *a: formal(*a, args.rules), "verify": lambda *a: verifier(*a, args.routes)}
    for p in args.parts:
        parts[p](think, guard, commit)
        print(f"{p}: done", flush=True)


if __name__ == "__main__":
    main()
