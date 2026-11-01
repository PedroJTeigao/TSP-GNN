"""Phase 5, step 3, part 2B: who formalises best. Every arm writes DSL rules for every instance of levels L0, L1, L2.

Arms: p0 old keyword parser (tsp_gnn.keyword_parser, primitive rules; the text is laid out for it by open_parser.p0_text);
p1 parser extended to the DSL (tsp_gnn.open_parser; on L0 it is the keyword parser); q7 qwen2.5:7b; q14 qwen3:14b;
g12 gemma3:12b. The LLMs get the same prompt (open_dsl: legend + text + DSL + 3 L0 examples), temperature 0, seed 0,
context 8192, format json; if the JSON is invalid (or Ollama aborts) there is 1 new attempt with seed 1, counted in the
time. The oracle needs no record: it is the label.

Writes results/phase5/open_language/formalize_tsp<n>.jsonl, one record per (instance, level, arm); a rerun skips the
records already there, so a stopped run resumes.

Example:
    python scripts/open_formalize.py --n 200 --arms q14 --think off
"""

import argparse
import json
import os
import time

from tsp_gnn import keyword_parser as K
from tsp_gnn import llm
from tsp_gnn import open_dsl as D
from tsp_gnn import open_eval as E
from tsp_gnn import open_parser as P
from tsp_gnn.utils import git_info

OUT = "results/phase5/open_language"
MODELS = {"q7": ("qwen2.5:7b", 6000), "q14": ("qwen3:14b", 11000), "q14t": ("qwen3:14b", 11000), "g12": ("gemma3:12b", 10000)}


def parser_arm(arm, inst, n):
    if arm == "p0":
        text = inst["text"] if inst["level"] == "L0" else P.p0_text(inst["legend"], inst["text"])
        return [{"dsl": None, "prims": [p]} for p in K.parse_keywords(text, n)], []
    obj = P.parse_l0(inst["text"], n) if inst["level"] == "L0" else P.parse(inst["text"], inst["legend"], n)
    return D.validate(obj, inst["ctx"])


def llm_arm(arm, inst, n, think, system, guard):
    model, mib = MODELS[arm]
    th = {"q14": think, "q14t": True}.get(arm)      # q14t: the same model with thinking on
    total, attempts, raws = 0.0, 0, []
    obj, raw = None, {}
    for seed in (0, 1):
        guard.check(model, mib, label=f"B {arm} n={n}")
        user = D.user_prompt(inst["legend_text"], inst["text"], n)
        content, sec, raw = llm.chat(system, user, seed=seed, num_predict=2600 if th else 1024, model=model, think=th)
        total += sec
        attempts += 1
        raws.append(content)
        obj = llm.extract_json(content)
        if obj is not None:
            break
    accepted, errors = D.validate(obj, inst["ctx"])
    if obj is None:
        errors = ["invalid JSON" if raws[-1] is not None else f"LLM aborted: {raw.get('error', '')[:200]}"]
    return accepted, errors, {"seconds": total, "attempts": attempts, "raw": raws, "valid_json": obj is not None,
                              "lkh_running": llm.lkh_running(),
                              "prompt_tokens": raw.get("prompt_eval_count"), "eval_tokens": raw.get("eval_count"),
                              "tok_s": raw.get("eval_count", 0) / (raw["eval_duration"] / 1e9) if raw.get("eval_duration") else 0.0}


def main():
    parser = argparse.ArgumentParser(description="Formalisation arms on the open-language levels.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--arms", nargs="+", default=["p0", "p1", "q14", "q7", "g12"])
    parser.add_argument("--levels", nargs="+", default=list(E.LEVELS))
    parser.add_argument("--num", type=int, default=None, help="test instances per level (default: all)")
    parser.add_argument("--num-l0", type=int, default=None, help="test instances of L0, if different from --num")
    parser.add_argument("--think", choices=("on", "off"), default="off", help="thinking mode of qwen3:14b")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    n = args.n
    think = args.think == "on"
    out = args.out or f"{OUT}/formalize_tsp{n}.jsonl"
    done = set()
    if os.path.exists(out):
        done = {(e["idx"], e["level"], e["arm"]) for e in map(json.loads, open(out))}
        print(f"{len(done)} records already in {out}: skipped")
    guard = llm.GpuGuard(f"{OUT}/gpu_guard.jsonl")
    system = D.system_prompt()
    commit, dirty = git_info()
    start, calls = time.perf_counter(), 0
    levels = {lv: E.load_level(n, lv, args.num_l0 if lv == "L0" and args.num_l0 else args.num) for lv in args.levels}
    with open(out, "a") as f:
        for arm in args.arms:                     # one arm at a time: the system prompt stays cached in Ollama
            for lv, insts in levels.items():
                for inst in insts:
                    if (inst["idx"], lv, arm) in done:
                        continue
                    rec = {"idx": inst["idx"], "n": n, "level": lv, "arm": arm, "git_commit": commit}
                    if arm in MODELS:
                        accepted, errors, extra = llm_arm(arm, inst, n, think, system, guard)
                        calls += extra["attempts"]
                        rec.update(extra)
                    else:
                        t0 = time.perf_counter()
                        accepted, errors = parser_arm(arm, inst, n)
                        rec.update(seconds=time.perf_counter() - t0, attempts=0, raw=None, valid_json=True)
                    rec.update(pred=accepted, errors=errors)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                print(f"  {arm} {lv}: done  ({calls} LLM calls, {time.perf_counter() - start:.0f} s)", flush=True)
            if arm in MODELS:
                llm.unload(MODELS[arm][0])
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out}")


if __name__ == "__main__":
    main()
