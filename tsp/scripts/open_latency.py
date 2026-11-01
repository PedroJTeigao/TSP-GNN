"""Phase 5, step 3: LLM latency measured alone, for the report (the runs before it may have had LKH-3 in parallel).

30 calls per LLM arm, 5 per (n, level), drawn with a fixed seed from the test instances, one arm at a time, with the same
prompt, retry and options as scripts/open_formalize.py. It waits until no LKH process of mine is running, and records for
every call whether one appeared and the load average, since other users of the server cannot be stopped. Each arm starts
on a cooled GPU (at most 70 C) and every call records the temperature and clock, because the T4 throttles to about 300 MHz
when it is hot: after hours of work a call takes about twice as long.

Writes results/phase5/open_language/latency_alone.jsonl; a rerun skips what is already there.

Example:
    python scripts/open_latency.py --think off
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

from tsp_gnn import llm
from tsp_gnn import open_dsl as D
from tsp_gnn import open_eval as E

OUT = "results/phase5/open_language"
SEED = 20261031
_spec = importlib.util.spec_from_file_location(
    "open_formalize", os.path.join(os.path.dirname(os.path.abspath(__file__)), "open_formalize.py"))
Z = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Z)


def main():
    parser = argparse.ArgumentParser(description="LLM latency with no LKH of mine running.")
    parser.add_argument("--think", choices=("on", "off"), default="off")
    parser.add_argument("--per-cell", type=int, default=5)
    args = parser.parse_args()
    think = args.think == "on"
    path = f"{OUT}/latency_alone.jsonl"
    done = {(e["arm"], e["n"], e["level"], e["idx"]) for e in map(json.loads, open(path))} if os.path.exists(path) else set()
    while llm.lkh_running():
        print("waiting: LKH still running", flush=True)
        time.sleep(60)
    guard = llm.GpuGuard(f"{OUT}/gpu_guard.jsonl")
    system = D.system_prompt()
    cells = {}
    for n in (200, 1000):
        for lv in E.LEVELS:
            insts = E.load_level(n, lv)
            pick = np.random.default_rng([SEED, n, E.LEVELS.index(lv)]).choice(len(insts), args.per_cell, replace=False)
            cells[(n, lv)] = [insts[int(k)] for k in pick]
    with open(path, "a") as f:
        for arm in ("q14", "q7", "g12"):
            model, mib = Z.MODELS[arm]
            llm.wait_cool()
            guard.check(model, mib, label="latency")
            llm.chat("Answer with one word.", "Ready?", fmt_json=False, model=model, num_predict=8,
                     think=think if arm == "q14" else None)     # loads the model; not counted
            for (n, lv), insts in cells.items():
                for inst in insts:
                    if (arm, n, lv, inst["idx"]) in done:
                        continue
                    before = llm.lkh_running()
                    _, _, x = Z.llm_arm(arm, inst, n, think, system, guard)
                    f.write(json.dumps({"arm": arm, "n": n, "level": lv, "idx": inst["idx"], "seconds": x["seconds"],
                                        "attempts": x["attempts"], "tok_s": x["tok_s"], "eval_tokens": x["eval_tokens"],
                                        "prompt_tokens": x["prompt_tokens"], "load1": os.getloadavg()[0],
                                        "temp_c": llm.gpu_state()[0], "clock_mhz": llm.gpu_state()[1],
                                        "lkh_running": max(before, x["lkh_running"])}) + "\n")
                    f.flush()
            llm.unload(model)
    print("ALL DONE latency")


if __name__ == "__main__":
    main()
