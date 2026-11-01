"""Phase 5, step 3, step 0: size and speed of the larger local models, and thinking on / off for qwen3:14b.

For each model (one loaded at a time, keep_alive 0 between them): 20 calls of a typical formalisation request (DSL prompt
+ a text_v2b instance + a legend of the size used at n = 1000), recording tokens/s, mean and p95 latency, GPU memory and
how much of the model sits in the GPU. For qwen3:14b, the same request on 20 text_v2b instances with think true and
think false, scored by exact rules, to choose the mode by the pair (hits, time).

Writes results/phase5/open_language/step0.json and step0_calls.jsonl (every raw answer); a rerun skips the finished
parts.

Example:
    python scripts/open_step0.py --models qwen3:14b gemma3:12b qwen2.5:7b
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np

from tsp_gnn import llm
from tsp_gnn import open_dsl as D
from tsp_gnn import open_gen as G

OUT = "results/phase5/open_language"
NEED = {"qwen3:14b": 11000, "gemma3:12b": 10000, "qwen2.5:7b": 6000}      # MiB, upper estimates before measuring


def gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60).stdout
    used, free = out.strip().split(", ")
    return {"used_mib": int(used), "free_mib": int(free)}


def ps(model):
    size, vram = llm.loaded().get(model, (0, 0))
    return {"size_gb": size / 1e9, "vram_gb": vram / 1e9, "in_gpu_pct": 100 * vram / size if size else None}


def requests(num):
    """(system, user, label, ctx) of num text_v2b instances at n = 1000, with a legend of the n = 1000 size, so that the
    prompt is as long as the real ones."""
    rows = [json.loads(line) for line in open("data/phase5/text_rules_v2b_tsp1000.jsonl")]
    rows = [r for r in rows if r["v2_complete"]][:num]
    arr = np.load("data/phase5/text_rules_tsp1000.npz")
    out = []
    for r in rows:
        coords = arr["coords"][r["idx"]]
        legend = G.make_legend(1000, r["idx"], coords)
        ctx = G.l0_context(r, coords, 1000)
        label = G.l0_label(r, ctx)
        out.append((D.system_prompt(), D.user_prompt(D.legend_block(legend), r["text_v2"], 1000), label, ctx))
    return out


def run(model, reqs, calls, tag, guard, think=None, num_predict=1024):
    guard.check(model, NEED[model], label=tag)
    llm.chat("Answer with one word.", "Ready?", fmt_json=False, model=model, num_predict=8, think=think)    # loads it
    rec = {"model": model, "tag": tag, "think": think, "load": ps(model), "gpu": gpu(), "calls": 20}
    secs, toks, hits, rules, exact = [], [], 0, 0, 0
    assert not llm.lkh_running(), "step 0 must be measured with no LKH of mine running"
    for k, (system, user, label, ctx) in enumerate(reqs):
        guard.check(model, NEED[model], label=tag)
        content, sec, raw = llm.chat(system, user, model=model, num_predict=num_predict, think=think)
        obj = llm.extract_json(content)            # a timeout gives content None: scored as no rules
        accepted, errors = D.validate(obj, ctx)
        s = D.score(label, [p for a in accepted for p in a["prims"]])
        hits += sum(s["correct"])
        rules += len(s["correct"])
        exact += s["exact"]
        secs.append(sec)
        speed = raw.get("eval_count", 0) / (raw.get("eval_duration", 1) / 1e9) if raw.get("eval_duration") else 0
        toks.append(speed)
        calls.write(json.dumps({"model": model, "tag": tag, "think": think, "k": k, "seconds": sec, "tok_s": speed,
                                "prompt_tokens": raw.get("prompt_eval_count"), "eval_tokens": raw.get("eval_count"),
                                "content": content, "thinking": raw.get("message", {}).get("thinking"),
                                "correct": s["correct"], "errors": errors}) + "\n")
        calls.flush()
    rec.update(lat_mean=float(np.mean(secs)), lat_p95=float(np.percentile(secs, 95)), tok_s=float(np.mean(toks)),
               rules_exact=hits, rules=rules, instances_exact=exact, gpu_end=gpu(), load_end=ps(model))
    return rec


def main():
    parser = argparse.ArgumentParser(description="Speed of the larger models; thinking on / off for qwen3:14b.")
    parser.add_argument("--models", nargs="+", default=["qwen3:14b", "gemma3:12b", "qwen2.5:7b"])
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "step0.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    guard = llm.GpuGuard(os.path.join(OUT, "gpu_guard.jsonl"))
    reqs = requests(20)
    with open(os.path.join(OUT, "step0_calls.jsonl"), "a") as calls:
        for model in args.models:
            modes = [("think_off", False), ("think_on", True)] if model == "qwen3:14b" else [("speed", None)]
            for tag, think in modes:
                key = f"{model} {tag}"
                if key in res:
                    continue
                print(f"{time.strftime('%H:%M:%S')} {key}", flush=True)
                res[key] = run(model, reqs, calls, tag, guard, think, 2600 if think else 1024)
                print(json.dumps(res[key]), flush=True)
                json.dump(res, open(path, "w"), indent=2)
            llm.unload(model)
    print("ALL DONE step 0")


if __name__ == "__main__":
    main()
