"""Phase C: the LLM as a language interface, requests in words -> DSL -> validator -> solvers. Stages, each with resume:

    gen       L1: rules from the formal generator, one world per request; a draw is kept if the oracle episode breaks no rule
    l2        L2: gemma3:12b rewrites L1 as running text, with no filter by any parser
    b         the translation arms: p0, p1, q14, q7
    hybrid    h1 from the logged answers: p1, or q14 if p1 leaves constraint sentences uncovered
    c         the episodes: every distinct set of accepted rules of a request runs once, and its report goes to all the arms that gave it
    latency   30 LLM calls per arm with nothing of mine running

Results in results/warehouse/phaseC/. GPU rules as in phase 5: nvidia-smi before a load and every 5 min, a pause below 2 GB free, one model at a time.

    python scripts/phaseC_language.py gen --num 200 --dev 30 --workers 4
"""

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import time
from multiprocessing import Pool

import numpy as np

from tsp_gnn import llm
from warehouse import constraints as C
from warehouse import dsl as L
from warehouse import episode as P
from warehouse import reqgen as G
from warehouse import reqparser as R
from warehouse import world as Y

OUT = "results/warehouse/phaseC"
DEV = 1000                                      # requests from here on are the parser's development set
MODELS = {"q14": ("qwen3:14b", 11000), "q7": ("qwen2.5:7b", 6000)}
L2_MODEL, L2_MIB = "gemma3:12b", 10000
ATTEMPTS = 5
LEVELS = ("L1", "L2")


def commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def jl(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def memo_path(idx):
    return f"{OUT}/memo/{idx}.pkl"


def gen_job(idx):
    t0 = time.perf_counter()
    w = Y.make(idx)
    ctx = L.Context(w)
    memo = {}
    for attempt in range(ATTEMPTS):
        rng = np.random.default_rng([G.SEED, idx, attempt])
        rules = G.sample_rules(rng, ctx, int(rng.integers(2, 5)))
        prims, rejected = L.check(rules, ctx)
        assert not rejected
        rep, memo = P.run(w, prims, C.to_checks(w, prims) + C.world_checks(w), idx, memo)
        if rep["ok"] and rep["delivered"] == rep["orders"]:
            break
    else:
        return {"idx": idx, "kept": False}
    os.makedirs(f"{OUT}/memo", exist_ok=True)
    pickle.dump(memo, open(memo_path(idx), "wb"))
    return {"idx": idx, "kept": True, "attempts": attempt + 1, "split": "dev" if idx >= DEV else "test", "seed": idx,
            "rules": [{k: r[k] for k in ("dsl", "sentence", "shape")} for r in rules], "background": G.distractors(rng, w, int(rng.integers(1, 3))),
            "text_l1": "\n".join(f"- {r['sentence']}" for r in rules), "oracle": rep, "elapsed": time.perf_counter() - t0}


def main_gen(args):
    path = f"{OUT}/requests.jsonl"
    done = {r["idx"] for r in jl(path)}
    jobs = [i for i in list(range(args.num)) + list(range(DEV, DEV + args.dev)) if i not in done]
    kept = len(done)
    with Pool(args.workers) as pool, open(path, "a") as f:
        for k, rec in enumerate(pool.imap_unordered(gen_job, jobs), 1):
            if rec["kept"]:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                kept += 1
            else:
                print("dropped", rec["idx"], flush=True)
            if k % 20 == 0:
                print(f"gen {k}/{len(jobs)}", flush=True)
    print(f"ALL DONE gen: {kept}")


def requests(split="test", num=None):
    rows = sorted((r for r in jl(f"{OUT}/requests.jsonl") if r["split"] == split), key=lambda r: r["idx"])
    return rows[:num] if num else rows


def paragraphs():
    return {r["idx"]: r for r in jl(f"{OUT}/l2.jsonl")}


def main_l2(args):
    path = f"{OUT}/l2.jsonl"
    done = paragraphs()
    guard = llm.GpuGuard(f"{OUT}/gpu_guard.jsonl")
    start = time.perf_counter()
    with open(path, "a") as f:
        for r in requests(args.split, args.num):
            if r["idx"] in done:
                continue
            prompt = G.l2_prompt([x["sentence"] for x in r["rules"]], r["background"])
            guard.check(L2_MODEL, L2_MIB, label="L2")
            text, sec, raw = llm.chat("You rewrite text faithfully.", prompt, temperature=0.7, seed=r["idx"], num_predict=700, fmt_json=False, model=L2_MODEL)
            f.write(json.dumps({"idx": r["idx"], "paragraph": " ".join((text or "").split()), "prompt": prompt, "seconds": sec, "tokens": raw.get("eval_count"), "lkh_running": llm.lkh_running()}) + "\n")
            f.flush()
    llm.unload(L2_MODEL)
    print(f"ALL DONE l2 {args.split} ({time.perf_counter() - start:.0f} s)")


def text_of(r, level, para):
    return r["text_l1"] if level == "L1" else para[r["idx"]]["paragraph"]


def parser_arm(arm, r, level, para, w):
    text = text_of(r, level, para)
    obj = R.parse_p0(text, w) if arm == "p0" else R.parse_p1(text, w)
    return obj


def llm_arm(arm, user, system, guard):
    model, mib = MODELS[arm]
    total, attempts, raws, obj, raw = 0.0, 0, [], None, {}
    for seed in (0, 1):
        guard.check(model, mib, label=f"B {arm}")
        content, sec, raw = llm.chat(system, user, seed=seed, num_predict=1024, model=model, think=False if arm == "q14" else None)
        total += sec
        attempts += 1
        raws.append(content)
        obj = llm.extract_json(content)
        if obj is not None:
            break
    return obj, {"seconds": total, "attempts": attempts, "raw": raws, "valid_json": obj is not None, "lkh_running": llm.lkh_running(), "prompt_tokens": raw.get("prompt_eval_count"),
                 "eval_tokens": raw.get("eval_count"), "tok_s": raw.get("eval_count", 0) / (raw["eval_duration"] / 1e9) if raw.get("eval_duration") else 0.0}


def main_b(args):
    path = f"{OUT}/translate.jsonl"
    done = {(e["idx"], e["level"], e["arm"]) for e in jl(path)}
    guard = llm.GpuGuard(f"{OUT}/gpu_guard.jsonl")
    system = L.system_prompt()
    para = paragraphs()
    with open(path, "a") as f:
        for arm in args.arms:
            for level in args.levels:
                count = args.q7 if arm == "q7" else args.num
                for r in requests("test", count):
                    if (r["idx"], level, arm) in done or (level == "L2" and not para.get(r["idx"], {}).get("paragraph")):
                        continue
                    w = Y.make(r["seed"])
                    rec = {"idx": r["idx"], "level": level, "arm": arm, "git_commit": commit() if arm in MODELS else None}
                    if arm in MODELS:
                        obj, extra = llm_arm(arm, L.user_prompt(w, text_of(r, level, para)), system, guard)
                        rec.update(extra)
                    else:
                        t0 = time.perf_counter()
                        obj = parser_arm(arm, r, level, para, w)
                        rec.update(seconds=time.perf_counter() - t0, attempts=0, valid_json=True)
                    rec["answer"] = obj
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                print(f"b {arm} {level} done", flush=True)
            if arm in MODELS:
                llm.unload(MODELS[arm][0])
    print("ALL DONE b")


def main_hybrid(args):
    path = f"{OUT}/translate.jsonl"
    rows = jl(path)
    have = {(e["idx"], e["level"]) for e in rows if e["arm"] == "h1"}
    get = {(e["idx"], e["level"], e["arm"]): e for e in rows}
    para = paragraphs()
    with open(path, "a") as f:
        for r in requests("test"):
            for level in LEVELS:
                if (r["idx"], level) in have or (r["idx"], level, "p1") not in get:
                    continue
                t0 = time.perf_counter()
                send = R.send_to_llm(text_of(r, level, para), Y.make(r["seed"]))
                sec = time.perf_counter() - t0
                p1, q = get[(r["idx"], level, "p1")], get.get((r["idx"], level, "q14"))
                if send and q is None:
                    continue
                used = q if send else p1
                f.write(json.dumps({"idx": r["idx"], "level": level, "arm": "h1", "sent": send, "seconds": p1["seconds"] + sec + (q["seconds"] if send else 0.0),
                                    "attempts": q["attempts"] if send else 0, "valid_json": used["valid_json"], "answer": used["answer"]}) + "\n")
    print("ALL DONE hybrid")


def accepted(rec, ctx):
    """(accepted primitives, rejected, dropped items, a key of the set of rules that reach the solvers)."""
    rules, errors = L.validate(rec["answer"], ctx)
    prims, rejected = L.check(rules, ctx)
    canon = json.dumps(sorted(json.dumps(p, sort_keys=True) for p in prims))
    return prims, rejected, errors, hashlib.sha1(canon.encode()).hexdigest()[:16]


def labels(r, ctx):
    return [{"dsl": x["dsl"], "prims": L.expand_rule(x["dsl"], ctx)} for x in r["rules"]]


def c_job(job):
    idx, seed, level, arm, key = job
    out = f"{OUT}/c/{idx}_{key}.json"
    if os.path.exists(out):
        return out
    r = next(x for x in REQ if x["idx"] == idx)
    w = Y.make(seed)
    ctx = L.Context(w)
    rec = next(e for e in TRANS[idx] if e["level"] == level and e["arm"] == arm)
    prims = accepted(rec, ctx)[0]
    truth = [p for x in labels(r, ctx) for p in x["prims"]]
    t0 = time.perf_counter()
    rep, _ = P.run(w, prims, C.to_checks(w, truth), seed, P.load_memo(memo_path(idx)))
    rep["elapsed"] = time.perf_counter() - t0
    assert rep["structural"], (idx, arm, level, rep["problems"])
    tmp = out + ".tmp"
    json.dump(rep, open(tmp, "w"))
    os.replace(tmp, out)
    return out


REQ, TRANS = [], {}


def main_c(args):
    REQ.extend(requests("test", args.num))
    for e in jl(f"{OUT}/translate.jsonl"):
        TRANS.setdefault(e["idx"], []).append(e)
    os.makedirs(f"{OUT}/c", exist_ok=True)
    jobs, seen = [], set()
    start = time.perf_counter()
    for r in REQ:
        ctx = L.Context(Y.make(r["seed"]))
        truth, _ = L.check(labels(r, ctx), ctx)
        arms = [("oracle", {"answer": {"rules": [x["dsl"] for x in r["rules"]]}})]
        for e in TRANS.get(r["idx"], []):
            if args.arms is None or e["arm"] in args.arms:
                key = accepted(e, ctx)[3]
                if (r["idx"], key) not in seen and not os.path.exists(f"{OUT}/c/{r['idx']}_{key}.json"):
                    seen.add((r["idx"], key))
                    jobs.append((r["idx"], r["seed"], e["level"], e["arm"], key))
        key = accepted(arms[0][1], ctx)[3]
        if (r["idx"], key) not in seen and not os.path.exists(f"{OUT}/c/{r['idx']}_{key}.json"):
            seen.add((r["idx"], key))
            TRANS.setdefault(r["idx"], []).append({"level": "L1", "arm": "oracle", **arms[0][1]})
            jobs.append((r["idx"], r["seed"], "L1", "oracle", key))
    print(f"{len(jobs)} episodes to run ({time.perf_counter() - start:.0f} s to list them)", flush=True)
    with Pool(args.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(c_job, jobs), 1):
            if k % 50 == 0:
                print(f"c {k}/{len(jobs)}  {time.perf_counter() - start:.0f} s", flush=True)
    print("ALL DONE c")


def main_latency(args):
    path = f"{OUT}/latency_alone.jsonl"
    done = {(e["arm"], e["level"], e["idx"]) for e in jl(path)}
    while llm.lkh_running():
        print("waiting: LKH still running", flush=True)
        time.sleep(60)
    guard = llm.GpuGuard(f"{OUT}/gpu_guard.jsonl")
    system = L.system_prompt()
    para = paragraphs()
    with open(path, "a") as f:
        for arm in MODELS:
            model, mib = MODELS[arm]
            llm.wait_cool()
            guard.check(model, mib, label="latency")
            llm.chat("Answer with one word.", "Ready?", fmt_json=False, model=model, num_predict=8, think=False if arm == "q14" else None)
            for level in LEVELS:
                pool = [r for r in requests("test") if level == "L1" or para.get(r["idx"], {}).get("paragraph")]
                if arm == "q7":
                    pool = pool[:100]
                pick = np.random.default_rng([G.SEED, LEVELS.index(level), 7]).choice(len(pool), 15, replace=False)
                for k in pick:
                    r = pool[int(k)]
                    if (arm, level, r["idx"]) in done:
                        continue
                    before = llm.lkh_running()
                    _, x = llm_arm(arm, L.user_prompt(Y.make(r["seed"]), text_of(r, level, para)), system, guard)
                    temp, clock = llm.gpu_state()
                    f.write(json.dumps({"arm": arm, "level": level, "idx": r["idx"], "seconds": x["seconds"], "attempts": x["attempts"], "tok_s": x["tok_s"], "eval_tokens": x["eval_tokens"],
                                        "prompt_tokens": x["prompt_tokens"], "load1": os.getloadavg()[0], "temp_c": temp, "clock_mhz": clock, "lkh_running": max(before, x["lkh_running"])}) + "\n")
                    f.flush()
            llm.unload(model)
    print("ALL DONE latency")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("what", choices=("gen", "l2", "b", "hybrid", "c", "latency"))
    p.add_argument("--num", type=int, default=None)
    p.add_argument("--dev", type=int, default=30)
    p.add_argument("--q7", type=int, default=100)
    p.add_argument("--split", default="test")
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--levels", nargs="+", default=list(LEVELS))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    globals()[f"main_{args.what}"](args)


if __name__ == "__main__":
    main()
