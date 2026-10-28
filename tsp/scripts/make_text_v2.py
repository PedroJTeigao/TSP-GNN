"""Phase 5, step 2, part A: text_v2, a second wording of the step-1 text rules (see tsp_gnn.textv2).

For each instance of data/phase5/text_rules_tsp<n>.jsonl (its formal rules are the label):
    1. in 20 % of the instances without a precedence, an extra non-binding time window (binding = False) that the
       reference route already meets is added to the rules (label and text_v1 included);
    2. names for 10 % of the customers, holding the customers of about half the rules (those rules use the name);
    3. 1-2 irrelevant sentences; in 20 % of the instances with 2+ rules, two rules are merged into one sentence;
    4. the local qwen2.5:7b (private Ollama, port 11435) writes 2 paraphrases per rule (and per merged pair) from
       the formal translation at temperature 0.8; candidates failing textv2.check_paraphrase are rejected and the
       call repeated (at most --calls calls). A rule with no accepted paraphrase falls back to its generator text
       (with names) and is marked "fallback"; a pair with no accepted merged sentence stays as two sentences.
Every LLM call is appended to data/phase5/text_v2_calls_tsp<n>.jsonl; a rerun replays it (resume after a stop).

Writes data/phase5/text_rules_v2_tsp<n>.jsonl: per instance text_v1, text_v2, the rules (formal label, binding,
named, v1 text, accepted paraphrases, the sentence used in text_v2 and its source), names, distractors, merged pair.

Example:
    python scripts/make_text_v2.py --n 200
"""

import argparse
import datetime
import json
import os
import re
import time
from collections import Counter

import numpy as np

from tsp_gnn import llm
from tsp_gnn import rules as R
from tsp_gnn import textv2 as T
from tsp_gnn.utils import git_info

SEED = 20261028
EXTRA_FRACTION = 0.2      # instances without a precedence that get an extra non-binding window
MERGE_FRACTION = 0.2      # instances with 2+ rules where two rules share one sentence


def plan_instance(rec, coords, tour, rng, extra, merge):
    """Everything about the instance except the LLM sentences."""
    n = rec["n"]
    rules = [dict(r, binding=True) for r in rec["rules"]]
    if extra:
        rules.append(T.extra_window(rng, coords, tour, rules, rec["pure_length"]))
    assert R.check_all(rules, coords, tour)
    name_rules = [k for k, r in enumerate(rules) if T.rule_customers(r) and rng.random() < 0.5]
    names = T.assign_names(rng, n, rules, name_rules)
    used = set()
    distractors = [T.distractor(rng, n, names, used) for _ in range(int(rng.integers(1, 3)))]
    pair = sorted(int(x) for x in rng.choice(len(rules), 2, replace=False)) if merge else None
    for k, r in enumerate(rules):
        r["named"] = k in name_rules
    return {"rules": rules, "names": names, "names_line": T.names_text(names, rng), "distractors": distractors,
            "merged": pair, "text_v1": R.instance_text(rules)}


def generate(key, rules, names, cache, log, args, seed0):
    """Up to 2 accepted sentences for rules (1 rule, or a merged pair), and the rejection reasons."""
    system, user = T.paraphrase_prompt(rules, names)
    ok, why = [], Counter()
    for call in range(args.calls):
        ck = f"{key}:{call}"
        if ck in cache:
            content = cache[ck]["content"]
        else:
            content, sec, _ = llm.chat(system, user, temperature=args.temperature, seed=seed0 + call,
                                       num_predict=256)
            entry = {"key": ck, "user": user, "content": content, "seconds": sec}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            cache[ck] = entry
        for s in T.parse_sentences(content):
            s = s.strip()
            reason = T.check_paraphrase(s, rules, names)
            if reason is None and s not in ok:
                ok.append(s)
            elif reason:
                why[reason] += 1
        if len(ok) >= 2:
            break
    return ok[:2], dict(why)


def with_names(text, names):
    for c, nm in names.items():
        text = re.sub(rf"\b[Cc]ustomer {c}\b", nm, text)
    return text[0].upper() + text[1:]


def main():
    parser = argparse.ArgumentParser(description="Build text_v2 (paraphrases, names, distractors) for step-1 data.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--calls", type=int, default=4, help="LLM calls per rule at most (2 sentences each)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=None, help="first instances only (tests)")
    parser.add_argument("--out", default=None, help="default data/phase5/text_rules_v2_tsp<n>")
    args = parser.parse_args()
    n = args.n
    out = args.out or f"data/phase5/text_rules_v2_tsp{n}"
    if os.path.exists(out + ".jsonl"):
        print(f"{out}.jsonl already exists: not overwritten")
        return
    recs = [json.loads(line) for line in open(f"data/phase5/text_rules_tsp{n}.jsonl")]
    arr = np.load(f"data/phase5/text_rules_tsp{n}.npz")
    if args.limit:
        recs = recs[:args.limit]

    rng = np.random.default_rng([SEED, n])
    no_prec = [r["idx"] for r in recs if all(x["type"] != "precedence" for x in r["rules"])]
    extra = set(int(i) for i in rng.choice(no_prec, round(EXTRA_FRACTION * len(no_prec)), replace=False))
    multi = [r["idx"] for r in recs if r["num_rules"] + (r["idx"] in extra) >= 2]
    merge = set(int(i) for i in rng.choice(multi, round(MERGE_FRACTION * len(multi)), replace=False))

    calls_path = f"data/phase5/text_v2_calls_tsp{n}.jsonl"
    cache = {}
    if os.path.exists(calls_path):
        for line in open(calls_path):
            e = json.loads(line)
            cache[e["key"]] = e
        print(f"replaying {len(cache)} cached LLM calls")
    commit, dirty = git_info()
    start = time.perf_counter()
    rows = []
    with open(calls_path, "a") as log:
        for rec in recs:
            i = rec["idx"]
            irng = np.random.default_rng([SEED, n, i])
            p = plan_instance(rec, arr["coords"][i], arr["tours"][i], irng, i in extra, i in merge)
            rules, names = p["rules"], p["names"]
            items = []
            for k, r in enumerate(rules):
                ok, why = generate(f"{i}:r{k}", [r], names, cache, log, args, (n * 1000 + i) * 100 + 10 * k)
                r["paraphrases"], r["rejected"] = ok, why
                r["v1_text"] = r.pop("text")
                if ok:
                    r["v2_text"], r["v2_source"] = ok[0], "paraphrase"
                else:
                    r["v2_text"], r["v2_source"] = with_names(r["v1_text"], names), "fallback"
            merged_text, merged_rejected = None, {}
            if p["merged"]:
                a, b = p["merged"]
                ok, merged_rejected = generate(f"{i}:m", [rules[a], rules[b]], names, cache, log, args,
                                               (n * 1000 + i) * 100 + 90)
                if ok:
                    merged_text = ok[0]
                    rules[a]["v2_source"] = rules[b]["v2_source"] = "merged"
            for k, r in enumerate(rules):
                if r["v2_source"] != "merged":
                    items.append(r["v2_text"])
            if merged_text:
                items.append(merged_text)
            items += p["distractors"]
            rows.append({"idx": i, "n": n, "rules": rules, "names": {str(c): v for c, v in names.items()},
                         "names_line": p["names_line"], "distractors": p["distractors"],
                         "merged": p["merged"], "merged_text": merged_text, "merged_rejected": merged_rejected,
                         "extra_window": i in extra, "kind_v2": R.problem_kind(rules),
                         "text_v1": p["text_v1"], "text_v2": T.compose_v2(p["names_line"], items, irng),
                         "reference": {k: rec[k] for k in ("kind", "length", "objective", "pure_length")}})
            if len(rows) % 20 == 0 or len(rows) == len(recs):
                print(f"  {len(rows)}/{len(recs)} instances  ({time.perf_counter() - start:.0f} s)", flush=True)

    meta = {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(), "seed": SEED,
            "model": llm.MODEL, "temperature": args.temperature, "calls": args.calls,
            "total_seconds": time.perf_counter() - start}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps({**row, "meta": meta}) + "\n")
    src = Counter(r["v2_source"] for row in rows for r in row["rules"])
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  sources {dict(src)}  |  "
          f"merged {sum(bool(r['merged_text']) for r in rows)}/{sum(bool(r['merged']) for r in rows)}  |  "
          f"saved to {out}.jsonl")


if __name__ == "__main__":
    main()
