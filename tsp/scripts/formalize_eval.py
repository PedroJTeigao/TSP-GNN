"""Phase 5, step 2, part B: formalisation arms (text -> formal rules as JSON) on text_v1 and text_v2b.

Arms (tsp_gnn.formalize):
    b0  regular expressions written over the generator's 4 formulations per type (no LLM);
    b0p "b0+": keyword parser built from the meaning filters of part A (tsp_gnn.keyword_parser; no LLM);
    b1  local qwen2.5:7b (private Ollama, port 11435), catalog of the 5 types + JSON schema, zero-shot,
        temperature 0, context 8192;
    b2  b1 + 5 worked examples (hand-written, not from the test set);
    b3  an API model if OPENAI_API_KEY / ANTHROPIC_API_KEY is set: none is, so it is recorded as not tested.
Every answer is validated against the schema (formalize.validate); items that do not fit are dropped and counted.

Reads data/phase5/text_rules_v2b_tsp<n>.jsonl. Writes data/phase5/formalize_tsp<n>.jsonl: one record per
(instance, text, arm) with the predicted rules, schema errors, latency and the raw answer. LLM calls are cached in
the same file: a rerun skips the records already there (resume after a stop). Scoring is done by the report script.

Example:
    python scripts/formalize_eval.py --n 200
"""

import argparse
import json
import os
import time

from tsp_gnn import formalize as F
from tsp_gnn import keyword_parser as K
from tsp_gnn import llm
from tsp_gnn.utils import git_info

TEXTS = ("v1", "v2")
ARMS = ("b0", "b1", "b2")


def main():
    parser = argparse.ArgumentParser(description="Text -> formal rules: regex parser and local LLM arms.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), help=f"of {ARMS} and b0p")
    parser.add_argument("--limit", type=int, default=None, help="first instances only (tests)")
    parser.add_argument("--out", default=None, help="default data/phase5/formalize_tsp<n>.jsonl")
    args = parser.parse_args()
    n = args.n
    out = args.out or f"data/phase5/formalize_tsp{n}.jsonl"
    rows = [json.loads(line) for line in open(f"data/phase5/text_rules_v2b_tsp{n}.jsonl")]
    if args.limit:
        rows = rows[:args.limit]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            e = json.loads(line)
            done.add((e["idx"], e["text"], e["arm"]))
        print(f"{len(done)} records already in {out}: skipped")
    commit, dirty = git_info()
    api = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if os.environ.get(k)]
    print(f"b3: {'keys ' + ', '.join(api) + ' found, but no API arm is implemented' if api else 'not tested (no API key)'}")
    start, calls = time.perf_counter(), 0
    with open(out, "a") as f:
        for arm in args.arms:                     # one arm at a time: the system prompt stays cached in Ollama
            for text in TEXTS:
                for row in rows:
                    key = (row["idx"], text, arm)
                    if key in done:
                        continue
                    t = row["text_v1"] if text == "v1" else row["text_v2"]
                    rec = {"idx": row["idx"], "n": n, "text": text, "arm": arm, "git_commit": commit}
                    if arm in ("b0", "b0p"):
                        t0 = time.perf_counter()
                        pred, errors = (F.parse_regex(t) if arm == "b0" else K.parse_keywords(t, n)), []
                        rec.update(seconds=time.perf_counter() - t0, raw=None)
                    else:
                        content, sec, raw = llm.chat(F.system_prompt(arm), F.user_prompt(t, n), temperature=0.0,
                                                     seed=0, num_predict=512)
                        obj = llm.extract_json(content)
                        pred, errors = F.validate(obj, n)
                        if content is None:
                            errors = [f"LLM aborted: {raw.get('error', '')[:200]}"]
                        elif obj is None:
                            errors = ["invalid JSON"]
                        rec.update(seconds=sec, raw=content, prompt_tokens=raw.get("prompt_eval_count"),
                                   eval_tokens=raw.get("eval_count"))
                        calls += 1
                    rec.update(pred=pred, errors=errors)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                print(f"  {arm} {text}: done  ({calls} LLM calls, {time.perf_counter() - start:.0f} s)", flush=True)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out}")


if __name__ == "__main__":
    main()
