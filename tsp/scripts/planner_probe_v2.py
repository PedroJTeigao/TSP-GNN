"""Phase 5, step 0b: prompt variants for the LLM planner, against a fixed rule, on the 8 cases of planner_probe.py.

    V0  the planner_probe.py prompt (results read from <out>/planner_probe.json, not re-run)
    V1  explicit procedure in the system prompt (list the actions that fit; none -> stop; lowest expected gap;
        ignore unavailable tools); 3-5 lines of reasoning in plain text BEFORE the JSON, which is taken from the end
    V2  V1 + 3 solved examples (cases different from the 8 test cases)
    V3  pre-filtered list: the code computes the actions that fit and their expected gap; the LLM only picks one
        (this is already most of the rule: only the arg-min is left to the model)
    R   the fixed rule (phase 5 arm (a)): among the actions that fit, the lowest expected gap; stop if none fits or
        none beats the current gap

The rule and V3 use EXPECTED below: the same measured points as the text tables of planner_probe.EFFECTS, as
(extra seconds, gap %) pairs. An action fits if its first point fits in the time left; lkh_whole always gets
time_limit = time left, and its expected gap is the last measured point that fits (a step function).
Writes <out>/planner_probe_v2.json, <out>/planner_probe_v2_calls.jsonl and <out>/planner_probe_v2.md (the one-line
verdict is read from <out>/planner_probe_v2_notes.md).

Example:
    python scripts/planner_probe_v2.py --variants V1 V2 V3 R --models qwen2.5:7b qwen2.5:3b --repeats 5
"""

import argparse
import datetime
import importlib.util
import json
import os
import time

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("planner_probe", os.path.join(_here, "planner_probe.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

from tsp_gnn.utils import git_info

CASES = probe.CASES
API_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

# (tool, args without time_limit, [(extra seconds, gap %), ...]) per size; same numbers as planner_probe.EFFECTS.
EXPECTED = {
    10000: [("lkh_whole", {"candidates": "QUADRANT5"}, [(2.7, 0.39), (5.9, 0.20), (11.7, 0.12), (19.0, 0.085)]),
            ("lkh_whole", {"candidates": "NN5"}, [(2.6, 0.45), (6.2, 0.33), (11.9, 0.22), (19.2, 0.16)]),
            ("lkh_whole", {"candidates": "NN8"}, [(7.3, 0.22), (13.4, 0.18), (20.1, 0.14)]),
            ("lkh_whole", {"candidates": "ALPHA"}, [(139.4, 0.20)]),
            ("lkh_whole", {"candidates": "POPMUSIC"}, [(52.2, 0.16)]),
            ("triage", {"frac": 10}, [(3.8, 4.36)]), ("triage", {"frac": 20}, [(6.3, 4.21)]),
            ("triage", {"frac": 50}, [(12.4, 3.98)]), ("triage", {"frac": 100}, [(20.1, 3.85)]),
            ("windows_model", {"decoder": "greedy"}, [(3.8, 4.45)]),
            ("windows_model", {"decoder": "beam10"}, [(15.7, 4.19)])],
    1000: [("lkh_whole", {"candidates": "QUADRANT5"}, [(0.16, 0.29), (0.48, 0.11), (1.01, 0.053), (1.71, 0.035)]),
           ("lkh_whole", {"candidates": "NN5"}, [(0.24, 0.45), (0.47, 0.26), (1.01, 0.14), (1.70, 0.105)]),
           ("lkh_whole", {"candidates": "NN8"}, [(0.39, 0.18), (1.09, 0.11), (1.76, 0.060)]),
           ("lkh_whole", {"candidates": "ALPHA"}, [(1.34, 0.16), (1.90, 0.050)]),
           ("lkh_whole", {"candidates": "POPMUSIC"}, [(1.12, 0.064), (1.80, 0.010)]),
           ("triage", {"frac": 10}, [(0.39, 4.51)]), ("triage", {"frac": 20}, [(0.54, 4.34)]),
           ("triage", {"frac": 50}, [(1.12, 4.08)]), ("triage", {"frac": 100}, [(1.76, 3.92)]),
           ("windows_model", {"decoder": "greedy"}, [(0.30, 4.65)]),
           ("windows_model", {"decoder": "beam10"}, [(1.34, 4.34)])],
    200: [("triage", {"frac": 10}, [(0.07, 4.31)]), ("triage", {"frac": 20}, [(0.11, 4.10)]),   # lkh_whole: not measured
          ("triage", {"frac": 50}, [(0.20, 3.72)]), ("triage", {"frac": 100}, [(0.28, 3.54)]),
          ("windows_model", {"decoder": "beam10"}, [(0.24, 4.12)])],
}


def feasible(case, ranked=True):
    """[(tool, args, expected seconds, expected gap)] of the actions that fit (best gap first if ranked, else in the
    order of the effects table); stop is not listed."""
    if case["length"] is None:                                  # no tour yet: nothing can run within this budget
        return []
    left = round(case["budget"] - case["used"], 2)
    out = []
    for tool, args, points in EXPECTED[case["n"]]:
        if tool in case["unavailable"]:
            continue
        fit = [(t, g) for t, g in points if t <= left]
        if not fit:
            continue
        t, g = fit[-1]
        if tool == "lkh_whole":
            args, t = {**args, "time_limit": left}, left
        if g < case["gap"]:                                     # must beat the current tour
            out.append((tool, args, t, g))
    return sorted(out, key=lambda a: a[3]) if ranked else out


def rule(case):
    acts = feasible(case)
    return {"tool": "stop", "args": {}, "reason": "nothing fits or improves"} if not acts else \
        {"tool": acts[0][0], "args": acts[0][1], "reason": f"lowest expected gap that fits ({acts[0][3]:g} %)"}


PROCEDURE = """
Procedure (follow it step by step):
1) List the actions whose total time fits in the time left (for lkh_whole: setup + at least one trial must fit; then
   use time_limit = the time left).
2) If none fits, the answer is stop.
3) Otherwise choose the one with the lowest expected gap; if it is not lower than the current gap, stop.
4) If a tool is marked "unavailable", ignore it.
First write your reasoning in 3-5 short lines of plain text, then the JSON object on the last line:
{"tool": "<tool>", "args": {...}, "reason": "<one sentence>"}"""

SYSTEM_V1 = probe.SYSTEM.rsplit("\n\nAnswer with", 1)[0] + "\n" + PROCEDURE

# Solved examples for V2 (not among the 8 test cases).
EXAMPLES = [
    ({"n": 1000, "budget": 3.0, "used": 0.26, "length": 24.23, "gap": 4.8, "dis": (0.121, 4.6),
      "history": ["greedy edge + 2-opt (0.26 s, 4.8 %)"], "unavailable": []},
     "Time left: 2.74 s.\nFits: all lkh_whole variants (largest measured points: QUADRANT5 0.035 %, NN5 0.105 %, "
     "NN8 0.060 %, ALPHA 0.050 %, POPMUSIC 0.010 %), all triage and windows_model rows (3.92-4.65 %).\n"
     "Lowest expected gap: POPMUSIC, 0.010 % at 1.80 s, below the current 4.8 %.\nUse all the time left.",
     {"tool": "lkh_whole", "args": {"candidates": "POPMUSIC", "time_limit": 2.74},
      "reason": "POPMUSIC has the lowest expected gap among the actions that fit."}),
    ({"n": 10000, "budget": 5.0, "used": 2.8, "length": 74.99, "gap": 4.6, "dis": (0.118, 4.2),
      "history": ["greedy edge + 2-opt (2.8 s, 4.6 %)"], "unavailable": []},
     "Time left: 2.2 s.\nlkh_whole needs at least 2.6 s (NN5) or 2.7 s (QUADRANT5); triage and windows_model need "
     "at least 3.8 s.\nNothing fits.",
     {"tool": "stop", "args": {}, "reason": "No action fits in the 2.2 s left."}),
    ({"n": 10000, "budget": 25.0, "used": 2.8, "length": 74.99, "gap": 4.6, "dis": (0.118, 4.2),
      "history": ["greedy edge + 2-opt (2.8 s, 4.6 %)"], "unavailable": ["lkh_whole"]},
     "Time left: 22.2 s. lkh_whole is unavailable.\nFits: triage 10/20/50/100 (4.36/4.21/3.98/3.85 %), "
     "windows_model greedy (4.45 %) and beam10 (4.19 %).\nLowest expected gap: triage frac 100, 3.85 % in 20.1 s.",
     {"tool": "triage", "args": {"frac": 100}, "reason": "Best expected gap among the available actions that fit."}),
]


def system_v2():
    parts = [SYSTEM_V1, "", "Solved examples:"]
    for i, (state, reasoning, answer) in enumerate(EXAMPLES, 1):
        parts += ["", f"### Example {i}", probe.report_text(state), "", "Answer:", reasoning, json.dumps(answer)]
    return "\n".join(parts)


SYSTEM_V3 = """You are the planner of a TSP solver. You receive the state of one instance and the list of the
actions that fit in the time left, with their expected gap (% above optimal) afterwards. Goal: the shortest tour.
Choose exactly one action from the list and copy its tool and args unchanged.
Answer with one JSON object and nothing else: {"tool": "<tool>", "args": {...}, "reason": "<one sentence>"}."""


def report_v3(case):
    left = case["budget"] - case["used"]
    cur = "none yet" if case["length"] is None else f"length {case['length']:.2f}, estimated gap {case['gap']:g} %"
    lines = [f"n = {case['n']} cities. Time left: {left:.2f} s. Current tour: {cur}.", "", "Actions that fit:"]
    acts = feasible(case, ranked=False)                         # not sorted: the arg-min is left to the model
    for i, (tool, args, t, g) in enumerate(acts, 1):
        lines.append(f"{i}. tool {tool}, args {json.dumps(args)}: about {t:g} s, expected gap {g:g} %")
    lines.append(f"{len(acts) + 1}. tool stop, args {{}}: 0 s, keeps the current tour")
    return "\n".join(lines)


def extract_json(content):
    """The last JSON object in the text that has a "tool" key (V1/V2 put reasoning before it)."""
    dec, found = json.JSONDecoder(), None
    for i, ch in enumerate(content):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(content[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "tool" in obj:
                found = obj
    return json.dumps(found) if found is not None else content


VARIANTS = {   # name: (system prompt, user text, format json?)
    "V1": (lambda: SYSTEM_V1, probe.report_text, False),
    "V2": (system_v2, probe.report_text, False),
    "V3": (lambda: SYSTEM_V3, report_v3, True),
}


def chat(host, model, system, user, fmt_json, timeout=600):
    body = {"model": model, "stream": False, "keep_alive": "10m",
            "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 512},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if fmt_json:
        body["format"] = "json"
    req = probe.urllib.request.Request(f"http://{host}/api/chat", data=json.dumps(body).encode(),
                                       headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with probe.urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.perf_counter() - start


def run_rule(repeats):
    res = {}
    for c in CASES:
        start = time.perf_counter()
        for _ in range(repeats):
            ans = rule(c)
        sec = (time.perf_counter() - start) / repeats
        r = probe.score(c, json.dumps(ans))
        res[str(c["id"])] = {"correct": repeats * r["ok"], "latency_mean": sec, "typical":
                             f"{ans['tool']} {json.dumps(ans['args'], sort_keys=True)}", "errors": []}
    return res


def run_llm(host, variant, model, repeats, calls):
    system_fn, user_fn, fmt_json = VARIANTS[variant]
    system = system_fn()
    _, load_s = chat(host, model, system, user_fn(CASES[0]), fmt_json)      # loads the model; not counted
    gpu = probe.gpu_memory()
    res, lat, ptok, otok = {}, [], [], []
    for c in CASES:
        rs = []
        for rep in range(repeats):
            out, sec = chat(host, model, system, user_fn(c), fmt_json)
            content = out["message"]["content"]
            r = probe.score(c, extract_json(content))
            r.update({"variant": variant, "model": model, "case": c["id"], "rep": rep, "seconds": sec,
                      "content": content, "prompt_tokens": out.get("prompt_eval_count"),
                      "output_tokens": out.get("eval_count")})
            calls.write(json.dumps(r) + "\n")
            calls.flush()
            rs.append(r)
            lat.append(sec)
            ptok.append(out.get("prompt_eval_count") or 0)
            otok.append(out.get("eval_count") or 0)
        answers = [f"{r['tool']} {json.dumps(r['args'], sort_keys=True)}" for r in rs]
        typical = max(set(answers), key=answers.count)
        res[str(c["id"])] = {"correct": sum(r["ok"] for r in rs), "latency_mean": sum(r["seconds"] for r in rs) / len(rs),
                             "typical": typical.replace("|", "/"), "errors": [r["error"] for r in rs if r["error"]]}
        print(f"  {variant} {model} case {c['id']}: {res[str(c['id'])]['correct']}/{repeats}  {typical}", flush=True)
    stats = {"load_seconds": load_s, "gpu": gpu, "latency_mean": sum(lat) / len(lat),
             "prompt_tokens_mean": sum(ptok) / len(ptok), "output_tokens_mean": sum(otok) / len(otok)}
    return res, stats


def build_md(out, summary):
    reps = summary["meta"]["repeats"]
    notes = os.path.join(out, "planner_probe_v2_notes.md")
    notes = open(notes).read().strip() if os.path.exists(notes) else f"(notes missing: {notes})"
    rows = summary["rows"]                      # [{"variant", "model", "results", "stats"}]
    lines = ["# Phase 5, step 0b: prompt variants for the LLM planner vs. the fixed rule", "",
             f"Generated by `scripts/planner_probe_v2.py` on {summary['meta']['date']}.", "", notes, "",
             "## Summary", "",
             "| Variant | Model | Correct / 40 | Mean latency per call (s) | Prompt tokens | Output tokens |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        s = r["stats"]
        total = sum(v["correct"] for v in r["results"].values())
        lat = f"{s['latency_mean']:.2f}" if s["latency_mean"] >= 0.01 else f"{1e6 * s['latency_mean']:.0f} µs"
        tok = lambda k: "—" if s.get(k) is None else f"{s[k]:.0f}"
        lines.append(f"| {r['variant']} | {r['model']} | {total} / {reps * len(CASES)} | {lat} | "
                     f"{tok('prompt_tokens_mean')} | {tok('output_tokens_mean')} |")
    lines.append(f"| V1 | API model (upper bound) | {summary['meta']['api_model']} | — | — | — |")
    lines += ["", "## Per case (correct / 5, most frequent answer)", "",
              "| Case | Expected | " + " | ".join(f"{r['variant']} {r['model']}" for r in rows) + " |",
              "|---|---|" + "---|" * len(rows)]
    for c in CASES:
        cells = [f"{r['results'][str(c['id'])]['correct']}: {r['results'][str(c['id'])]['typical']}" for r in rows]
        lines.append(f"| {c['id']} | {c['expected']} | " + " | ".join(cells) + " |")
    lines += ["", "V0: the planner_probe.py prompt (format = json, context 4096). V1-V3: num_ctx 8192, num_predict "
              "512, temperature 0, seed 0; V1/V2 without format = json (reasoning first; the last JSON object with "
              "a \"tool\" key is scored). V3 hands the model the list of the actions that fit with their expected "
              "gap (in the order of the effects table, not sorted), so it already contains about 80 % of the rule: only the arg-min is left to the model. R: the "
              "rule in Python (latency: per decision, CPU). Latency: wall-clock per call with the model loaded. "
              "Same 8 cases, checks and 5 repeats as planner_probe.md."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Phase 5 step 0b: planner prompt variants vs. the fixed rule.")
    parser.add_argument("--variants", nargs="+", default=["V1", "V2", "V3", "R"])
    parser.add_argument("--models", nargs="+", default=["qwen2.5:7b", "qwen2.5:3b"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--host", default="127.0.0.1:11435")
    parser.add_argument("--out", default="results/phase5")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--show", default=None, help="print the system prompt and case 1 of one variant and exit")
    args = parser.parse_args()
    if args.show:
        s, u, _ = VARIANTS[args.show]
        print(s() + "\n\n" + u(CASES[0]))
        return
    path = os.path.join(args.out, "planner_probe_v2.json")
    if args.report_only:
        with open(path) as f:
            summary = json.load(f)
    else:
        if args.host.endswith(":11434"):
            parser.error("11434 is the shared Ollama: use the private instance (scripts/ollama_pc.sh)")
        commit, dirty = git_info()
        api = [k for k in API_KEYS if os.environ.get(k)]
        summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                            "repeats": args.repeats, "host": args.host,
                            "api_model": "not tested (no API key in the environment)" if not api else
                            f"key found ({api[0]}) but not run"}, "rows": []}
        with open(os.path.join(args.out, "planner_probe.json")) as f:                 # V0, from planner_probe.py
            v0 = json.load(f)
        for m in v0["meta"]["models"]:
            summary["rows"].append({"variant": "V0", "model": m, "results": v0["results"][m], "stats": v0["models"][m]})
        with open(os.path.join(args.out, "planner_probe_v2_calls.jsonl"), "w") as calls:
            for v in args.variants:
                if v == "R":
                    res = run_rule(args.repeats)
                    lat = sum(x["latency_mean"] for x in res.values()) / len(res)
                    summary["rows"].append({"variant": "R", "model": "rule (Python)", "results": res,
                                            "stats": {"latency_mean": lat}})
                    print(f"  R: {sum(x['correct'] for x in res.values())}/{args.repeats * len(CASES)}", flush=True)
                    continue
                for m in args.models:
                    res, stats = run_llm(args.host, v, m, args.repeats, calls)
                    summary["rows"].append({"variant": v, "model": m, "results": res, "stats": stats})
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "planner_probe_v2.md"), "w") as f:
        f.write(build_md(args.out, summary))
    print(f"saved to {args.out}/planner_probe_v2.md")


if __name__ == "__main__":
    main()
