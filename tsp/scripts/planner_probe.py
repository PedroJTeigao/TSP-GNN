"""Phase 5, step 0: can a local LLM pick the next tool from a written state report? (no tool calling yet)

Eight hand-written calibration reports (n, time budget, time used, current tour, network disagreement, tool history and
the table of expected effects of each tool at that size, taken from the phase 3 triage results and the benchmark E1/E3)
are sent to a private Ollama instance (scripts/ollama_pc.sh, port 11435) through /api/chat with format = json and
temperature 0. The answer must be {"tool", "args", "reason"} and is checked against the expected decision of the case.
Each case runs --repeats times per model; accuracy, wall-clock latency per call and tokens are recorded.

Writes <out>/planner_probe.json (summary), <out>/planner_probe_calls.jsonl (every raw answer) and
<out>/planner_probe.md (case x model table; the one-line verdict is read from <out>/planner_probe_notes.md).

Example:
    python scripts/planner_probe.py --models qwen2.5:7b qwen2.5:3b --repeats 5
"""

import argparse
import datetime
import json
import os
import subprocess
import time
import urllib.request

from tsp_gnn.utils import git_info

TOOLS = ("lkh_whole", "triage", "windows_model", "stop")
CANDIDATES = ("QUADRANT5", "NN5", "NN8", "ALPHA", "POPMUSIC")

SYSTEM = """You are the planner of a TSP solver. You receive a report on one instance and choose the NEXT action.
Goal: the shortest tour when the total time budget runs out. The total time (time already used + the time of the
next action) must never exceed the budget; an action that does not fit gives no tour at all.

Tools (all of them start from the current tour, which already exists unless the report says otherwise):
- lkh_whole: run LKH-3 on the whole instance. args: {"candidates": one of "QUADRANT5", "NN5", "NN8", "ALPHA",
  "POPMUSIC"; "time_limit": seconds this action may use in total, setup included}.
- triage: cut the tour into windows of 20 cities and run LKH only on the top frac % of windows by network
  disagreement. args: {"frac": 10, 20, 50 or 100}.
- windows_model: re-solve every window with the graph network. args: {"decoder": "greedy" or "beam10"}.
- stop: return the current tour. args: {}.
A tool marked "unavailable" cannot be used.

Answer with one JSON object and nothing else: {"tool": "<tool>", "args": {...}, "reason": "<one sentence>"}."""

# Expected effects from the current greedy edge + 2-opt tour: (tool, args, extra seconds, gap % afterwards).
# 10k and 1k: Fu et al. standard sets (benchmark E3 and E1, "ours" start = our tour, QUADRANT 5 / NN / ALPHA from the
# E1 tables and e1_prep fixed costs); 200: phase 3 triage report (our tsp200 test set). Times are wall-clock per
# instance on this server; "extra" excludes the initial tour.
EFFECTS = {
    10000: {"initial": (2.8, 4.6), "rows": [
        ("lkh_whole", "QUADRANT5", "setup 1.3 s; after 1 trial (2.7 s) 0.39 %; 5.9 s 0.20 %; 11.7 s 0.12 %; 19.0 s 0.085 %"),
        ("lkh_whole", "NN5", "setup 0.9 s; after 1 trial (2.6 s) 0.45 %; 6.2 s 0.33 %; 11.9 s 0.22 %; 19.2 s 0.16 %"),
        ("lkh_whole", "NN8", "setup 0.9 s; after 1 trial (7.3 s) 0.22 %; 13.4 s 0.18 %; 20.1 s 0.14 %"),
        ("lkh_whole", "ALPHA", "setup 138 s (does not fit below 140 s); then 0.20 % after 1 trial"),
        ("lkh_whole", "POPMUSIC", "setup 51 s (does not fit below 53 s); then 0.16 % after 1 trial"),
        ("triage", "frac 10", "3.8 s, 4.36 %"),
        ("triage", "frac 20", "6.3 s, 4.21 %"),
        ("triage", "frac 50", "12.4 s, 3.98 %"),
        ("triage", "frac 100", "20.1 s, 3.85 %"),
        ("windows_model", "greedy", "3.8 s, 4.45 %"),
        ("windows_model", "beam10", "15.7 s, 4.19 %"),
    ]},
    1000: {"initial": (0.26, 4.8), "rows": [
        ("lkh_whole", "QUADRANT5", "setup 0.05 s; after 1 trial (0.16 s) 0.29 %; 0.48 s 0.11 %; 1.01 s 0.053 %; 1.71 s 0.035 %"),
        ("lkh_whole", "NN5", "setup 0.03 s; after 1 trial (0.24 s) 0.45 %; 0.47 s 0.26 %; 1.01 s 0.14 %; 1.70 s 0.105 %"),
        ("lkh_whole", "NN8", "setup 0.03 s; after 1 trial (0.39 s) 0.18 %; 1.09 s 0.11 %; 1.76 s 0.060 %"),
        ("lkh_whole", "ALPHA", "setup 1.3 s; after 1 trial (1.34 s) 0.16 %; 1.90 s 0.050 %"),
        ("lkh_whole", "POPMUSIC", "setup 0.76 s; 1.12 s 0.064 %; 1.80 s 0.010 %"),
        ("triage", "frac 10", "0.39 s, 4.51 %"),
        ("triage", "frac 20", "0.54 s, 4.34 %"),
        ("triage", "frac 50", "1.12 s, 4.08 %"),
        ("triage", "frac 100", "1.76 s, 3.92 %"),
        ("windows_model", "greedy", "0.30 s, 4.65 %"),
        ("windows_model", "beam10", "1.34 s, 4.34 %"),
    ]},
    200: {"initial": (0.06, 4.8), "rows": [
        ("lkh_whole", "any", "not measured at this size (at n = 1000: QUADRANT5 setup 0.05 s, 1 trial 0.16 s, 0.29 %)"),
        ("triage", "frac 10", "0.07 s, 4.31 %"),
        ("triage", "frac 20", "0.11 s, 4.10 %"),
        ("triage", "frac 50", "0.20 s, 3.72 %"),
        ("triage", "frac 100", "0.28 s, 3.54 %"),
        ("windows_model", "beam10", "0.24 s, 4.12 %"),
    ]},
}


def in_range(lo, hi):
    return lambda a: isinstance(a.get("time_limit"), (int, float)) and lo <= a["time_limit"] <= hi


def lkh(cands, lo, hi):
    """Expected: lkh_whole with one of cands and lo <= time_limit <= hi (0.7-1.0 x the time left)."""
    tl = in_range(lo, hi)
    return lambda t, a: t == "lkh_whole" and a.get("candidates") in cands and tl(a)


def triage_frac(*fracs):
    return lambda t, a: t == "triage" and a.get("frac") in fracs


STOP = lambda t, a: t == "stop"


def either(*checks):
    return lambda t, a: any(c(t, a) for c in checks)


# Each case: id, state, expected decision (text), check(tool, args) -> bool.
CASES = [
    {"id": 1, "n": 10000, "budget": 8.4, "used": 3.1, "length": 74.99, "gap": 4.5, "dis": (0.118, 4.2),
     "history": ["greedy edge + 2-opt (3.1 s, 4.5 %)"], "unavailable": [],
     "expected": "lkh_whole QUADRANT5 or NN5, time_limit ≈ 5 (3.7–5.3)",
     "check": lkh(("QUADRANT5", "NN5"), 3.7, 5.3)},
    {"id": 2, "n": 10000, "budget": 30.0, "used": 3.1, "length": 74.99, "gap": 4.5, "dis": (0.118, 4.2),
     "history": ["greedy edge + 2-opt (3.1 s, 4.5 %)"], "unavailable": [],
     "expected": "lkh_whole QUADRANT5, time_limit ≈ 25 (18.8–26.9)",
     "check": lkh(("QUADRANT5",), 18.8, 26.9)},
    {"id": 3, "n": 10000, "budget": 1.0, "used": 0.0, "length": None, "gap": None, "dis": None,
     "history": [], "unavailable": [],
     "expected": "stop (nothing fits)", "check": STOP},
    {"id": 4, "n": 10000, "budget": 8.4, "used": 8.0, "length": 71.91, "gap": 0.2, "dis": (0.118, 4.2),
     "history": ["greedy edge + 2-opt (3.1 s, 4.5 %)", "lkh_whole QUADRANT5 time_limit 4.9 (4.9 s, 0.2 %)"],
     "unavailable": [], "expected": "stop", "check": STOP},
    {"id": 5, "n": 1000, "budget": 0.72, "used": 0.29, "length": 24.23, "gap": 4.8, "dis": (0.121, 4.6),
     "history": ["greedy edge + 2-opt (0.29 s, 4.8 %)"], "unavailable": [],
     "expected": "lkh_whole NN8 or QUADRANT5, time_limit ≈ 0.4 (0.30–0.43)",
     "check": lkh(("NN8", "QUADRANT5"), 0.30, 0.43)},
    {"id": 6, "n": 1000, "budget": 1.94, "used": 0.29, "length": 24.23, "gap": 4.8, "dis": (0.121, 4.6),
     "history": ["greedy edge + 2-opt (0.29 s, 4.8 %)"], "unavailable": [],
     "expected": "lkh_whole QUADRANT5, time_limit ≈ 1.6 (1.15–1.65)",
     "check": lkh(("QUADRANT5",), 1.15, 1.65)},
    {"id": 7, "n": 200, "budget": 0.1, "used": 0.06, "length": 11.25, "gap": 4.8, "dis": (0.130, 5.1),
     "history": ["greedy edge + 2-opt (0.06 s, 4.8 %)"], "unavailable": [],
     "expected": "triage frac 20 or stop", "check": either(triage_frac(20), STOP)},
    {"id": 8, "n": 10000, "budget": 15.0, "used": 2.5, "length": 74.99, "gap": 4.5, "dis": (0.118, 4.2),
     "history": ["greedy edge + 2-opt (2.5 s, 4.5 %)"], "unavailable": ["lkh_whole"],
     "expected": "triage frac 50 or windows_model",
     "check": either(triage_frac(50), lambda t, a: t == "windows_model")},
]


def report_text(case):
    """The state report the model reads."""
    n, eff = case["n"], EFFECTS[case["n"]]
    left = case["budget"] - case["used"]
    lines = [f"Instance: random uniform TSP, n = {n} cities.",
             f"Time budget: {case['budget']:g} s in total. Used so far: {case['used']:g} s. Left: {left:.2f} s."]
    if case["length"] is None:
        lines.append(f"Current tour: none yet. Every tool needs the greedy edge + 2-opt tour first, which takes "
                     f"{eff['initial'][0]:g} s at this size (gap ≈ {eff['initial'][1]:g} %).")
    else:
        lines.append(f"Current tour: length {case['length']:.2f}, estimated gap to optimal {case['gap']:g} %.")
    if case["dis"] is not None:
        lines.append(f"Network disagreement over the windows of 20 cities: mean {case['dis'][0]:.3f}, "
                     f"{case['dis'][1]:.1f} % of windows above 0.35.")
    lines.append("Tool history: " + ("; ".join(case["history"]) if case["history"] else "none") + ".")
    lines.append("")
    lines.append(f"Expected effect of each tool at n = {n}, measured from the greedy edge + 2-opt tour "
                 "(extra seconds on top of the time used; gap % afterwards):")
    for tool, arg, eff_txt in eff["rows"]:
        status = " [unavailable]" if tool in case["unavailable"] else ""
        lines.append(f"- {tool} ({arg}){status}: {eff_txt}")
    lines.append("- stop: 0 s, keeps the current gap")
    return "\n".join(lines)


def chat(host, model, user, timeout=600):
    body = {"model": model, "stream": False, "format": "json", "keep_alive": "10m",
            "options": {"temperature": 0, "seed": 0},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    req = urllib.request.Request(f"http://{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.perf_counter() - start


def parse(content):
    """(tool, args, reason, error): error is None when the answer is a valid {"tool","args","reason"} object."""
    try:
        d = json.loads(content)
    except json.JSONDecodeError:
        return None, None, None, "not JSON"
    if not isinstance(d, dict) or not {"tool", "args", "reason"} <= set(d):
        return None, None, None, "missing keys"
    args = d["args"] if isinstance(d["args"], dict) else None
    if d["tool"] not in TOOLS or args is None:
        return d["tool"], args, d["reason"], "unknown tool or args not an object"
    return d["tool"], args, d["reason"], None


def score(case, content):
    tool, args, reason, err = parse(content)
    ok = err is None and bool(case["check"](tool, args))
    return {"tool": tool, "args": args, "reason": reason, "error": err, "ok": ok}


def gpu_memory():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in out.splitlines() if "ollama" in line]


def summary_md(summary, notes):
    models = summary["meta"]["models"]
    reps = summary["meta"]["repeats"]
    lines = ["# Phase 5, step 0: planner probe (local LLM, no tools yet)", "",
             f"Generated by `scripts/planner_probe.py` on {summary['meta']['date']}.", "", notes, "",
             "| Case | n | Budget (s) | Used (s) | Expected | " +
             " | ".join(f"{m}: correct / {reps} | {m}: latency (s) | {m}: answer" for m in models) + " |",
             "|---|---|---|---|---|" + "---|---|---|" * len(models)]
    for c in CASES:
        row = f"| {c['id']} | {c['n']} | {c['budget']:g} | {c['used']:g} | {c['expected']} |"
        for m in models:
            r = summary["results"][m][str(c["id"])]
            row += f" {r['correct']} | {r['latency_mean']:.2f} | {r['typical']} |"
        lines.append(row)
    total = " | ".join(f"{m}: {sum(summary['results'][m][str(c['id'])]['correct'] for c in CASES)} / "
                       f"{reps * len(CASES)}, {summary['models'][m]['latency_mean']:.2f} s per call, "
                       f"{summary['models'][m]['prompt_tokens_mean']:.0f} prompt + "
                       f"{summary['models'][m]['output_tokens_mean']:.0f} output tokens, "
                       f"load {summary['models'][m]['load_seconds']:.1f} s"
                       for m in models)
    lines += ["", f"**Total:** {total}.", "",
              "Each case was sent " + f"{reps} times per model (/api/chat, format = json, temperature 0, seed 0, "
              "context 4096, KV cache q8_0). \"Answer\": the most frequent (tool, args) of the repeats. Latency: "
              "wall-clock per call with the model already loaded (the first, loading call is not counted). "
              "time_limit must lie in 0.7–1.0 × the time left (setup included). The expected-effect tables in the "
              "reports come from the phase 3 triage report (n = 200) and the benchmark E1/E3 (n = 1000, 10000).", "",
              "GPU memory with the model loaded (nvidia-smi, MiB): " +
              "; ".join(f"{m}: {summary['models'][m]['gpu']}" for m in models) + "."]
    return "\n".join(lines) + "\n"


def write_md(out, summary):
    notes = os.path.join(out, "planner_probe_notes.md")
    notes = open(notes).read().strip() if os.path.exists(notes) else f"(notes missing: {notes})"
    with open(os.path.join(out, "planner_probe.md"), "w") as f:
        f.write(summary_md(summary, notes))


def main():
    parser = argparse.ArgumentParser(description="Phase 5 step 0: LLM planner probe on hand-written reports.")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:7b", "qwen2.5:3b"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--host", default="127.0.0.1:11435")
    parser.add_argument("--out", default="results/phase5")
    parser.add_argument("--show-report", type=int, default=None, help="print the report of one case and exit")
    parser.add_argument("--report-only", action="store_true", help="rewrite planner_probe.md from planner_probe.json")
    args = parser.parse_args()
    if args.show_report is not None:
        print(SYSTEM + "\n\n" + report_text(next(c for c in CASES if c["id"] == args.show_report)))
        return
    if args.report_only:
        with open(os.path.join(args.out, "planner_probe.json")) as f:
            write_md(args.out, json.load(f))
        return
    if args.host.endswith(":11434"):
        parser.error("11434 is the shared Ollama: use the private instance (scripts/ollama_pc.sh)")
    os.makedirs(args.out, exist_ok=True)
    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                        "models": args.models, "repeats": args.repeats, "host": args.host},
               "results": {}, "models": {}}
    calls = open(os.path.join(args.out, "planner_probe_calls.jsonl"), "w")
    for m in args.models:
        warm, load_s = chat(args.host, m, report_text(CASES[0]))            # loads the model; not counted
        gpu = gpu_memory()
        print(f"{m}: loaded in {load_s:.1f} s, GPU {gpu}", flush=True)
        lat, ptok, otok = [], [], []
        summary["results"][m] = {}
        for c in CASES:
            user = report_text(c)
            rs = []
            for rep in range(args.repeats):
                out, sec = chat(args.host, m, user)
                content = out["message"]["content"]
                r = score(c, content)
                r.update({"model": m, "case": c["id"], "rep": rep, "seconds": sec, "content": content,
                          "prompt_tokens": out.get("prompt_eval_count"), "output_tokens": out.get("eval_count")})
                calls.write(json.dumps(r) + "\n")
                calls.flush()
                rs.append(r)
                lat.append(sec)
                ptok.append(out.get("prompt_eval_count") or 0)
                otok.append(out.get("eval_count") or 0)
            answers = [f"{r['tool']} {json.dumps(r['args'], sort_keys=True)}" for r in rs]
            typical = max(set(answers), key=answers.count)
            summary["results"][m][str(c["id"])] = {
                "correct": sum(r["ok"] for r in rs), "latency_mean": sum(r["seconds"] for r in rs) / len(rs),
                "typical": typical.replace("|", "/"), "errors": [r["error"] for r in rs if r["error"]]}
            print(f"  case {c['id']}: {sum(r['ok'] for r in rs)}/{args.repeats}  {typical}", flush=True)
        summary["models"][m] = {"load_seconds": load_s, "gpu": gpu,
                                "latency_mean": sum(lat) / len(lat),
                                "prompt_tokens_mean": sum(ptok) / len(ptok), "output_tokens_mean": sum(otok) / len(otok)}
    calls.close()
    with open(os.path.join(args.out, "planner_probe.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_md(args.out, summary)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {args.out}/planner_probe.md")


if __name__ == "__main__":
    main()
