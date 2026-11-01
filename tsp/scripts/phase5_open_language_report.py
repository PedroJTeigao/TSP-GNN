"""Report of phase 5, step 3: docs/phase5_open_language_report.md, from results/phase5/open_language/.

The hand-written parts (conclusion, design notes, caveats) are read from results/phase5/open_language/notes.md, with the
sections "## Conclusion" and "## Caveats". Everything else is computed here.

Example:
    python scripts/phase5_open_language_report.py
"""

import datetime
import importlib.util
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest

from tsp_gnn import formalize as F
from tsp_gnn import open_dsl as D
from tsp_gnn import open_eval as E

OUT = "results/phase5/open_language"
NS = (200, 1000)
ARMS = ("p0", "p1", "q7", "q14", "q14t", "g12")
NAME = {"oracle": "o oracle", "p0": "p0 old keywords", "p1": "p1 DSL parser", "q7": "q7 qwen2.5:7b", "q14": "q14 qwen3:14b",
        "q14t": "q14t qwen3:14b, thinking on",
        "g12": "g12 gemma3:12b"}
ERRORS = ("missing", "wrong selector", "exception lost", "inverted", "wrong type", "wrong times")


def jl(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def wilson(k, m, z=1.96):
    if m == 0:
        return float("nan"), float("nan")
    p = k / m
    d = 1 + z * z / m
    c = (p + z * z / (2 * m)) / d
    h = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
    return 100 * (c - h), 100 * (c + h)


def pc(k, m):
    lo, hi = wilson(k, m)
    return f"{100 * k / m:.1f} [{lo:.1f}, {hi:.1f}]" if m else "—"


def score_all():
    """{(n, level, arm): [record]} with per-rule hits, instance hit and error classes."""
    out = defaultdict(list)
    for n in NS:
        insts = {(i["idx"], lv): i for lv in E.LEVELS for i in E.load_level(n, lv)}
        for e in jl(f"{OUT}/formalize_tsp{n}.jsonl"):
            inst = insts.get((e["idx"], e["level"]))
            if inst is None:
                continue
            prims = [p for r in e["pred"] for p in r["prims"]]
            s = D.score(inst["label"], prims)
            errs = [D.classify_error(r, prims, inst["ctx"]) for r, ok in zip(inst["label"], s["correct"]) if not ok]
            # wrong primitives that touch no label rule: invented; from a distractor if they name its places
            label = D.keyset(E.label_prims(inst))
            touched = set().union(*[D._key_customers(k) for k in label]) if label else set()
            names = " ".join(inst["background"]).lower()
            bg = {p["id"] for p in inst["legend"] if p["name"] and p["name"].lower() in names}
            invented = [k for k in D.keyset(prims) - label if not D._key_customers(k) & touched]
            distract = sum(bool(D._key_customers(k) & bg) for k in invented)
            out[(n, e["level"], e["arm"])].append({
                "idx": e["idx"], "correct": s["correct"], "exact": s["exact"], "errors": errs, "invented": len(invented),
                "from_distractor": distract, "seconds": e["seconds"], "attempts": e["attempts"],
                "valid_json": e["valid_json"], "schema_errors": len(e["errors"]), "lkh_running": e.get("lkh_running", 0),
                "kinds": [r["dsl"].get("type") for r in inst["label"]],
                "shapes": [r.get("shape", "L0") for r in inst["label"]]})
    return out


def filtered(rows, n, level, excluded):
    return [r for r in rows if not (level == "L2" and (n, r["idx"]) in excluded)]


def alone():
    """{(n, level, arm): [seconds]} of the sample measured with no LKH of mine running (scripts/open_latency.py)."""
    out = defaultdict(list)
    for r in jl(f"{OUT}/latency_alone.jsonl"):
        out[(r["n"], r["level"], r["arm"])].append(r["seconds"])
    return out


def alone_cell(n, lv, arm):
    v = alone().get((n, lv, arm))
    return f"{np.mean(v):.2f}" if v else ("~0" if arm in ("p0", "p1") else "—")


def table_latency():
    rows = jl(f"{OUT}/latency_alone.jsonl")
    lines = ["Measured alone, last in the run, with no LKH of mine running (`lkh_running` was checked before every call); each "
             "arm started at 70 C or less, and the temperature and clock were read after every call.", "",
             "| arm | calls | mean (s) | median (s) | p95 (s) | tokens/s | load average at the calls | calls with LKH running | "
             "temperature (C) | clock (MHz, min–max) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ("q7", "q14", "g12"):
        v = [r for r in rows if r["arm"] == arm]
        if v:
            s = [r["seconds"] for r in v]
            lines.append(f"| {NAME[arm]} | {len(v)} | {np.mean(s):.2f} | {np.median(s):.2f} | {np.percentile(s, 95):.2f} | "
                         f"{np.mean([r['tok_s'] for r in v]):.1f} | {min(r['load1'] for r in v):.2f}–"
                         f"{max(r['load1'] for r in v):.2f} | {sum(r['lkh_running'] > 0 for r in v)} | "
                         f"{min(r['temp_c'] for r in v)}–{max(r['temp_c'] for r in v)} | "
                         f"{min(r['clock_mhz'] for r in v)}–{max(r['clock_mhz'] for r in v)} |")
    return "\n".join(lines)


def thermal():
    rows = [x.split(", ") for x in open(f"{OUT}/thermal.csv")]
    busy = [r for r in rows if int(r[4]) > 50]
    low = sum(int(r[2]) < 500 for r in busy)
    hot = sum("Active" == r[5].strip() for r in busy)
    return (f"GPU heat: {len(busy)} samples (one every 30 s) with the GPU in use, from {rows[0][0]} on; at most {max(int(r[1]) for r in rows)} C; "
            f"clock under 500 MHz (maximum 1590) in {100 * low / len(busy):.0f} % of them, median "
            f"{int(np.median([int(r[2]) for r in busy]))} MHz; software thermal slowdown flagged in {100 * hot / len(busy):.0f} %.")


def table_b(S, excluded=frozenset()):
    lines = ["| n | level | arm | instances | rules exact % [95 % CI] | instances exact % [95 % CI] | invalid JSON after retry % | "
             "mean time alone (s) | mean time of the run (s), * = under LKH load |", "|---|---|---|---|---|---|---|---|---|"]
    for n in NS:
        for lv in E.LEVELS:
            for arm in ARMS:
                rows = filtered(S.get((n, lv, arm), []), n, lv, excluded)
                if not rows:
                    continue
                k = sum(sum(r["correct"]) for r in rows)
                m = sum(len(r["correct"]) for r in rows)
                ex = sum(r["exact"] for r in rows)
                bad = sum(not r["valid_json"] for r in rows)
                lines.append(f"| {n} | {lv} | {NAME[arm]} | {len(rows)} | {pc(k, m)} | {pc(ex, len(rows))} | "
                             f"{100 * bad / len(rows):.1f} | {alone_cell(n, lv, arm)} | {np.mean([r['seconds'] for r in rows]):.2f}"
                             f"{'*' if any(r['lkh_running'] for r in rows) else ''} |")
    return "\n".join(lines)


def mcnemar(S, a, b, excluded=frozenset()):
    lines = [f"| n | level | {a} right, {b} wrong | {a} wrong, {b} right | exact McNemar p |", "|---|---|---|---|---|"]
    for n in NS:
        for lv in E.LEVELS:
            ra = {r["idx"]: r for r in filtered(S.get((n, lv, a), []), n, lv, excluded)}
            rb = {r["idx"]: r for r in filtered(S.get((n, lv, b), []), n, lv, excluded)}
            x = y = 0
            for i in ra.keys() & rb.keys():
                for p, q in zip(ra[i]["correct"], rb[i]["correct"]):
                    x += p and not q
                    y += q and not p
            if ra and rb:
                pv = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
                lines.append(f"| {n} | {lv} | {x} | {y} | {pv:.3g} |")
    return "\n".join(lines)


def table_errors(S):
    head = list(ERRORS) + ["invented, from a distractor", "invented, other"]
    lines = ["Counts over both n. The first six columns count label rules missed; the last two count primitive constraints "
             "predicted that touch no label rule.", "", "| level | arm | rules | " + " | ".join(head) + " |",
             "|---|---|---|" + "---|" * len(head)]
    for lv in E.LEVELS:
        for arm in ARMS:
            rows = [r for n in NS for r in S.get((n, lv, arm), [])]
            if not rows:
                continue
            c = Counter(x for r in rows for x in r["errors"])
            inv = sum(r["invented"] for r in rows)
            dis = sum(r["from_distractor"] for r in rows)
            cells = [str(c.get(e, 0)) for e in ERRORS] + [str(dis), str(inv - dis)]
            lines.append(f"| {lv} | {NAME[arm]} | {sum(len(r['correct']) for r in rows)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_shapes(S):
    shapes = sorted({s for k, v in S.items() if k[1] == "L1" for r in v for s in r["shapes"]})
    lines = ["Rules exact (%) by template on L2, both n (rule count in brackets on the first column).", "",
             "| shape | " + " | ".join(NAME[a] for a in ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for sh in shapes:
        cells, total = [], 0
        for arm in ARMS:
            hit = [ok for n in NS for r in S.get((n, "L2", arm), []) for ok, s in zip(r["correct"], r["shapes"]) if s == sh]
            total = max(total, len(hit))
            cells.append(f"{100 * sum(hit) / len(hit):.0f}" if hit else "—")
        lines.append(f"| {sh} ({total}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def llm_time(n, lv, arm, rows):
    """The latency measured alone if the arm has one; q14t has none, so its time in the run (no LKH was running)."""
    v = alone().get((n, lv, arm))
    return float(np.mean(v)) if v else float(np.mean([r["llm_seconds"] for r in rows]))


def table_c(excluded=frozenset()):
    lines = ["| n | level | arm | routes | valid routes % [95 % CI] | hard rules broken / route | length gap vs oracle (%) | "
             "cost gap vs oracle (%) | rules rejected by the code check / route | LKH-3 no route (%) | mean time LLM + LKH (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in NS:
        rows = jl(f"{OUT}/pipeline_tsp{n}.jsonl")
        for lv in E.LEVELS:
            for arm in ("oracle",) + ARMS:
                rs = [r for r in rows if r["level"] == lv and r["arm"] == arm and
                      not (lv == "L2" and (n, r["idx"]) in excluded)]
                if not rs:
                    continue
                v = sum(r["valid"] for r in rs)
                lines.append(f"| {n} | {lv} | {NAME[arm]} | {len(rs)} | {pc(v, len(rs))} | "
                             f"{np.mean([len(r['violated']) for r in rs]):.2f} | "
                             f"{100 * np.mean([r['length_gap'] for r in rs]):+.2f} | "
                             f"{100 * np.mean([r['cost_gap'] for r in rs]):+.2f} | "
                             f"{np.mean([len(r['rejected']) for r in rs]):.2f} | "
                             f"{100 * np.mean([not r['solver_feasible'] for r in rs]):.1f} | "
                             f"{llm_time(n, lv, arm, rs) + np.mean([r['lkh_seconds'] for r in rs]):.1f} |")
    return "\n".join(lines)


def mcnemar_routes(a, b):
    lines = [f"| n | level | routes | {a} valid, {b} not | {a} not, {b} valid | exact McNemar p |", "|---|---|---|---|---|---|"]
    for n in NS:
        rows = jl(f"{OUT}/pipeline_tsp{n}.jsonl")
        for lv in E.LEVELS:
            ra = {r["idx"]: r["valid"] for r in rows if r["level"] == lv and r["arm"] == a}
            rb = {r["idx"]: r["valid"] for r in rows if r["level"] == lv and r["arm"] == b}
            both = ra.keys() & rb.keys()
            x = sum(ra[i] and not rb[i] for i in both)
            y = sum(rb[i] and not ra[i] for i in both)
            if both:
                pv = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
                lines.append(f"| {n} | {lv} | {len(both)} | {x} | {y} | {pv:.3g} |")
    return "\n".join(lines)


def table_rejects():
    c = defaultdict(Counter)
    for n in NS:
        for r in jl(f"{OUT}/pipeline_tsp{n}.jsonl"):
            for x in r["rejected"]:
                c[r["arm"]][x["reason"]] += 1
    reasons = sorted({k for v in c.values() for k in v})
    if not reasons:
        return "No rule was rejected."
    lines = ["| arm | " + " | ".join(reasons) + " |", "|---|" + "---|" * len(reasons)]
    for arm in ARMS:
        if arm in c:
            lines.append(f"| {NAME[arm]} | " + " | ".join(str(c[arm].get(x, 0)) for x in reasons) + " |")
    return "\n".join(lines)


def table_step0():
    res = json.load(open(f"{OUT}/step0.json"))
    lines = ["| model, mode | size in memory (GB) | in GPU (%) | GPU used (MiB) | tokens/s | mean latency (s) | p95 (s) | "
             "rules exact / 20 instances |", "|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        lines.append(f"| {k} | {r['load_end']['size_gb']:.1f} | {r['load_end']['in_gpu_pct']:.0f} | {r['gpu_end']['used_mib']} | "
                     f"{r['tok_s']:.1f} | {r['lat_mean']:.1f} | {r['lat_p95']:.1f} | {r['rules_exact']} / {r['rules']} |")
    return "\n".join(lines)


def table_pauses():
    rows = jl(f"{OUT}/gpu_guard.jsonl")
    checks = sum(r["event"] == "check" for r in rows)
    pauses = [r for r in rows if r["event"] == "resume"]
    frees = [r["free"] for r in rows if r["event"] == "check"]
    total = sum(r["seconds"] for r in pauses)
    open_pause = sum(r["event"] == "pause" for r in rows) - len(pauses)
    txt = (f"{checks} checks; free memory at the checks {min(frees)}–{max(frees)} MiB; {len(pauses)} pauses, "
           f"{total / 60:.0f} min in total" + (f"; {open_pause} pause(s) cut short by a restart" if open_pause else "") + ".")
    if pauses:
        txt += "\n\n| resumed at | seconds | model | during |\n|---|---|---|---|\n" + \
            "\n".join(f"| {r['time']} | {r['seconds']:.0f} | {r['model']} | {r['label']} |" for r in pauses)
    return txt


def step3():
    spec = importlib.util.spec_from_file_location("open_step3", "scripts/open_step3.py")
    s3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s3)
    lines = ["| task | metric | 7B | 14B |", "|---|---|---|---|"]
    # planner
    p = jl(f"{OUT}/step3_planner.jsonl")
    v0 = json.load(open("results/phase5/planner_probe.json"))
    v2 = json.load(open("results/phase5/planner_probe_v2.json"))
    old = {r["variant"]: r for r in v2["rows"] if r["model"] == "qwen2.5:7b"}
    for var in ("V0", "V3"):
        rows = [r for r in p if r["variant"] == var]
        o = old[var]
        hits = sum(x["correct"] for x in o["results"].values())
        lines.append(f"| planner {var} | correct / 40 | {hits} | {sum(r['ok'] for r in rows)} |")
        lines.append(f"| planner {var} | mean latency (s) | {o['stats']['latency_mean']:.2f} | "
                     f"{np.mean([r['seconds'] for r in rows]):.2f} |")
    # formalisation on the same instances (b2 of step 2)
    f14 = jl(f"{OUT}/step3_formal.jsonl")
    for n in NS:
        rows = {e["idx"]: e for e in jl(f"data/phase5/text_rules_v2b_tsp{n}.jsonl")}
        old7 = {e["idx"]: e for e in F.load_preds(f"data/phase5/formalize_tsp{n}.jsonl", n)
                if e["arm"] == "b2" and e["text"] == "v2"}
        k7 = k14 = m = i7 = i14 = 0
        s7, s14 = [], []
        for e in (x for x in f14 if x["n"] == n):
            label = [{"type": r["type"], "formal": r["formal"]} for r in rows[e["idx"]]["rules"]]
            a, b = F.score(label, old7[e["idx"]]["pred"]), F.score(label, e["pred"])
            k7 += sum(a["correct"])
            k14 += sum(b["correct"])
            m += len(label)
            i7 += a["exact"]
            i14 += b["exact"]
            s7.append(old7[e["idx"]]["seconds"])
            s14.append(e["seconds"])
        if m:
            lines.append(f"| formalisation text_v2b, n = {n} | rules exact % ({m} rules) | {pc(k7, m)} | {pc(k14, m)} |")
            lines.append(f"| formalisation text_v2b, n = {n} | instances exact % ({len(s7)}) | {pc(i7, len(s7))} | {pc(i14, len(s7))} |")
            lines.append(f"| formalisation text_v2b, n = {n} | mean latency (s) | {np.mean(s7):.2f} | {np.mean(s14):.2f} |")
    # verifier on the same routes
    d14 = jl(f"{OUT}/step3_verify.jsonl")
    oldd = {(e["idx"], e["route"], e["variant"]): e for e in jl("data/phase5/llm_verify_tsp200.jsonl")}

    def stats(rows):
        tp = fp = fn = right = alarms = oracle = found = inj = 0
        for r in rows:
            truth, claim = set(r["truth"]), set(r["claimed"] or [])
            tp += len(truth & claim)
            fp += len(claim - truth)
            fn += len(truth - claim)
            right += truth == claim
            if r["route"] == "oracle":
                oracle += 1
                alarms += bool(claim)
            else:
                inj += 1
                found += bool(truth & claim)
        return tp, fp, fn, right, alarms, oracle, found, inj
    for var in ("plain", "report"):
        rows = [r for r in d14 if r["variant"] == var]
        o = [oldd[(r["idx"], r["route"], var)] for r in rows]
        for label, f in (("precision %", lambda s: 100 * s[0] / max(s[0] + s[1], 1)),
                         ("recall %", lambda s: 100 * s[0] / max(s[0] + s[2], 1)),
                         ("route verdict right %", lambda s: 100 * s[3] / max(len(rows), 1)),
                         ("false alarms on oracle routes %", lambda s: 100 * s[4] / max(s[5], 1)),
                         ("injected error found %", lambda s: 100 * s[6] / max(s[7], 1))):
            lines.append(f"| verifier {var} ({len(rows)} routes) | {label} | {f(stats(o)):.1f} | {f(stats(rows)):.1f} |")
        if rows:
            lines.append(f"| verifier {var} | mean latency (s) | {np.mean([r['seconds'] for r in o]):.2f} | "
                         f"{np.mean([r['seconds'] for r in rows]):.2f} |")
    return "\n".join(lines)


def audit():
    path = f"{OUT}/l2_audit.json"
    if not os.path.exists(path):
        return "(audit missing)", frozenset()
    a = json.load(open(path))
    items = a["items"]
    bad = [e for e in items if not e["ok"]]
    lo, hi = wilson(len(bad), len(items))
    txt = [f"Sample: {len(items)} L2 texts drawn at random ({a['seed_note']}). I read each one against its formal label.",
           f"**Label error rate: {len(bad)} / {len(items)} = {100 * len(bad) / len(items):.1f} % (95 % Wilson CI {lo:.1f}–{hi:.1f} %).**"]
    if bad:
        txt += ["", "| n | instance | what is wrong |", "|---|---|---|"] + \
            [f"| {e['n']} | {e['idx']} | {e['why']} |" for e in bad]
    return "\n".join(txt), frozenset((e["n"], e["idx"]) for e in bad)


def section(notes, name):
    parts = notes.split(f"## {name}\n", 1)
    return parts[1].split("\n## ", 1)[0].strip() if len(parts) == 2 else f"(notes missing: {name})"


def dataset():
    lines = ["| n | level | instances | label rules | primitive constraints per instance (mean) | prompt tokens (mean, q14) |",
             "|---|---|---|---|---|---|"]
    for n in NS:
        rec = jl(f"{OUT}/formalize_tsp{n}.jsonl")
        for lv in E.LEVELS:
            insts = E.load_level(n, lv)
            toks = [e["prompt_tokens"] for e in rec if e["level"] == lv and e["arm"] == "q14" and e.get("prompt_tokens")]
            lines.append(f"| {n} | {lv} | {len(insts)} | {sum(len(i['label']) for i in insts)} | "
                         f"{np.mean([len(E.label_prims(i)) for i in insts]):.1f} | {np.mean(toks) if toks else 0:.0f} |")
    return "\n".join(lines)


def main():
    notes = open(f"{OUT}/notes.md").read()
    S = score_all()
    audit_txt, excluded = audit()
    out = [f"# Phase 5, step 3 — open language with a larger local model", "",
           f"Generated {datetime.date.today().isoformat()} by scripts/phase5_open_language_report.py. Data in "
           "results/phase5/open_language/ (outside git); prompts in src/tsp_gnn/open_dsl.py and src/tsp_gnn/open_gen.py and "
           "with every L2 record.", "", "## Conclusion", "", section(notes, "Conclusion"), "",
           "## Goal", "", section(notes, "Goal"), "", "## Design", "", section(notes, "Design"), "",
           "## Step 0: models and speed", "", table_step0(), "", section(notes, "Step 0"), "",
           "**The step-0 times above were measured with no LKH-3 of mine running, on a GPU already throttled by temperature.** "
           "An earlier run with LKH-3 in parallel, on a cold GPU, is kept in step0_under_load.json and is not used as a latency. "
           "The latencies measured on a cooled GPU are in the next table.", "", table_latency(), "", thermal(), "",
           "GPU guard: " + table_pauses(), "",
           "## Step 1: the data set", "", dataset(), "", "### Audit of the L2 labels", "", audit_txt, "",
           "## Step 2B: who formalises best", "", "All texts (no exclusion):", "", table_b(S), ""]
    if excluded:
        out += ["Without the L2 texts the audit found wrong:", "", table_b({k: v for k, v in S.items() if k[1] == "L2"}, excluded), ""]
    out += ["McNemar, q14 against p1 (rule level, paired, all texts):", "", mcnemar(S, "q14", "p1"), "",
            "McNemar, q14 against q7:", "", mcnemar(S, "q14", "q7"), "", "McNemar, q14 with thinking against p1 and against q14 without "
            "(same instances):", "", mcnemar(S, "q14t", "p1"), "", mcnemar(S, "q14t", "q14"), "", "Error types:", "", table_errors(S), "",
            table_shapes(S), "", "## Step 2C: what it costs in the solver", "",
            "A rule set equal to the label reuses the reference route, so its LKH-3 time counts as 0 here: the oracle and the "
            "exact answers show no solver time, while a real run would pay about 31 s at n = 200 and 161 s at n = 1000. The "
            "time column therefore compares the extra cost of formalising, not a full run. LLM time: the latency measured "
            "alone; for q14t, its time in the run.", "", table_c(), "", "Valid routes, q14 against p1, paired:", "",
            mcnemar_routes("q14", "p1"), ""]
    if excluded:
        out += ["Without the L2 texts the audit found wrong (L2 rows only are affected):", "", table_c(excluded), ""]
    out += ["Rules rejected by the code check, by reason (all levels, both n):", "", table_rejects(), "",
            "## Step 3: the larger model on the older tasks", "",
            "The 7B column is computed on the same cases from the runs of steps 0b and 2.", "", step3(), "",
            section(notes, "Step 3"), "", "## Caveats", "", section(notes, "Caveats"), ""]
    with open("docs/phase5_open_language_report.md", "w") as f:
        f.write("\n".join(out))
    print("saved to docs/phase5_open_language_report.md")


if __name__ == "__main__":
    main()
