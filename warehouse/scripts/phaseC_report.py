"""Report of phase C: docs/phaseC_language.md, from results/warehouse/phaseC/. The hand-written parts are read from docs/phaseC_notes.md (one
"## " section per part); every number and table is computed here.

    python scripts/phaseC_report.py
"""

import json
import math
import os
import re
import sys
from statistics import median

import numpy as np
from scipy.stats import binomtest
from scipy.stats import t as student

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phaseC_language as Q
from warehouse import dsl as L
from warehouse import world as Y

OUT = Q.OUT
ARMS = ("oracle", "p0", "p1", "h1", "q14", "q7")
ERRORS = ("wrong selector", "exception lost", "inverted", "wrong value", "missing")


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
    return f"{k}/{m} = {100 * k / m:.1f} % [{lo:.1f}, {hi:.1f}]" if m else "-"


def mean_ci(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return f"{x.mean():+.1f}" if len(x) else "-"
    return f"{x.mean():+.1f} ± {student.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)):.1f}"


def load():
    """rows: one per (request, level, arm), with its score and, once run, its episode. Parsing the worlds takes a while, so this is done once."""
    rows = []
    trans = {}
    for e in Q.jl(f"{OUT}/translate.jsonl"):
        trans.setdefault(e["idx"], []).append(e)
    para = Q.paragraphs()
    for r in Q.requests("test"):
        ctx = L.Context(Y.make(r["seed"]))
        label = Q.labels(r, ctx)
        truth = Q.accepted({"answer": {"rules": [x["dsl"] for x in r["rules"]]}}, ctx)
        ep_o = episode(r["idx"], truth[3])
        back = " ".join(r["background"]).lower()
        for e in trans.get(r["idx"], []):
            rules, errors = L.validate(e["answer"], ctx)
            prims, rejected, _, key = Q.accepted(e, ctx)
            s = L.score(label, [p for x in rules for p in x["prims"]])
            kinds = [L.classify_error(x, [p for y in rules for p in y["prims"]], ctx) for x, ok in zip(label, s["correct"]) if not ok]
            made = L.invented(rules, label)
            rows.append({"idx": r["idx"], "level": e["level"], "arm": e["arm"], "correct": s["correct"], "exact": s["exact"], "types": [x["dsl"]["type"] for x in r["rules"]],
                         "shapes": [x["shape"] for x in r["rules"]], "errors": kinds, "invented": len(made), "distractor": sum(any(w in back for w in names_in(m)) for m in made),
                         "dropped": len(errors), "rejected": len(rejected), "reasons": [x["reason"] for x in rejected], "seconds": e["seconds"], "attempts": e.get("attempts", 0),
                         "valid_json": e.get("valid_json", True), "sent": e.get("sent"), "ep": episode(r["idx"], key), "oracle": ep_o, "same_as_oracle": key == truth[3], "lkh_running": e.get("lkh_running", 0)})
    return rows


def names_in(rule):
    return [w.lower() for w in re.findall(r'"([^"]+)"', json.dumps(rule)) if w not in ("type", "first", "then", "a", "b", "who", "kind", "customer", "corridor", "station", "t", "t1", "t2", "corridors", "except", "robot", "order", "all", "any_of", "priority")]


def episode(idx, key):
    path = f"{OUT}/c/{idx}_{key}.json"
    return json.load(open(path)) if os.path.exists(path) else None


def unclean(rows):
    """Every request but the audited L2 texts found right: what is left is a small sample without label errors."""
    path = f"{OUT}/l2_audit.json"
    good = {e["idx"] for e in json.load(open(path))["items"] if e["ok"]}
    return {r["idx"] for r in rows} - good


def excluded():
    path = f"{OUT}/l2_audit.json"
    return {e["idx"] for e in json.load(open(path))["items"] if not e["ok"]} if os.path.exists(path) else set()


def pick(rows, level, arm, skip=frozenset()):
    return [r for r in rows if r["level"] == level and r["arm"] == arm and not (level == "L2" and r["idx"] in skip)]


def table_b(rows, skip=frozenset()):
    lines = ["| level | arm | rules exact [Wilson 95 %] | requests exact [Wilson 95 %] | valid JSON | retries | rules dropped by the schema |", "|---|---|---:|---:|---:|---:|---:|"]
    for lv in Q.LEVELS:
        for arm in ARMS[1:]:
            s = pick(rows, lv, arm, skip)
            if not s:
                continue
            k, m = sum(sum(r["correct"]) for r in s), sum(len(r["correct"]) for r in s)
            e = sum(r["exact"] for r in s)
            lines.append(f"| {lv} | {arm} | {pc(k, m)} | {pc(e, len(s))} | {100 * np.mean([r['valid_json'] for r in s]):.1f} % | {sum(r['attempts'] > 1 for r in s)} | {sum(r['dropped'] for r in s)} |")
        q7 = {r["idx"] for r in pick(rows, lv, "q7", skip)}
        s = [r for r in pick(rows, lv, "q14", skip) if r["idx"] in q7]
        if s:
            k, m = sum(sum(r["correct"]) for r in s), sum(len(r["correct"]) for r in s)
            lines.append(f"| {lv} | q14 on the requests of q7 | {pc(k, m)} | {pc(sum(r['exact'] for r in s), len(s))} | | | |")
    return "\n".join(lines)


def mcnemar(rows, a, b, skip=frozenset()):
    lines = [f"| level | unit | {a} right, {b} wrong | {a} wrong, {b} right | exact McNemar p |", "|---|---|---:|---:|---:|"]
    for lv in Q.LEVELS:
        ra, rb = {r["idx"]: r for r in pick(rows, lv, a, skip)}, {r["idx"]: r for r in pick(rows, lv, b, skip)}
        x = y = u = v = 0
        for i in ra.keys() & rb.keys():
            for p, q in zip(ra[i]["correct"], rb[i]["correct"]):
                x += p and not q
                y += q and not p
            u += ra[i]["exact"] and not rb[i]["exact"]
            v += rb[i]["exact"] and not ra[i]["exact"]
        lines.append(f"| {lv} | rule | {x} | {y} | {binomtest(x, x + y, 0.5).pvalue if x + y else 1.0:.3g} |")
        lines.append(f"| {lv} | request | {u} | {v} | {binomtest(u, u + v, 0.5).pvalue if u + v else 1.0:.3g} |")
    return "\n".join(lines)


def table_errors(rows):
    lines = ["| level | arm | rules missed | " + " | ".join(ERRORS) + " | rules invented | of them from a distractor |", "|---|---|---:|" + "---:|" * (len(ERRORS) + 2)]
    for lv in Q.LEVELS:
        for arm in ARMS[1:]:
            s = pick(rows, lv, arm)
            if s:
                all_e = [x for r in s for x in r["errors"]]
                lines.append(f"| {lv} | {arm} | {len(all_e)} | " + " | ".join(str(all_e.count(k)) for k in ERRORS) + f" | {sum(r['invented'] for r in s)} | {sum(r['distractor'] for r in s)} |")
    return "\n".join(lines)


def table_types(rows):
    kinds = ("pick_order", "close", "no_mix", "robot_only", "deadline", "station")
    lines = ["| level | arm | " + " | ".join(kinds) + " |", "|---|---|" + "---:|" * len(kinds)]
    for lv in Q.LEVELS:
        for arm in ARMS[1:]:
            s = pick(rows, lv, arm)
            if s:
                cells = []
                for k in kinds:
                    got = [ok for r in s for t, ok in zip(r["types"], r["correct"]) if t == k]
                    cells.append(f"{100 * np.mean(got):.0f} %" if got else "-")
                lines.append(f"| {lv} | {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(a, b):
    return 100 * (a - b) / b


def translate_alone(level, arm, s):
    """Seconds of translation for an arm, with the LLM latency measured alone: its median, once per request sent to the LLM."""
    data = Q.jl(f"{OUT}/latency_alone.jsonl")
    sec = lambda a: median(e["seconds"] for e in data if e["arm"] == a and e["level"] == level)
    if arm in Q.MODELS:
        return sec(arm)
    return np.mean([bool(r["sent"]) for r in s]) * sec("q14") if arm == "h1" else 0.0


def table_valid(rows, skip=frozenset()):
    """Paired test on the episodes: one arm valid (0 true rules broken) and the other not."""
    lines = ["| level | arms | first valid, second not | first not, second valid | exact McNemar p |", "|---|---|---:|---:|---:|"]
    for lv in Q.LEVELS:
        for a, b in (("q14", "p1"), ("h1", "q14"), ("h1", "p1")):
            ra, rb = {r["idx"]: r for r in pick(rows, lv, a, skip)}, {r["idx"]: r for r in pick(rows, lv, b, skip)}
            x = sum(ra[i]["ep"]["broken"] == 0 and rb[i]["ep"]["broken"] > 0 for i in ra)
            y = sum(ra[i]["ep"]["broken"] > 0 and rb[i]["ep"]["broken"] == 0 for i in ra)
            lines.append(f"| {lv} | {a} vs {b} | {x} | {y} | {binomtest(x, x + y, 0.5).pvalue if x + y else 1.0:.3g} |")
    return "\n".join(lines)


def table_outliers(rows):
    """The median request costs nothing; the mean is moved by a few large cases."""
    lines = ["| level | arm | makespan vs oracle %, median | requests above +20 % | requests below -5 % | largest % |", "|---|---|---:|---:|---:|---:|"]
    for lv in Q.LEVELS:
        for arm in ARMS[1:]:
            p = [pct(r["ep"]["makespan"], r["oracle"]["makespan"]) for r in pick(rows, lv, arm)]
            lines.append(f"| {lv} | {arm} | {median(p):+.1f} | {sum(x > 20 for x in p)} | {sum(x < -5 for x in p)} | {max(p):+.1f} |")
    return "\n".join(lines)


def table_own(rows):
    """Episodes that broke a rule of their own arm, rules the planner was given."""
    bad = [r for r in rows if not r["ep"]["ok"]]
    return f"{len(bad)} of {len(rows)} arm episodes broke a rule the arm itself had given the solvers; {sum(not r['oracle']['ok'] for r in rows if r['arm'] == 'p1' and r['level'] == 'L1')} of 200 oracle episodes did."


def table_c(rows, skip=frozenset()):
    lines = ["| level | arm | episodes | 0 violations of the true rules [Wilson 95 %] | requests with a rule rejected (rules) | makespan | vs oracle % | mean delay | vs oracle % | true rules broken per episode | LLM + parser s, main run | total s, LLM alone | ms/step planner |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lv in Q.LEVELS:
        base = [r for r in pick(rows, lv, "p1", skip) if r["oracle"]]
        if base:
            o = [r["oracle"] for r in base]
            ok = sum(x["broken"] == 0 for x in o)
            lines.append(f"| {lv} | oracle | {len(o)} | {pc(ok, len(o))} | 0 | {np.mean([x['makespan'] for x in o]):.0f} | - | {np.mean([x['mean_delay'] for x in o]):.0f} | - | {np.mean([x['broken'] for x in o]):.2f} | 0 | {np.mean([x['elapsed'] for x in o]):.1f} | {np.mean([x['step_ms'] for x in o]):.1f} |")
        for arm in ARMS[1:]:
            s = [r for r in pick(rows, lv, arm, skip) if r["ep"] and r["oracle"]]
            if s:
                ok = sum(r["ep"]["broken"] == 0 for r in s)
                rej = sum(r["rejected"] > 0 for r in s)
                lines.append(f"| {lv} | {arm} | {len(s)} | {pc(ok, len(s))} | {rej} ({sum(r['rejected'] for r in s)}) | {np.mean([r['ep']['makespan'] for r in s]):.0f} | {mean_ci([pct(r['ep']['makespan'], r['oracle']['makespan']) for r in s])} |"
                             f" {np.mean([r['ep']['mean_delay'] for r in s]):.0f} | {mean_ci([pct(r['ep']['mean_delay'], r['oracle']['mean_delay']) for r in s])} | {np.mean([r['ep']['broken'] for r in s]):.2f} | {np.mean([r['seconds'] for r in s]):.2f} | {translate_alone(lv, arm, s) + np.mean([r['ep']['elapsed'] for r in s]):.1f} | {np.mean([r['ep']['step_ms'] for r in s]):.1f} |")
    return "\n".join(lines)


def table_wrong(rows):
    """What a wrong translation costs: the episodes of requests not translated exactly, split by whether they broke a true rule."""
    lines = ["| level | arm | not exact | of them invalid | makespan vs oracle %, invalid | delay vs oracle %, invalid | makespan vs oracle %, valid | delay vs oracle %, valid | true rule violations per invalid episode | atomic checks broken per invalid episode |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lv in Q.LEVELS:
        for arm in ARMS[1:]:
            s = [r for r in pick(rows, lv, arm) if r["ep"] and r["oracle"] and not r["same_as_oracle"]]
            if not s:
                continue
            bad = [r for r in s if r["ep"]["broken"]]
            good = [r for r in s if not r["ep"]["broken"]]
            f = lambda g, k: mean_ci([pct(r["ep"][k], r["oracle"][k]) for r in g])
            lines.append(f"| {lv} | {arm} | {len(s)} | {len(bad)} | {f(bad, 'makespan')} | {f(bad, 'mean_delay')} | {f(good, 'makespan')} | {f(good, 'mean_delay')} | {np.mean([r['ep']['broken'] for r in bad]) if bad else 0:.1f} | {np.mean([len(r['ep']['by_check']) for r in bad]) if bad else 0:.1f} |")
    return "\n".join(lines)


def table_rejects(rows):
    count = {}
    for r in rows:
        for x in r["reasons"]:
            count[(r["arm"], x)] = count.get((r["arm"], x), 0) + 1
    lines = ["| arm | why the validator rejected | rules |", "|---|---|---:|"]
    for (arm, why), n in sorted(count.items(), key=lambda x: (ARMS.index(x[0][0]), -x[1])):
        lines.append(f"| {arm} | {why} | {n} |")
    return "\n".join(lines)


def table_hybrid(rows):
    lines = ["| level | sent to the LLM | of the texts p1 got wrong, sent | of the texts p1 got right, sent |", "|---|---:|---:|---:|"]
    for lv in Q.LEVELS:
        h, p = pick(rows, lv, "h1"), {r["idx"]: r for r in pick(rows, lv, "p1")}
        if h:
            bad = [r for r in h if not p[r["idx"]]["exact"]]
            good = [r for r in h if p[r["idx"]]["exact"]]
            lines.append(f"| {lv} | {sum(bool(r['sent']) for r in h)}/{len(h)} | {sum(bool(r['sent']) for r in bad)}/{len(bad)} | {sum(bool(r['sent']) for r in good)}/{len(good)} |")
    return "\n".join(lines)


def table_latency(rows):
    data = Q.jl(f"{OUT}/latency_alone.jsonl")
    plan = [r["oracle"] for r in pick(rows, "L1", "p1") if r["oracle"]]
    ms = np.mean([x["step_ms"] for x in plan]) if plan else float("nan")
    steps = np.mean([x["steps"] for x in plan]) if plan else float("nan")
    lines = ["| arm | level | calls | median s | p95 s | tokens/s | GPU C | MHz | calls with LKH of mine running | translation in planner steps |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in Q.MODELS:
        for lv in Q.LEVELS:
            s = [e for e in data if e["arm"] == arm and e["level"] == lv]
            if s:
                sec = [e["seconds"] for e in s]
                lines.append(f"| {arm} | {lv} | {len(s)} | {median(sec):.1f} | {np.percentile(sec, 95):.1f} | {median(e['tok_s'] for e in s):.0f} | {median(e['temp_c'] for e in s):.0f} | {median(e['clock_mhz'] for e in s):.0f} | {sum(e['lkh_running'] > 0 for e in s)} | {1e3 * median(sec) / ms:.0f} |")
    lines.append("")
    lines.append(f"The planner takes {ms:.1f} ms per step, or {ms * steps / 1e3:.1f} s over an episode of {steps:.0f} steps. The request is translated once, before the episode starts, so the LLM never enters the real-time loop.")
    return "\n".join(lines)


def table_run_latency(rows):
    lines = ["| level | arm | mean s in the main run | p95 | of those calls, with LKH of mine running |", "|---|---|---:|---:|---:|"]
    for lv in Q.LEVELS:
        for arm in ("p0", "p1", "h1", "q14", "q7"):
            s = pick(rows, lv, arm)
            if s:
                sec = [r["seconds"] for r in s]
                lines.append(f"| {lv} | {arm} | {np.mean(sec):.2f} | {np.percentile(sec, 95):.2f} | {sum(r['lkh_running'] > 0 for r in s)} |")
    return "\n".join(lines)


def table_pauses():
    rows = Q.jl(f"{OUT}/gpu_guard.jsonl")
    pauses = [e for e in rows if e.get("event") == "pause"]
    resumes = [e for e in rows if e.get("event") == "resume"]
    checks = [e for e in rows if e.get("event") == "check"]
    low = min((e["free"] for e in checks), default=float("nan"))
    return f"{len(checks)} checks, {len(pauses)} pauses, {sum(e['seconds'] for e in resumes) / 60:.1f} min paused in all; the least free memory seen was {low} MiB."


def audit():
    path = f"{OUT}/l2_audit.json"
    if not os.path.exists(path):
        return "The audit has not been done."
    items = json.load(open(path))["items"]
    bad = [e for e in items if not e["ok"]]
    lo, hi = wilson(len(bad), len(items))
    lines = [f"{len(bad)} of {len(items)} L2 texts carry a wrong label: {100 * len(bad) / len(items):.1f} % [{lo:.1f}, {hi:.1f}].", "", "| request | what is wrong |", "|---:|---|"]
    lines += [f"| {e['idx']} | {e['why']} |" for e in bad]
    return "\n".join(lines)


def budget():
    req = Q.jl(f"{OUT}/requests.jsonl")
    para = Q.jl(f"{OUT}/l2.jsonl")
    tr = Q.jl(f"{OUT}/translate.jsonl")
    lat = Q.jl(f"{OUT}/latency_alone.jsonl")
    eps = [json.load(open(f"{OUT}/c/{x}")) for x in os.listdir(f"{OUT}/c")] if os.path.isdir(f"{OUT}/c") else []
    gen = sum(r["elapsed"] for r in req) / 3600
    c = sum(e["elapsed"] for e in eps) / 3600
    g2 = sum(e["seconds"] for e in para) / 3600
    g3 = sum(e["seconds"] for e in tr if e["arm"] in Q.MODELS) / 3600
    g4 = sum(e["seconds"] for e in lat) / 3600
    return (f"| stage | hours |\n|---|---:|\n| CPU: L1 with its oracle episodes | {gen:.2f} |\n| CPU: {len(eps)} episodes | {c:.2f} |\n| GPU: L2 | {g2:.2f} |\n| GPU: translation | {g3:.2f} |\n"
            f"| GPU: latency alone | {g4:.2f} |\n| total | {gen + c + g2 + g3 + g4:.2f} |")


def dataset():
    req = Q.requests("test")
    n = [len(r["rules"]) for r in req]
    shapes = {}
    for r in req:
        for x in r["rules"]:
            shapes[x["shape"]] = shapes.get(x["shape"], 0) + 1
    para = Q.paragraphs()
    empty = sum(not para.get(r["idx"], {}).get("paragraph") for r in req)
    return (f"{len(req)} test requests and {len(Q.requests('dev'))} development requests, one world each; {sum(n)} rules, {min(n)}-{max(n)} per request. "
            f"Draws needed by the oracle check: {sum(r['attempts'] for r in req)} for {len(req)} requests. L2 texts missing: {empty}. Rule shapes: " + ", ".join(f"{k} {v}" for k, v in sorted(shapes.items())) + ".")


def section(notes, name):
    m = re.search(rf"^## {name}\n(.*?)(?=^## |\Z)", notes, re.S | re.M)
    return m.group(1).strip() if m else f"(missing: {name})"


def main():
    rows = load()
    skip = excluded()
    clean = unclean(rows)
    notes = open("docs/phaseC_notes.md").read()
    head = "# Warehouse, phase C: the LLM as a language interface\n"
    parts = [head, section(notes, "Conclusion"), "## Design\n", section(notes, "Design"), "## Data\n", dataset(),
             "## Table B: translation\n", table_b(rows), "\nWithout the L2 texts the audit found wrong:\n", table_b(rows, skip),
             "\nq14 against p1:\n", mcnemar(rows, "q14", "p1"), "\nh1 against q14:\n", mcnemar(rows, "h1", "q14"), "\nh1 against p1:\n", mcnemar(rows, "h1", "p1"),
             "\nThe same tests without the L2 texts the audit found wrong, q14 against p1:\n", mcnemar(rows, "q14", "p1", skip), "\nh1 against q14:\n", mcnemar(rows, "h1", "q14", skip),
             "\nOn the 39 audited L2 texts found right, the only L2 sample known to be free of label errors (the L1 rows are unchanged):\n", table_b(rows, clean),
             "\n### Types of error\n", table_errors(rows), "\n### Rules exact by type\n", table_types(rows), "\n### The trigger of the hybrid\n", table_hybrid(rows),
             "## Table C: episodes\n", section(notes, "Reading table C"), "\n", table_c(rows), "\nWithout the L2 texts the audit found wrong:\n", table_c(rows, skip), "\nOn the 39 audited L2 texts found right:\n", table_c(rows, clean),
             "\nPaired, valid episodes:\n", table_valid(rows), "\nThe same without the L2 texts the audit found wrong:\n", table_valid(rows, skip), "\n", table_own(rows),
             "\n### What a wrong translation costs\n", table_wrong(rows), "\n", table_outliers(rows), "\n### What the validator rejected\n", table_rejects(rows),
             "## Latency\n", section(notes, "Latency"), "\nMeasured alone, 30 calls per arm:\n", table_latency(rows), "\nIn the main run, for reference:\n", table_run_latency(rows),
             "## GPU\n", table_pauses(), "\n", section(notes, "GPU"), "## Audit of L2\n", section(notes, "Audit"), "\n", audit(), "## Budget\n", budget(), "\n", section(notes, "Budget"), "## Caveats\n", section(notes, "Caveats")]
    open("docs/phaseC_language.md", "w").write("\n".join(parts) + "\n")
    print("written", len(rows))


if __name__ == "__main__":
    main()
