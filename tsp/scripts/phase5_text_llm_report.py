"""Phase 5, step 2 report: the LLM where the rule is not enough, judged by the code checker.

Reads data/phase5/text_rules_v2b_tsp<n>.jsonl (part A), formalize_tsp<n>.jsonl (B), pipeline_tsp<n>.jsonl (C),
llm_verify_tsp200.jsonl (D), the manual check (text_v2b_manual_check.json, text_v2b_label_errors.json) and the notes
results/phase5/text_rules_llm_notes.md (opening summary, if present). Writes results/phase5/text_rules_llm.md and a
copy in docs/phase5_text_rules_llm_report.md.

Evaluation set: the instances whose text_v2b is complete and has no label error (manual check); the same set is used
for text_v1 so the two wordings are compared on the same instances.

Example:
    python scripts/phase5_text_llm_report.py
"""

import datetime
import json
import os
import shutil
from collections import Counter, defaultdict

import numpy as np

from tsp_gnn import formalize as F
from tsp_gnn import rules as R

SIZES = (200, 1000)
DATA = "data/phase5"
OUT = "results/phase5/text_rules_llm.md"
DOCS = "docs/phase5_text_rules_llm_report.md"
ARMS = ("b0", "b0p", "b1", "b2")
ARM_NAME = {"b0": "b0 regex", "b0p": "b0+ keywords", "b1": "b1 7B zero-shot", "b2": "b2 7B few-shot", "oracle": "oracle (label)"}
TEXT_NAME = {"v1": "text_v1", "v2": "text_v2b"}
TYPES = R.RULE_TYPES
HARD = ("precedence", "forbidden_edge", "mandatory_edge", "time_window")


def jl(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def load():
    errors = json.load(open(f"{DATA}/text_v2b_label_errors.json"))
    rows, keep = {}, {}
    for n in SIZES:
        rows[n] = {r["idx"]: r for r in jl(f"{DATA}/text_rules_v2b_tsp{n}.jsonl")}
        keep[n] = {i for i, r in rows[n].items() if r["v2_complete"] and i not in set(errors.get(str(n), []))}
    return rows, keep, errors


def label(row):
    return [{"type": r["type"], "formal": r["formal"]} for r in row["rules"]]


def pc(x):
    return f"{100 * x:.1f}" if x == x else "—"


# ---------------------------------------------------------------------------------------------------- part A

def part_a(rows, keep, errors):
    check = json.load(open(f"{DATA}/text_v2b_manual_check.json"))
    out = ["## A. The linguistic test set (text_v2b)", "",
           "| | n = 200 | n = 1000 |", "|---|---|---|"]
    stat = {}
    for n in SIZES:
        rs = list(rows[n].values())
        rules = [r for x in rs for r in x["rules"]]
        src = Counter(r["v2_source"] for r in rules)
        stat[n] = {"instances": len(rs), "rules": len(rules), "paraphrase": src["paraphrase"],
                   "merged": src["merged"], "missing": src["missing"],
                   "named": sum(r["named"] for r in rules),
                   "extra": sum(x["extra_window"] for x in rs),
                   "merged_inst": sum(bool(x["merged_text"]) for x in rs),
                   "distract": sum(len(x["distractors"]) for x in rs),
                   "names": int(np.mean([len(x["names"]) for x in rs])),
                   "errors": len(errors.get(str(n), [])), "incomplete": sum(not x["v2_complete"] for x in rs),
                   "kept": len(keep[n])}
    for key, name in [("instances", "instances"), ("rules", "rules (label)"),
                      ("paraphrase", "rules written as their own paraphrase"),
                      ("merged", "rules inside a merged two-rule sentence"),
                      ("missing", "rules with no accepted sentence (instance left out)"),
                      ("named", "rules whose customers are named in the dictionary (by design)"),
                      ("names", "names in the dictionary per instance"),
                      ("extra", "instances with an extra non-binding window"),
                      ("distract", "irrelevant sentences (total)"),
                      ("errors", "instances with a label error (manual check, left out)"),
                      ("incomplete", "instances with a missing rule (left out)"),
                      ("kept", "**evaluation set**")]:
        out.append(f"| {name} | {stat[200][key]} | {stat[1000][key]} |")
    s, m, p = check["random_sample"], check["all_merged"], check["all_precedences"]
    out += ["", "Manual check of text_v2b (the paraphrases are written by the same 7B that translates them in b1 / b2):",
            f"- random sample of {s['size']} sentences ({s['per_n']} per n): {s['meaning_changed']} with the meaning "
            f"changed ({pc(s['meaning_changed'] / s['size'])} %); single-rule sentences "
            f"{s['single_sentences']['meaning_changed']}/{s['single_sentences']['checked']}, merged sentences "
            f"{s['merged_sentences']['meaning_changed']}/{s['merged_sentences']['checked']};",
            f"- all {m['checked']} merged sentences: {m['meaning_changed']} wrong "
            f"({pc(m['meaning_changed'] / m['checked'])} %): "
            + "; ".join(f"n = {n} #{i}: {why}" for n, d in m["cases"].items() for i, why in d.items()) + ";",
            f"- all precedences ({p['single_sentence']['200'] + p['single_sentence']['1000']} single-rule, "
            f"{p['merged']['200'] + p['merged']['1000']} merged): {p['single_sentence']['inverted'] + p['merged']['inverted']} "
            f"inverted, {p['merged']['other_errors']} contradictory ({p['merged']['note']}).",
            "- These are label errors (the sentence does not say what the label says), not model errors: the "
            "instances are left out of B (both texts), C and D. Single-rule sentences passed the keyword meaning "
            "filters of text_v2b; merged sentences carry most of the remaining error.", ""]
    return out


# ---------------------------------------------------------------------------------------------------- part B

def part_b(rows, keep):
    out = ["## B. Formalisation: text → formal rules (JSON), exact match against the label", ""]
    preds = {n: jl(f"{DATA}/formalize_tsp{n}.jsonl") + jl(f"{DATA}/formalize_b0p_tsp{n}.jsonl") for n in SIZES}
    res = {}
    out += ["| n | text | arm | rules exact (%) | instances exact (%) | answers with schema errors (%) | "
            "invalid JSON (%) | mean latency (s) |", "|---|---|---|---|---|---|---|---|"]
    conf = defaultdict(Counter)
    for n in SIZES:
        for text in ("v1", "v2"):
            for arm in ARMS:
                ps = [p for p in preds[n] if p["text"] == text and p["arm"] == arm and p["idx"] in keep[n]]
                if not ps:
                    continue
                rc = rt = ic = 0
                bytype = defaultdict(lambda: [0, 0])
                for p in ps:
                    s = F.score(label(rows[n][p["idx"]]), p["pred"])
                    rc += sum(s["correct"])
                    rt += len(s["correct"])
                    ic += s["exact"]
                    for (t, _), ok in zip(s["pairs"], s["correct"]):
                        bytype[t][0] += ok
                        bytype[t][1] += 1
                    for t, q in s["pairs"]:
                        conf[(text, arm)][(t, q)] += 1
                    for q in s["spurious"]:
                        conf[(text, arm)][("spurious", q)] += 1
                res[(n, text, arm)] = {"rule": rc / rt, "inst": ic / len(ps), "bytype": dict(bytype),
                                       "lat": np.mean([p["seconds"] for p in ps])}
                err = np.mean([bool(p["errors"]) for p in ps])
                bad = np.mean([p["errors"] == ["invalid JSON"] for p in ps])
                out.append(f"| {n} | {TEXT_NAME[text]} | {ARM_NAME[arm]} | {pc(rc / rt)} | {pc(ic / len(ps))} | "
                           f"{pc(err)} | {pc(bad)} | {res[(n, text, arm)]['lat']:.2f} |")
    out += ["", "b3 (API model): not tested, no OPENAI_API_KEY / ANTHROPIC_API_KEY on the server.", "",
            "Rules exact per type (%), both n together:", "",
            "| text | arm | " + " | ".join(TYPES) + " |", "|---|---|" + "---|" * len(TYPES)]
    for text in ("v1", "v2"):
        for arm in ARMS:
            cells = []
            for t in TYPES:
                c = sum(res[(n, text, arm)]["bytype"].get(t, [0, 0])[0] for n in SIZES if (n, text, arm) in res)
                tot = sum(res[(n, text, arm)]["bytype"].get(t, [0, 0])[1] for n in SIZES if (n, text, arm) in res)
                cells.append(f"{pc(c / tot)} ({tot})" if tot else "—")
            out.append(f"| {TEXT_NAME[text]} | {ARM_NAME[arm]} | " + " | ".join(cells) + " |")
    out += ["", "Rules exact on text_v2b by how the rule is written (both n): its own sentence with customer numbers "
            "only, its own sentence with at least one customer named, or inside a merged two-rule sentence. Soft "
            "zones have no customer and are counted under 'numbers'.", "",
            "| arm | own sentence, numbers | own sentence, named | merged sentence |", "|---|---|---|---|"]
    for arm in ARMS:
        cnt = defaultdict(lambda: [0, 0])
        for n in SIZES:
            for p in preds[n]:
                if p["text"] != "v2" or p["arm"] != arm or p["idx"] not in keep[n]:
                    continue
                row = rows[n][p["idx"]]
                s = F.score(label(row), p["pred"])
                for r, ok in zip(row["rules"], s["correct"]):
                    named = any(str(c) in row["names"] for c in F._customers(r))
                    kind = "merged" if r["v2_source"] == "merged" else "named" if named else "numbers"
                    cnt[kind][0] += ok
                    cnt[kind][1] += 1
        if cnt:
            out.append(f"| {ARM_NAME[arm]} | " + " | ".join(
                f"{pc(cnt[k][0] / cnt[k][1])} ({cnt[k][1]})" if cnt[k][1] else "—"
                for k in ("numbers", "named", "merged")) + " |")
    out += ["", "Confusion of rule types on text_v2b, both n (rows: label type; columns: the type of the prediction "
            "matched to it — same key, else same customers / zone, else any shared customer; 'missing' = no "
            "prediction; last row: predictions matched to no label rule). Diagonal counts include wrong "
            "parameters.", ""]
    cols = list(TYPES) + ["missing"]
    for arm in ARMS:
        c = conf[("v2", arm)]
        if not c:
            continue
        out += [f"**{ARM_NAME[arm]}**", "", "| label \\ predicted | " + " | ".join(cols) + " |",
                "|---|" + "---|" * len(cols)]
        for t in list(TYPES) + ["spurious"]:
            out.append(f"| {t} | " + " | ".join(str(c[(t, q)]) if c[(t, q)] else "·" for q in cols) + " |")
        out.append("")
    return out, res


# ---------------------------------------------------------------------------------------------------- part C

def part_c(rows, keep):
    out = ["## C. Full pipeline: arm's translation → LKH-3 → code checker against the CORRECT label", "",
           "Routes judged against the label. *valid*: meets every hard rule of the label; *violated*: mean number of "
           "hard label rules broken per instance; *cost*: objective (length + soft-zone penalty, of the label) "
           "against the reference, mean over all instances. Same LKH-3 parameters as the reference (the oracle IS "
           "the reference).", "",
           "| n | text | arm | translation exact (%) | valid routes (%) | violated rules / instance | "
           "cost vs reference (%) | LKH found no route for the translation (%) | repaired translations (%) |",
           "|---|---|---|---|---|---|---|---|---|"]
    res, vtypes = {}, defaultdict(Counter)
    for n in SIZES:
        recs = [r for r in jl(f"{DATA}/pipeline_tsp{n}.jsonl") + jl(f"{DATA}/pipeline_b0p_tsp{n}.jsonl")
                if r["idx"] in keep[n]]
        for text in ("v1", "v2"):
            for arm in ARMS + ("oracle",):
                rs = [r for r in recs if r["text"] == text and r["arm"] == arm]
                if not rs:
                    continue
                for r in rs:
                    vtypes[(text, arm)].update(r["violated"])
                res[(n, text, arm)] = {"valid": np.mean([r["valid"] for r in rs]),
                                       "viol": np.mean([len(r["violated"]) for r in rs]),
                                       "cost": np.mean([r["cost_vs_reference"] for r in rs])}
                out.append(f"| {n} | {TEXT_NAME[text]} | {ARM_NAME[arm]} | "
                           f"{pc(np.mean([r['reused_reference'] for r in rs]))} | {pc(res[(n, text, arm)]['valid'])} | "
                           f"{res[(n, text, arm)]['viol']:.2f} | {100 * res[(n, text, arm)]['cost']:+.2f} | "
                           f"{pc(np.mean([not r['solver_feasible'] for r in rs]))} | "
                           f"{pc(np.mean([bool(r['repairs']) for r in rs]))} |")
    out += ["", "Violated label rules by type (both n, count over instances):", "",
            "| text | arm | " + " | ".join(HARD) + " |", "|---|---|" + "---|" * len(HARD)]
    for text in ("v1", "v2"):
        for arm in ARMS + ("oracle",):
            if vtypes[(text, arm)] or any((n, text, arm) in res for n in SIZES):
                out.append(f"| {TEXT_NAME[text]} | {ARM_NAME[arm]} | "
                           + " | ".join(str(vtypes[(text, arm)][t]) for t in HARD) + " |")
    out.append("")
    return out, res


# ---------------------------------------------------------------------------------------------------- part D

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return p, r


def part_d():
    recs = jl(f"{DATA}/llm_verify_tsp200.jsonl")
    out = ["## D. The 7B as route verifier (n = 200, text_v2b)", "",
           "n = 1000 left out: a route written as visits with times is ~10k tokens and does not fit the 8192-token "
           "context. Routes: the oracle route of each instance (no violation) and the same route with one injected "
           "error, verified in code to break exactly one rule. Unit: (route, hard rule of the label); truth: the "
           "code checker (100 % by construction). *plain*: text + route; *report*: also the code checker's report.",
           ""]
    if not recs:
        return out + ["(no results)", ""], {}
    res = {}
    out += ["| variant | routes | invalid JSON (%) | route verdict right (%) | false alarms on oracle routes (%) | "
            "injected error found (%) | false alarms on non-binding windows (%) | claims matching no rule | "
            "mean latency (s) |", "|---|---|---|---|---|---|---|---|---|"]
    for variant in ("plain", "report"):
        rs = [r for r in recs if r["variant"] == variant]
        ok = [r["claimed"] is not None for r in rs]
        right = [r["claimed"] is not None and r["claimed"] == r["truth"] for r in rs]
        orc = [r for r in rs if r["route"] == "oracle"]
        inj = [r for r in rs if r["route"] == "injected"]
        fa = np.mean([r["claimed"] is None or bool(r["claimed"]) for r in orc]) if orc else float("nan")
        found = np.mean([r["claimed"] is not None and r["truth"][0] in r["claimed"] for r in inj]) if inj else float("nan")
        nb_units = [(r, k) for r in rs for k, x in enumerate(r["rules"]) if not x["binding"]]
        nb = np.mean([r["claimed"] is not None and k in r["claimed"] for r, k in nb_units]) if nb_units else float("nan")
        res[variant] = {"right": np.mean(right), "fa": fa, "found": found}
        out.append(f"| {variant} | {len(rs)} | {pc(1 - np.mean(ok))} | {pc(np.mean(right))} | {pc(fa)} | "
                   f"{pc(found)} | {pc(nb)} ({len(nb_units)}) | {sum(r['unmatched_claims'] for r in rs)} | "
                   f"{np.mean([r['seconds'] for r in rs]):.1f} |")
    out += ["", "Precision / recall per rule type (unit: route × hard rule; a JSON failure counts as no claim):", "",
            "| type | violations (truth) | rule units | plain P | plain R | report P | report R | "
            "random at base rate P = R |", "|---|---|---|---|---|---|---|---|"]
    for t in HARD + ("all",):
        cells, base = [], None
        for variant in ("plain", "report"):
            tp = fp = fn = units = pos = 0
            for r in recs:
                if r["variant"] != variant:
                    continue
                claimed = set(r["claimed"] or [])
                for k, x in enumerate(r["rules"]):
                    if x["type"] == "soft_zone" or (t != "all" and x["type"] != t):
                        continue
                    units += 1
                    truth = k in r["truth"]
                    pos += truth
                    tp += truth and k in claimed
                    fp += (not truth) and k in claimed
                    fn += truth and k not in claimed
            p, rr = prf(tp, fp, fn)
            cells += [pc(p), pc(rr)]
            base = pos / units if units else float("nan")
            res.setdefault("prf", {})[(t, variant)] = (p, rr)
        out.append(f"| {t} | {pos} | {units} | " + " | ".join(cells) + f" | {pc(base)} |")
    out += ["", "Chance baselines: 'never report a violation' has recall 0 and is right on every oracle route "
            "(50 % route verdicts if half the routes are injected); a random checker that flags each rule with the "
            "base-rate probability has precision = recall = base rate.", ""]
    return out, res


# ---------------------------------------------------------------------------------------------------- main

def main():
    rows, keep, errors = load()
    a = part_a(rows, keep, errors)
    b, rb = part_b(rows, keep)
    c, rc = part_c(rows, keep)
    d, rd = part_d()
    notes = open("results/phase5/text_rules_llm_notes.md").read().strip() if \
        os.path.exists("results/phase5/text_rules_llm_notes.md") else ""
    commits = sorted({p.get("git_commit") for n in SIZES for p in jl(f"{DATA}/formalize_tsp{n}.jsonl")} - {None})
    head = ["# Phase 5, step 2 — the LLM where the rule is not enough, judged by the code checker", "",
            f"Generated {datetime.date.today().isoformat()} by scripts/phase5_text_llm_report.py "
            f"(runs at commit {', '.join(commits) or '—'}). Model: qwen2.5:7b (Q4_K_M) on the private Ollama, "
            "context 8192, temperature 0 (paraphrases: 0.8). Data: the step-1 instances, n = 200 and 1000, "
            "200 each.", ""]
    if notes:
        head += [notes, ""]
    text = "\n".join(head + a + b + c + d) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(text)
    os.makedirs(os.path.dirname(DOCS), exist_ok=True)
    shutil.copy(OUT, DOCS)
    print(f"wrote {OUT} and {DOCS}")
    for k, v in sorted(rb.items()):
        print("B", k, {x: round(y, 3) for x, y in v.items() if x in ("rule", "inst", "lat")})
    for k, v in sorted(rc.items()):
        print("C", k, {x: round(float(y), 4) for x, y in v.items()})
    for k, v in rd.items():
        print("D", k, v)


if __name__ == "__main__":
    main()
