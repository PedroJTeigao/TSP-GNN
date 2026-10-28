"""Phase 5, step 1 report: the text-rule TSP data set (scripts/make_text_rules.py).

Reads data/phase5/text_rules_tsp{200,1000}.jsonl and writes results/phase5/text_rules_data.md: rule types and how
LKH-3 solves each one, text examples, LKH-3 time per type and how much each rule lengthens the route against the
pure TSP solved with the same pipeline. The opening notes are read from results/phase5/text_rules_data_notes.md.

Example:
    python scripts/phase5_text_rules_report.py
"""

import argparse
import datetime
import json
import os

import numpy as np

from tsp_gnn import rules as R

SIZES = (200, 1000)
LKH_MODEL = {
    "precedence": ("`{\"before\": a, \"after\": b}`: position of a < position of b on the route from the depot",
                   "TYPE = SOP on n + 1 nodes (depot first, a copy of it last); matrix entry (b, a) = −1"),
    "forbidden_edge": ("`{\"i\", \"j\"}`: i and j not adjacent on the route",
                       "cost 10⁶ (≫ any route) on (i, j) and (j, i) of the explicit matrix"),
    "mandatory_edge": ("`{\"i\", \"j\"}`: i and j adjacent on the route",
                       "FIXED_EDGES_SECTION; in SOP / TSPTW the fixed edge is directed, so both directions of each "
                       "mandatory edge are solved and the best feasible route is kept"),
    "time_window": ("`{\"customer\": k, \"t1\", \"t2\"}`: service at k starts in [t1, t2] (speed 1, waiting allowed)",
                    "TYPE = TSPTW, TIME_WINDOW_SECTION, DEPOT_SECTION = 1; travel times rounded up, so LKH-feasible "
                    "routes are feasible with the true distances"),
    "soft_zone": ("`{\"zone\", \"rect\", \"weight\": 1}`: minimise length + weight × length driven inside the rectangle",
                  "each matrix entry + weight × (length of the segment inside the zone); never infeasible"),
}


def load(data):
    out = {}
    for n in SIZES:
        path = os.path.join(data, f"text_rules_tsp{n}.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                out[n] = [json.loads(line) for line in f]
    return out


def pct(rec):
    return 100 * (rec["length"] / rec["pure_length"] - 1)


def stats(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return "—"
    return f"{v.mean():.2f} / {np.median(v):.2f} / {v.max():.2f}"


def table_types():
    lines = ["| Type | Formal translation (checked in code) | How LKH-3 solves it |", "|---|---|---|"]
    for t in R.RULE_TYPES:
        lines.append(f"| {t} | {LKH_MODEL[t][0]} | {LKH_MODEL[t][1]} |")
    return "\n".join(lines)


def table_texts(D):
    lines = []
    example = {}
    for recs in D.values():
        for rec in recs:
            for r in rec["rules"]:
                example.setdefault(r["type"], r)
    for t in R.RULE_TYPES:
        if t not in example:
            continue
        formal = example[t]["formal"]
        lines.append(f"**{t}** (formal: `{json.dumps(formal)}`)")
        lines.append("")
        for v in range(len(R.TEMPLATES[t])):
            lines.append(f"{v + 1}. {R.make_rule(t, v, **formal)['text']}")
        lines.append("")
    return "\n".join(lines)


def table_overview(D):
    lines = ["| n | Instances | Rules per instance (1 / 2 / 3) | LKH kind (TSP / SOP / TSPTW) | Feasible | "
             "Re-drawn rules | Rules the pure route breaks | LKH time, pure TSP (s, mean) | Total labelling time |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, recs in D.items():
        k = [sum(r["num_rules"] == j for r in recs) for j in (1, 2, 3)]
        kinds = [sum(r["kind"] == j for r in recs) for j in ("TSP", "SOP", "TSPTW")]
        breaks = [b for r in recs for b in r["pure_breaks"]]
        lines.append(f"| {n} | {len(recs)} | {' / '.join(map(str, k))} | {' / '.join(map(str, kinds))} | "
                     f"{sum(r['feasible'] for r in recs)} | {sum(r['attempts'] > 1 for r in recs)} | "
                     f"{100 * np.mean(breaks):.0f} % | {np.mean([r['pure_seconds'] for r in recs]):.2f} | "
                     f"{recs[0]['meta']['total_seconds'] / 60:.0f} min |")
    lines += ["", "\"Rules the pure route breaks\": the share of rules that the pure-TSP route (either direction for "
              "precedences and windows, as listed for the others) does not meet; soft zones are never broken."]
    return "\n".join(lines)


def table_by_type(D):
    """Single-rule instances: LKH time and route-length increase per type; all instances: per number of rules."""
    lines = ["| n | Rule type | Single-rule instances | LKH time (s): mean / median / max | LKH runs (mean) | "
             "Length increase vs. pure TSP (%): mean / median / max | Instances with increase > 0.01 % |",
             "|---|---|---|---|---|---|---|"]
    for n, recs in D.items():
        for t in R.RULE_TYPES:
            rs = [r for r in recs if r["num_rules"] == 1 and r["rules"][0]["type"] == t and r["feasible"]]
            if not rs:
                lines.append(f"| {n} | {t} | 0 | — | — | — | — |")
                continue
            inc = [pct(r) for r in rs]
            lines.append(f"| {n} | {t} | {len(rs)} | {stats([r['seconds'] for r in rs])} | "
                         f"{np.mean([r['lkh_runs'] for r in rs]):.1f} | {stats(inc)} | "
                         f"{sum(x > 0.01 for x in inc)} / {len(rs)} |")
    lines += ["", "| n | Rules per instance | Instances | LKH time (s): mean / median / max | "
              "Length increase (%): mean / median / max |", "|---|---|---|---|---|"]
    for n, recs in D.items():
        for k in (1, 2, 3):
            rs = [r for r in recs if r["num_rules"] == k and r["feasible"]]
            lines.append(f"| {n} | {k} | {len(rs)} | {stats([r['seconds'] for r in rs])} | "
                         f"{stats([pct(r) for r in rs])} |")
    lines += ["", "| n | Instances containing ... | Instances | LKH time (s): mean / median / max | "
              "Length increase (%): mean / median / max |", "|---|---|---|---|---|"]
    for n, recs in D.items():
        for t in R.RULE_TYPES:
            rs = [r for r in recs if any(x["type"] == t for x in r["rules"]) and r["feasible"]]
            lines.append(f"| {n} | {t} | {len(rs)} | {stats([r['seconds'] for r in rs])} | "
                         f"{stats([pct(r) for r in rs])} |")
    lines += ["", "Length increase = 100 × (reference length / pure-TSP length − 1), both from LKH-3 with the same "
              "pipeline and parameters, true (float) lengths. Small negative values are LKH noise. LKH time: all runs "
              "of the instance (both directions of each mandatory edge in SOP / TSPTW), files included."]
    return "\n".join(lines)


def table_soft(D):
    lines = ["| n | Instances with a soft zone | Length driven inside the zone: pure → reference (mean) | "
             "Length increase (%) | Objective (length + penalty) decrease (%) |", "|---|---|---|---|---|"]
    for n, recs in D.items():
        rs = [r for r in recs if r["zone_length"] and r["feasible"]]
        if not rs:
            continue
        zp = np.mean([sum(r["pure_zone_length"]) for r in rs])
        zr = np.mean([sum(r["zone_length"]) for r in rs])
        dec = [100 * (1 - r["objective"] / r["pure_objective"]) for r in rs]
        lines.append(f"| {n} | {len(rs)} | {zp:.3f} → {zr:.3f} | {stats([pct(r) for r in rs])} | {stats(dec)} |")
    return "\n".join(lines)


def build_report(data, results):
    D = load(data)
    notes = os.path.join(results, "phase5", "text_rules_data_notes.md")
    notes = open(notes).read().strip() if os.path.exists(notes) else "(notes missing: phase5/text_rules_data_notes.md)"
    ex = []
    for n, recs in D.items():
        r = max(recs[:20], key=lambda x: (len({y["type"] for y in x["rules"]}), x["idx"] == 0))
        ex += [f"Instance {r['idx']} of n = {n} ({r['kind']}; length +{pct(r):.2f} % over the pure TSP):", "",
               "```", r["text"], "```", "", f"Formal: `{json.dumps([x['formal'] for x in r['rules']])}`", ""]
    meta = next(iter(D.values()))[0]["meta"] if D else {}
    parts = [
        "# Phase 5, step 1: TSP instances with text rules (data set)", "",
        f"Generated by `scripts/phase5_text_rules_report.py` on {datetime.date.today().isoformat()}.", "", notes, "",
        "## Rule types", "", table_types(), "",
        "Conventions shared by every instance (they open every instance text):", "", f"> {R.PREAMBLE}", "",
        "A time window and a precedence rule are never drawn for the same instance (LKH-3.0.6 has no TSPTW with "
        "precedences). Rules are made to bind against the pure-TSP route where possible: precedences are pairs that "
        "the pure route, in a random direction, breaks (a single precedence still costs nothing, because the reversed "
        "route meets it); forbidden edges are edges of the pure route; mandatory edges join a near neighbour (5 "
        "nearest) that the pure route does not join; time windows (width 10 % of the pure length) close before the "
        "pure route reaches the customer in either direction (a later window only costs free waiting).", "",
        "## Text formulations (4 per type)", "", table_texts(D),
        "## Example instances", "", *ex,
        "## Data set", "", table_overview(D), "",
        "## LKH-3 time and cost of the rules", "", table_by_type(D), "",
        "### Soft zones", "", table_soft(D), "",
        "## Setup", "",
        f"LKH-3.0.6, RUNS = 1, MAX_TRIALS = {meta.get('max_trials')}, TIME_LIMIT = {meta.get('time_limit')} s per "
        f"run, SEED = 1; distances × {R.LKH_SCALE} rounded up; 8 instances in parallel on the shared 8-core Xeon "
        "(nice 10). Seeds [20261027, n, i] (new). Files: data/phase5/text_rules_tsp{n}.jsonl (rules, text, labels) "
        "and .npz (coords, pure and reference tours); scripts/make_text_rules.py, src/tsp_gnn/rules.py, "
        "tests/test_rules.py.",
    ]
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Phase 5 step 1 report (text-rule data set).")
    parser.add_argument("--data", default="data/phase5")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    out = os.path.join(args.results, "phase5", "text_rules_data.md")
    with open(out, "w") as f:
        f.write(build_report(args.data, args.results))
    print(f"saved to: {out}")


if __name__ == "__main__":
    main()
