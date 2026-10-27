"""Phase 2 ablation report: test graphs with k = 0.2 n (Joshi et al. 2021) vs. our k = 0.5 n and k = 10.

Reads results/phase2_scale/<run>/khalf100/<set>.npz (existing: k = min(0.5 n, 100)) and
results/phase2_k02n/<run>/{k02n,k10,khalf}/<set>.npz (scripts/scale_eval.py with --two-opt-k-mode khalf100, so
the 2-opt neighbour lists are the same in every column and only the network's graph changes).
Writes results/phase2_k02n_report.md. Missing results are reported as "missing".

Example:
    python scripts/phase2_k02n_report.py
"""

import argparse
import datetime
import importlib.util
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarize = _load("summarize")
ALPHA = summarize.ALPHA

MODELS = {"tagcn_k3": "TAGCN K=3", "gcn": "GCN"}
SEEDS = (0, 1, 2)
SIZES = (50, 100, 200, 500, 1000)
METHODS = ("greedy", "greedy+2opt", "beam10", "beam10+2opt")
# column: (label, folder root, k-mode)
MODES = {"k10": ("k = 10", "phase2_k02n", "k10"), "k02n": ("k = 0.2 n (Joshi)", "phase2_k02n", "k02n"),
         "khalf100": ("k = min(0.5 n, 100) (ours)", "phase2_scale", "khalf100"),
         "khalf": ("k = 0.5 n, uncapped", "phase2_k02n", "khalf")}


def k_of(mode, n):
    k = {"k10": 10, "k02n": round(0.2 * n), "khalf100": min(round(0.5 * n), 100), "khalf": round(0.5 * n)}[mode]
    return min(k, n - 1)


def load_all(results):
    R = {}
    for m in MODELS:
        for s in SEEDS:
            for mode, (_, root, folder) in MODES.items():
                for n in SIZES:
                    path = os.path.join(results, root, f"{m}_seed{s}", folder, f"tsp{n}_test.npz")
                    if os.path.exists(path):
                        with np.load(path) as f:
                            R[(m, s, mode, n)] = {k: f[k] for k in f.files}
    return R


def runs(R, m, mode, n):
    out = [R.get((m, s, mode, n)) for s in SEEDS]
    return None if any(r is None for r in out) else out


def cell(R, m, mode, n, method):
    rs = runs(R, m, mode, n)
    if rs is None:
        return "—" if mode == "khalf" and n < 500 else "missing"
    v = [float(np.nanmean(r[method])) for r in rs]
    return f"{np.mean(v):.2f} ± {np.std(v):.2f}"


def paired(a_list, ref_list, threshold):
    """a − ref per seed; verdict only if every seed's p < threshold and the medians agree in sign
    (as phase2_beam_report.paired: a seed with a zero median is neutral)."""
    res = [summarize.paired_compare(a, r) for a, r in zip(a_list, ref_list)]
    mean_diff = float(np.mean([x["mean_diff"] for x in res]))
    p_max = max(x["p"] for x in res)
    signs = {np.sign(x["median_diff"]) for x in res if abs(x["median_diff"]) > summarize.TIE_TOL}
    equal = float(np.mean([x["equal"] for x in res]))
    if len(signs) > 1:
        verdict = "n.s."
    else:
        direction = signs.pop() if signs else np.sign(mean_diff)
        verdict = ("better" if direction < 0 else "worse") if p_max < threshold and direction != 0 else "n.s."
    if equal == 100:
        verdict = "identical"
    return {"mean_diff": mean_diff, "p_max": p_max, "verdict": verdict, "equal": equal}


def run_tests(R, ref_mode="khalf100"):
    """{(n, model, method): test of k = 0.5 n (ref_mode) − k = 0.2 n}; Bonferroni within each size."""
    tests = {}
    for n in SIZES:
        family = [(m, meth) for m in MODELS for meth in METHODS]
        for m, meth in family:
            a, b = runs(R, m, ref_mode, n), runs(R, m, "k02n", n)
            if a is None or b is None:
                continue
            ok = lambda r: np.nan_to_num(r[meth], nan=np.inf)       # a 2-opt timeout counts as worst
            tests[(n, m, meth)] = paired([ok(r) for r in a], [ok(r) for r in b], ALPHA / len(family))
    return tests


def table_gaps(R, methods):
    modes = list(MODES)
    lines = ["| Model | n | Method | " + " | ".join(f"{MODES[k][0]}" for k in modes) + " |",
             "|---|---|---|" + "---|" * len(modes)]
    for m, label in MODELS.items():
        for n in SIZES:
            for meth in methods:
                ks = ", ".join(f"{k_of(k, n)}" for k in modes)
                lines.append(f"| {label} | {n} | {meth} | " + " | ".join(cell(R, m, k, n, meth) for k in modes) + " |")
    lines.append("")
    lines.append("Mean gap (%) ± std over the 3 seeds (best.pt, trained on TSP-20 with k = 10). k per size: " +
                 "; ".join(f"n = {n}: " + ", ".join(f"{k_of(k, n)}" for k in modes) for n in SIZES) +
                 " (columns in order). 2-opt: neighbour lists of k = min(0.5 n, 100) in every column.")
    return "\n".join(lines)


def table_tests(tests, title_ref="k = min(0.5 n, 100)"):
    if not tests:
        return "missing"
    lines = [f"| n | Model | Method | {title_ref} − 0.2 n (pp) | p (max over seeds) | Verdict for 0.5 n |",
             "|---|---|---|---|---|---|"]
    for (n, m, meth), t in sorted(tests.items(), key=lambda kv: (kv[0][0], kv[0][1], METHODS.index(kv[0][2]))):
        v = f"**{t['verdict']}**" if t["verdict"] in ("better", "worse") else t["verdict"]
        lines.append(f"| {n} | {MODELS[m]} | {meth} | {t['mean_diff']:+.3f} | {t['p_max']:.1e} | {v} |")
    lines.append("")
    lines.append("Paired two-sided Wilcoxon per seed on the per-instance gaps; verdict only if all 3 seeds have "
                 f"p < 0.05/8 (Bonferroni over the 8 tests of a size) and agree in direction. \"identical\": every "
                 "instance has the same gap (at n = 500, 0.2 n = 100 = min(0.5 n, 100): the same graph).")
    return "\n".join(lines)


def headline(tests):
    """One line: is 0.5 n better, equal or worse than 0.2 n, judged on the decoders with 2-opt."""
    by_n = {}
    for (n, m, meth), t in tests.items():
        if meth.endswith("+2opt"):
            by_n.setdefault(n, []).append(t["verdict"])
    if not by_n:
        return "missing"
    parts = []
    for n in sorted(by_n):
        v = by_n[n]
        word = ("same graph" if all(x == "identical" for x in v) else
                "better" if all(x == "better" for x in v) else "worse" if all(x == "worse" for x in v) else
                "equal" if all(x in ("n.s.", "identical") for x in v) else
                f"mixed ({sum(x == 'better' for x in v)} better, {sum(x == 'worse' for x in v)} worse, "
                f"{sum(x in ('n.s.', 'identical') for x in v)} n.s. of {len(v)})")
        parts.append(f"n = {n}: {word}")
    return "; ".join(parts)


def build_report(results):
    R = load_all(results)
    notes = os.path.join(results, "phase2_k02n_notes.md")
    notes = open(notes).read().strip() if os.path.exists(notes) else "(notes missing: results/phase2_k02n_notes.md)"
    tests = run_tests(R)
    tests_full = {k: v for k, v in run_tests(R, "khalf").items() if k[0] >= 500}
    parts = [
        "# Phase 2 ablation: test graph k = 0.2 n (Joshi et al. 2021) vs. k = 0.5 n and k = 10",
        "",
        f"Generated by `scripts/phase2_k02n_report.py` on {datetime.date.today().isoformat()}.",
        "",
        notes,
        "",
        f"**Automatic verdict (with 2-opt, k = min(0.5 n, 100) vs. 0.2 n, per size):** {headline(tests)}.",
        "",
        "Nothing was retrained: the TAGCN K=3 and GCN models of phases 1–2 (TSP-20, k = 10, best.pt, seeds 0–2) are "
        "evaluated with `scripts/scale_eval.py` on TSP-50 to TSP-1000. Only the network's test graph changes between "
        "columns; the 2-opt always uses the neighbour lists of min(0.5 n, 100), so the k = min(0.5 n, 100) column is "
        "the existing phase 2 scale result (reproduced exactly on tsp200 before launching). Our k is capped at 100, "
        "so at n = 500 it equals 0.2 n, and at n = 1000 it is below 0.2 n; the uncapped 0.5 n was run at 500 and 1000.",
        "",
        "## Gaps with 2-opt",
        "",
        table_gaps(R, ("greedy+2opt", "beam10+2opt")),
        "",
        "## Gaps without 2-opt",
        "",
        table_gaps(R, ("greedy", "beam10")),
        "",
        "## Paired tests: k = min(0.5 n, 100) vs. k = 0.2 n",
        "",
        table_tests(tests),
        "",
        "## Paired tests: uncapped k = 0.5 n vs. k = 0.2 n (n = 500, 1000)",
        "",
        f"**With 2-opt:** {headline(tests_full)}.",
        "",
        table_tests(tests_full, "k = 0.5 n"),
        "",
        "## Notes",
        "",
        "- The k = 10 column is re-run with the same pipeline (sparse decoders, 2-opt lists of min(0.5 n, 100)); the "
        "phase 1 k = 10 numbers (dense decoders, full 2-opt) are not comparable for the 2-opt columns.",
        "- At n = 500 both columns use the same graph (k = 100); the tiny TAGCN beam differences there come from "
        "non-deterministic GPU sums in the network, not from the graph.",
        "- Instances: TSP-50 and TSP-100 500 each, TSP-200 100, TSP-500 50, TSP-1000 20.",
    ]
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Phase 2 k = 0.2 n ablation report.")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    out = os.path.join(args.results, "phase2_k02n_report.md")
    with open(out, "w") as f:
        f.write(build_report(args.results))
    print(f"saved to: {out}")


if __name__ == "__main__":
    main()
