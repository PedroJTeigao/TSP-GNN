"""Phase 2 beam search report: greedy vs. beam search decoding of already trained models.

Reads results/phase2_beam/<config>/<run>/ (written by scripts/beam_eval.py) and writes:
    1. mean gap (mean ± std over the 3 seeds) of greedy, beam 10 and beam 100, without and with 2-opt;
    2. paired Wilcoxon tests of beam 100 + 2-opt against the model's own greedy + 2-opt, classical
       greedy + 2-opt and Christofides + 2-opt (every configuration is tested: no selection on the test set);
    3. mean decoding time per instance for each method;
    4. how often beam search gives a longer tour than greedy (pure beam search does not forbid it);
    5. a consistency check: the greedy gaps must match the earlier evaluations of the same checkpoints.
Missing results are reported as "missing" instead of stopping the report.

Example:
    python scripts/phase2_beam_report.py
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
batch2 = _load("phase2_batch2_report")
checks = _load("phase2_checks_report")
MODELS, ALPHA, TIE_TOL = summarize.MODELS, summarize.ALPHA, summarize.TIE_TOL
DATASETS, SEEDS = batch2.DATASETS, batch2.SEEDS
BEAM = "phase2_beam"  # folder inside results/
WIDTHS = (10, 100)
DECODERS = {"greedy": "greedy", **{f"beam{b}": f"B={b}" for b in WIDTHS}}
BASE_REFS = {"greedy+2opt": "Classical greedy + 2-opt", "christofides+2opt": "Christofides + 2-opt"}

# name -> (description, models, run name of (model, seed), earlier evaluation folders of the same
# checkpoint with k = 0.5 n, for the consistency check; later folders win).
CONFIGS = {
    "phase1": ("Phase 1 (k = 10, BatchNorm, best val gap)", ("gcn", "gat", "gated"),
               lambda m, s: f"{m}_seed{s}",
               lambda m, s: [f"runs/{m}_seed{s}", f"phase2_batch2/kfrac05/{m}_seed{s}"]),
    "ln_last": ("LayerNorm, last epoch (last.pt)", ("gcn", "gated"),
                lambda m, s: f"{m}_layernorm_ckpt_seed{s}",
                lambda m, s: [f"phase2_checks/ln/last_kfrac05/{m}_layernorm_ckpt_seed{s}"]),
    "data100k": ("Step 5, 100k training instances (best val gap)", ("gated",),
                 lambda m, s: f"{m}_data100k_seed{s}",
                 lambda m, s: [f"phase2_datasize/kfrac05/{m}_data100k_seed{s}"]),
}


# ---------------------------------------------------------------- loading

def load_all(results):
    """{(config, model): [per seed: (eval.json, per-instance dict) or None]}."""
    return {(c, m): [batch2.load_folder(os.path.join(results, BEAM, c, run(m, s))) for s in SEEDS]
            for c, (_, models, run, _) in CONFIGS.items() for m in models}


def complete(runs, d):
    """True if every seed has dataset d."""
    return all(r is not None and d in r[0] for r in runs)


def fmt(values):
    return f"{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}" if len(values) > 1 else f"{values[0]:.2f}"


def paired(a_list, ref_list, threshold):
    """Paired comparison a − ref, one per seed (max p over seeds, as in phase 1), with neutral zero medians.

    Unlike phase2_checks_report.paired, a seed whose median difference is zero (|median| ≤ TIE_TOL: more
    than half of the instances tie) is neutral: it neither supports nor contradicts a direction. The other
    seeds' medians must agree in sign; if every median is zero, the direction comes from the mean difference.
    """
    res = [summarize.paired_compare(a, r) for a, r in zip(a_list, ref_list)]
    avg = {k: float(np.mean([x[k] for x in res])) for k in res[0]}
    p_max = max(x["p"] for x in res)
    signs = {np.sign(x["median_diff"]) for x in res if abs(x["median_diff"]) > TIE_TOL}
    if len(signs) > 1:
        verdict = "n.s."                                  # seeds disagree on the direction
    else:
        direction = signs.pop() if signs else np.sign(avg["mean_diff"])
        verdict = ("better" if direction < 0 else "worse") if p_max < threshold and direction != 0 else "n.s."
    return {**avg, "p_max": p_max, "verdict": verdict}


# ---------------------------------------------------------------- tables

def table_gaps(data, baselines, with_2opt):
    suffix = "+2opt" if with_2opt else ""
    head = [f"{DATASETS[d]} {lab}" for d in DATASETS for lab in DECODERS.values()]
    lines = ["| Model | Configuration | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(head)]
    for b, name in summarize.BASELINES.items():
        cells = [f"{baselines[d][b + suffix]['gap_mean']:.2f}" if d in baselines else "missing"
                 for d in DATASETS]
        lines.append(f"| {name} | baseline | " + " | ".join(x for c in cells for x in (c, "—", "—")) + " |")
    for (c, m), runs in data.items():
        cells = []
        for d in DATASETS:
            for dec in DECODERS:
                cells.append(fmt([r[0][d][dec + suffix]["gap_mean"] for r in runs]) if complete(runs, d)
                             else "missing")
        lines.append(f"| {MODELS[m]} | {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def beam_tests(data, base):
    """[(config, model, dataset, reference, result)]: beam 100 + 2-opt vs. each reference, all configurations."""
    tests = []
    for (c, m), runs in data.items():
        for d in DATASETS:
            if not complete(runs, d):
                continue
            beam = [r[1][f"{d}__beam{WIDTHS[-1]}+2opt"] for r in runs]
            refs = {"own greedy + 2-opt": [r[1][f"{d}__greedy+2opt"] for r in runs]}
            refs.update({lab: [base[f"{d}__{b}"]] * len(runs) for b, lab in BASE_REFS.items()
                         if f"{d}__{b}" in base})
            tests += [(c, m, d, lab, beam, ref) for lab, ref in refs.items()]
    return tests


def table_tests(tests):
    threshold = ALPHA / max(len(tests), 1)
    lines = [f"Diff. = beam {WIDTHS[-1]} + 2-opt − reference (negative = beam search is better). "
             f"Bonferroni: {ALPHA}/{len(tests)} tests, threshold {threshold:.2e}. *Equal* = difference ≤ {TIE_TOL:g} pp. "
             "Two-sided Wilcoxon test per seed (ties dropped); *max p* is the largest p-value across the 3 seeds; "
             "significant only if max p is below the Bonferroni threshold and the medians of the seeds agree in sign; "
             f"a seed whose median is zero (|median| ≤ {TIE_TOL:g} pp, i.e. more than half of the instances tie) is "
             "neutral, and if every median is zero the direction comes from the mean difference. The baselines are deterministic, so every seed is compared with the same baseline gaps.", "",
             "| Configuration | Model | Dataset | Reference | Beam gap | Ref. gap | " + checks.HEADER[0],
             "|---|---|---|---|---|---|" + checks.HEADER[1]]
    verdicts = []
    for c, m, d, lab, beam, ref in tests:
        r = paired(beam, ref, threshold)
        verdicts.append((c, m, d, lab, r))
        label = (f"{c} | {MODELS[m]} | {DATASETS[d]} | {lab} | {np.mean([b.mean() for b in beam]):.2f} | "
                 f"{np.mean([x.mean() for x in ref]):.2f}")
        lines.append(checks.row(label, r))
    return "\n".join(lines), verdicts, threshold


def table_times(data):
    """Mean decoding time per instance (ms), averaged over every run that has the dataset."""
    keys = [k for dec in DECODERS for k in (dec, dec + "+2opt")]
    labels = [lab + suffix for lab in DECODERS.values() for suffix in ("", " + 2-opt")]
    lines = ["| Dataset | " + " | ".join(labels) + " | runs |", "|---|" + "---|" * (len(keys) + 1)]
    for d in DATASETS:
        runs = [r for rs in data.values() for r in rs if r is not None and d in r[0]]
        if not runs:
            lines.append(f"| {DATASETS[d]} | " + " | ".join(["missing"] * len(keys)) + " | 0 |")
            continue
        cells = [f"{np.mean([r[0][d][k]['decode_ms_per_inst'] for r in runs]):.1f}" for k in keys]
        lines.append(f"| {DATASETS[d]} | " + " | ".join(cells) + f" | {len(runs)} |")
    return "\n".join(lines)


def table_worse(data):
    """% of instances (over all seeds) where a longer-beam decoder gives a longer tour, without 2-opt."""
    pairs = [(f"beam{WIDTHS[0]}", "greedy"), (f"beam{WIDTHS[-1]}", "greedy"), (f"beam{WIDTHS[-1]}", f"beam{WIDTHS[0]}")]
    head = [f"{DATASETS[d]}: {DECODERS[a]} worse than {DECODERS[b]}" for d in DATASETS for a, b in pairs]
    lines = ["| Model | Configuration | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(head)]
    for (c, m), runs in data.items():
        cells = []
        for d in DATASETS:
            for a, b in pairs:
                if complete(runs, d):
                    worse = np.concatenate([r[1][f"{d}__{a}"] > r[1][f"{d}__{b}"] + TIE_TOL for r in runs])
                    cells.append(f"{100 * worse.mean():.1f}%")
                else:
                    cells.append("missing")
        lines.append(f"| {MODELS[m]} | {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def consistency(data, results):
    """Greedy gaps of this run vs. the earlier evaluation of the same checkpoint, instance by instance."""
    lines = ["| Configuration | Model | Dataset | Instances compared | Different (> 1e-6 pp) | max abs. diff. (pp) |",
             "|---|---|---|---|---|---|"]
    for (c, m), runs in data.items():
        earlier_of = CONFIGS[c][3]
        for d in DATASETS:
            n, diff_count, max_diff = 0, 0, 0.0
            for s, r in zip(SEEDS, runs):
                if r is None or d not in r[0]:
                    continue
                old = None
                for folder in earlier_of(m, s):
                    loaded = batch2.load_folder(os.path.join(results, folder))
                    if loaded and f"{d}__greedy" in loaded[1]:
                        old = loaded[1][f"{d}__greedy"]
                if old is None or len(old) != len(r[1][f"{d}__greedy"]):
                    continue
                diff = np.abs(r[1][f"{d}__greedy"] - old)
                n, diff_count, max_diff = n + len(diff), diff_count + int((diff > TIE_TOL).sum()), max(max_diff, diff.max())
            if n:
                lines.append(f"| {c} | {MODELS[m]} | {DATASETS[d]} | {n} | {diff_count} | {max_diff:.2e} |")
            else:
                lines.append(f"| {c} | {MODELS[m]} | {DATASETS[d]} | no earlier evaluation | — | — |")
    return "\n".join(lines)


def summary_points(data, verdicts):
    points = []
    b = WIDTHS[-1]
    for d in ("tsp50_test", "tsp100_test"):
        parts = []
        for (c, m), runs in data.items():
            if complete(runs, d):
                g = np.mean([r[0][d]["greedy"]["gap_mean"] for r in runs])
                gb = np.mean([r[0][d][f"beam{b}"]["gap_mean"] for r in runs])
                g2 = np.mean([r[0][d]["greedy+2opt"]["gap_mean"] for r in runs])
                gb2 = np.mean([r[0][d][f"beam{b}+2opt"]["gap_mean"] for r in runs])
                parts.append(f"{MODELS[m]} {c} {g:.2f} → {gb:.2f} (+ 2-opt {g2:.2f} → {gb2:.2f})")
        points.append(f"**{DATASETS[d]}, greedy → B={b}**: " + ("; ".join(parts) if parts else "missing") + ".")
    for d in ("tsp50_test", "tsp100_test"):
        both = [f"{MODELS[m]} {c}" for (c, m) in data
                if all(any(v[:4] == (c, m, d, lab) and v[4]["verdict"] == "better" for v in verdicts)
                       for lab in BASE_REFS.values())]
        points.append(f"**{DATASETS[d]}: B={b} + 2-opt significantly better than both baselines + 2-opt**: "
                      + (", ".join(both) if both else "none") + ".")
    return "\n".join(f"{i}. {p}" for i, p in enumerate(points, 1))


def build_report(results):
    data = load_all(results)
    baselines = batch2.load_baselines(results)
    base = checks.load_base_per_instance(results)
    tests = beam_tests(data, base)
    t_tests, verdicts, _ = table_tests(tests)
    missing = [f"{c}/{CONFIGS[c][2](m, s)}" for (c, m), runs in data.items() for s, r in zip(SEEDS, runs) if r is None]
    lines = [
        "# Phase 2 — beam search decoding",
        "",
        f"Generated by `scripts/phase2_beam_report.py` on {datetime.date.today().isoformat()}.",
        "No model was retrained: the same edge probabilities are decoded with greedy and with beam search "
        f"(B = {', '.join(map(str, WIDTHS))}), each without and with 2-opt (2-opt is applied to the tour that beam "
        "search returns). Test graphs: k = 0.5 n (k = 10 on TSP-20, 25 on TSP-50, 50 on TSP-100). Seeds 0, 1, 2.",
        "",
        "Beam search (`tsp_gnn.decode.beam_search_decode`) works on the same edge choices as greedy: a partial "
        "solution is a set of accepted edges; it is extended by one valid edge at a time, partial solutions are "
        "ranked by the sum of the log-probabilities of their edges (pairs outside the graph: log 1e-12), duplicate "
        "edge sets are removed, and the B best are kept. At the end the shortest of the B tours is returned "
        "(selection by length, as in Joshi et al. 2019). With B = 1 it is exactly greedy. It is *pure* beam "
        "search: the greedy tour can be pruned from the beam, so on some instances beam search is worse than "
        "greedy (Table 4).",
        "",
        "Configurations:",
        "",
        *[f"- `{c}` — {desc} ({', '.join(MODELS[m] for m in models)})" for c, (desc, models, _, _) in CONFIGS.items()],
        "",
        "## Summary",
        "",
        summary_points(data, verdicts),
        "",
        "## Table 1 — mean gap to LKH (%), without 2-opt",
        "",
        "Models: mean ± std across the 3 seeds. Baselines are deterministic (no beam).",
        "",
        table_gaps(data, baselines, with_2opt=False),
        "",
        "## Table 2 — mean gap to LKH (%), with 2-opt",
        "",
        table_gaps(data, baselines, with_2opt=True),
        "",
        f"## Table 3 — beam {WIDTHS[-1]} + 2-opt against greedy + 2-opt and the baselines + 2-opt",
        "",
        t_tests,
        "",
        "## Table 4 — instances where beam search is worse (no 2-opt)",
        "",
        "Percentage of instances (all 3 seeds together) where the tour is longer than with the other decoder.",
        "",
        table_worse(data),
        "",
        "## Table 5 — mean decoding time per instance (ms)",
        "",
        "Decoding only (CPU, NumPy, one instance at a time, 3 evaluations running in parallel at niceness 10); "
        "the network's forward pass is not included. *+ 2-opt* = decoder + 2-opt.",
        "",
        table_times(data),
        "",
        "## Table 6 — consistency check: greedy here vs. the earlier evaluation of the same checkpoint",
        "",
        "Same checkpoint, same test graphs: the greedy gaps should be identical (tiny differences can come from "
        "non-deterministic GPU sums that flip near-ties).",
        "",
        consistency(data, results),
        "",
        "## Missing results",
        "",
        ("None." if not missing else "Not found (failed or not yet run): " + ", ".join(missing) + "."),
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 beam search report.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/phase2_beam_report.md")
    args = parser.parse_args()
    report = build_report(args.results)
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
