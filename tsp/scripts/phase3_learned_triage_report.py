"""Report of the learned triage signal (phase 3, pending item).

Reads results/phase3_learned_triage/features/ (step 1), models/ (step 2: metrics.json per TAGCN seed), e2/ (step 3:
ranked policies run with scripts/triage_ablation.py, random / disagreement / learned / oracle, all re-run together
so that times are comparable) and the opening notes results/phase3_learned_triage_notes.md (if present).
Writes results/phase3_learned_triage_report.md and a copy in docs/.

Example:
    python scripts/phase3_learned_triage_report.py
"""

import datetime
import importlib.util
import json
import os
import shutil

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("summarize", os.path.join(_here, "summarize.py"))
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)

ROOT = "results/phase3_learned_triage"
OUT = "results/phase3_learned_triage_report.md"
DOCS = "docs/phase3_learned_triage_report.md"
NOTES = "results/phase3_learned_triage_notes.md"
SEEDS = (0, 1, 2)
RANDOM_SEEDS = (0, 1, 2, 3, 4)
FRACS = (20, 50)
ALPHA = 0.05
SETS = {"tsp200_test": ("validation half", 200), "tsp1000_test_concorde": ("validation half (Fu-1k)", 1000),
        "tsp1000_test": ("test", 1000), "tsp10000_test": ("test", 10000),
        "tsp10000_test_concorde": ("test (Fu-10k)", 10000)}
TEST_SETS = ("tsp1000_test", "tsp10000_test", "tsp10000_test_concorde")
LABELS = {"random": "Random (5 seeds)", "disagreement": "Disagreement (3 seeds)", "learned": "Learned (3 seeds)",
          "oracle": "Oracle: real LKH gain"}
MODEL_NAMES = {"logreg": "logistic regression", "hgb": "gradient boosting (classifier)",
               "hgb_reg": "gradient boosting (regressor on the gain)"}


def ms(values):
    values = [v for v in values if v == v]
    if not values:
        return "—"
    return f"{np.mean(values):.3f}" if len(values) == 1 else f"{np.mean(values):.3f} ± {np.std(values):.3f}"


def load_metrics():
    out = {}
    for s in SEEDS:
        p = os.path.join(ROOT, "models", f"tagcn_k3_seed{s}", "metrics.json")
        if os.path.exists(p):
            out[s] = json.load(open(p))
    return out


def npz(path):
    return dict(np.load(path)) if os.path.exists(path) else None


def policy(name, score, x):
    """Per-instance gap (mean over the runs), mean time (s, greedy + refinement), per-run mean gaps, LKH share,
    and the mean network / score time (s)."""
    if score == "random":
        r = npz(os.path.join(ROOT, "e2", "random", f"{name}.npz"))
        runs = [] if r is None else [(r, f"random_s{s}_top{x}") for s in RANDOM_SEEDS if f"random_s{s}_top{x}__gap" in r]
    elif score in ("disagreement", "learned"):
        runs = [(npz(os.path.join(ROOT, "e2", f"{score}_seed{s}", f"{name}.npz")), f"{score}_top{x}") for s in SEEDS]
        runs = [(r, k) for r, k in runs if r is not None]
    else:
        r = npz(os.path.join(ROOT, "e2", score, f"{name}.npz"))
        runs = [] if r is None else [(r, f"{score}_top{x}")]
    if not runs:
        return None
    gaps = np.array([r[f"{k}__gap"] for r, k in runs])
    secs = np.array([((r["initial__ms"] + r[f"{k}__ms"]) / 1000).mean() for r, k in runs])
    return {"per_inst": gaps.mean(axis=0), "run_means": gaps.mean(axis=1), "s": float(secs.mean()),
            "lkh": float(np.mean([r[f"{k}__to_lkh"].sum() / r[f"{k}__windows"].sum() for r, k in runs])),
            "net_s": float(np.mean([r[f"{k}__ms_net"].mean() / 1000 for r, k in runs])),
            "score_s": float(np.mean([r[f"{k}__ms_score"].mean() / 1000 for r, k in runs])),
            "initial": float(runs[0][0]["initial__gap"].mean()), "runs": len(runs)}


# ---------------------------------------------------------------------------------------------------- tables

def table_data():
    lines = ["| Set | Role | n | Instances | Windows (per seed) | LKH improves (%) | Mean window gain (%) |",
             "|---|---|---|---|---|---|---|"]
    for name, (role, n) in SETS.items():
        p = os.path.join(ROOT, "features", "tagcn_k3_seed0", f"{name}.json")
        if not os.path.exists(p):
            lines.append(f"| {name} | {role} | {n} | missing | | | |")
            continue
        d = json.load(open(p))
        role = "train + validation (halves by instance)" if role.startswith("validation") else role
        lines.append(f"| {name} | {role} | {n} | {d['meta']['instances']} | {d['windows']} | "
                     f"{100 * d['improves_frac']:.1f} | {d['mean_gain_rel_pct']:.2f} |")
    return lines


def table_selection(M):
    lines = ["| Model | Hyper-parameters chosen (per seed) | Validation AP, tsp200 | Validation AP, Fu-1k | "
             "Chosen (seeds) |", "|---|---|---|---|---|"]
    for kind, label in MODEL_NAMES.items():
        params = "; ".join(json.dumps(M[s]["models"][kind]["params"]) for s in M)
        lines.append(f"| {label} | {params} | {ms([M[s]['models'][kind]['val_ap']['tsp200_test'] for s in M])} | "
                     f"{ms([M[s]['models'][kind]['val_ap']['tsp1000_test_concorde'] for s in M])} | "
                     f"{sum(M[s]['chosen'] == kind for s in M)} / {len(M)} |")
    return lines


def table_ap(M):
    cols = [("random", "Random (= positive rate)"), ("simple", "Simple rule"), ("disagreement", "Disagreement"),
            ("logreg", "Logistic"), ("hgb", "GB classifier"), ("hgb_reg", "GB regressor"),
            ("chosen", "**Chosen**"), ("+log10_n", "Chosen + log10 n"), ("+logp", "Chosen + beam logp")]
    lines = ["| Set | Windows | " + " | ".join(c for _, c in cols) + " | Chosen − disagreement |",
             "|---|---|" + "---|" * (len(cols) + 1)]
    for name, (role, n) in SETS.items():
        vals = {}
        for key, _ in cols:
            v = []
            for s in M:
                m, test = M[s], name in TEST_SETS
                if key in ("random", "simple", "disagreement"):
                    v.append(m["raw"][name][key])
                elif key in MODEL_NAMES or key == "chosen":
                    kind = m["chosen"] if key == "chosen" else key
                    v.append(m["models"][kind]["test_ap" if test else "val_ap"][name])
                else:
                    v.append(m["ablations"][key[1:]]["test_ap" if test else "val_ap"][name])
            vals[key] = v
        diff = [a - b for a, b in zip(vals["chosen"], vals["disagreement"])]
        windows = M[next(iter(M))]["raw"][name]["windows"]
        lines.append(f"| {name} ({role}) | {windows} | " + " | ".join(ms(vals[k]) for k, _ in cols) +
                     f" | {np.mean(diff):+.3f} |")
    return lines


def table_policy():
    lines = ["| Set | x | Policy | Gap (%) | Gain (pp) | Share of oracle gain | Time (s) | of which network (s) | "
             "of which score (s) | Windows to LKH |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name in TEST_SETS:
        for x in FRACS:
            pol = {s: policy(name, s, x) for s in LABELS}
            orc = pol["oracle"]
            for s, label in LABELS.items():
                p = pol[s]
                if p is None:
                    lines.append(f"| {name} | {x} % | {label} | missing | | | | | | |")
                    continue
                gain = p["initial"] - p["per_inst"].mean()
                share = f"{100 * gain / (orc['initial'] - orc['per_inst'].mean()):.0f} %" if orc else "—"
                gap = (f"{p['per_inst'].mean():.3f}" if p["runs"] == 1 else
                       f"{p['per_inst'].mean():.3f} ± {np.std(p['run_means']):.3f}")
                star = " *" if s == "oracle" else ""
                lines.append(f"| {name} | {x} % | {label} | {gap} | {gain:.3f} | {share} | {p['s']:.2f}{star} | "
                             f"{p['net_s']:.2f} | {p['score_s']:.2f} | {100 * p['lkh']:.0f} % |")
    lines += ["", "w = 20, offsets 0 and w/2, ≤ 3 passes, from the greedy + 2-opt tour; in every half-pass exactly the "
              "top x % of the windows by the score go to LKH (E2 of the benchmark). Gap: mean over instances of the "
              "per-instance mean over runs (± std of the run means). Time: greedy + refinement, mean per instance. "
              "*score* = time to compute the scores outside the network (for the learned signal: instance context, "
              "features and predict). * the oracle runs LKH on every window to know its gain: its time is not that "
              "of a usable policy."]
    return lines


def table_tests():
    pairs = [(x, other) for x in FRACS for other in ("disagreement", "random")]
    lines = ["| Set | x | Learned vs. | Better / equal / worse (% inst.) | Mean diff (pp) | p (Wilcoxon) | Verdict |",
             "|---|---|---|---|---|---|---|"]
    for name in TEST_SETS:
        for x, other in pairs:
            a, b = policy(name, "learned", x), policy(name, other, x)
            if a is None or b is None:
                lines.append(f"| {name} | {x} % | {LABELS[other]} | missing | | | |")
                continue
            t = summarize.paired_compare(a["per_inst"], b["per_inst"])
            n = len(a["per_inst"])
            if n < 5:
                lines.append(f"| {name} ({n}) | {x} % | {LABELS[other]} | {t['better']:.0f} / {t['equal']:.0f} / "
                             f"{t['worse']:.0f} | {t['mean_diff']:+.3f} | — | no test (n = {n}) |")
                continue
            thr = ALPHA / len(pairs)
            verdict = "n.s." if t["p"] >= thr else ("**better**" if t["median_diff"] < 0 or
                                                    (t["median_diff"] == 0 and t["mean_diff"] < 0) else "**worse**")
            lines.append(f"| {name} ({n}) | {x} % | {LABELS[other]} | {t['better']:.0f} / {t['equal']:.0f} / "
                         f"{t['worse']:.0f} | {t['mean_diff']:+.3f} | {t['p']:.1e} | {verdict} |")
    lines += ["", "Paired two-sided Wilcoxon on the per-instance gaps (learned and disagreement averaged over their 3 "
              "TAGCN seeds, random over its 5 seeds); negative difference = learned shorter. Bonferroni within each "
              f"set: {ALPHA} / {len(pairs)} = {ALPHA / len(pairs):.4f}. tsp10000_test has 3 instances: no test."]
    return lines


def table_time():
    lines = ["| Set | Policy (sorted by time) | Time (s) | Gap (%) |", "|---|---|---|---|"]
    for name in TEST_SETS:
        rows = []
        for s in ("random", "disagreement", "learned"):
            for x in FRACS:
                p = policy(name, s, x)
                if p is not None:
                    rows.append((p["s"], p["per_inst"].mean(), f"{LABELS[s].split(' (')[0]} top {x} %"))
        for sec, gap, label in sorted(rows):
            lines.append(f"| {name} | {label} | {sec:.2f} | {gap:.3f} |")
    lines += ["", "The same-time question: a policy is worth its cost only if no cheaper policy reaches the same gap."]
    return lines


def table_importance(M):
    lines = ["| Feature | Logistic coefficient (standardised) | Permutation importance of the chosen model "
             "(validation AP drop) |", "|---|---|---|"]
    feats = M[next(iter(M))]["meta"]["features"]
    for f in feats:
        lines.append(f"| {f} | {ms([M[s]['logreg_coef'][f] for s in M])} | "
                     f"{ms([M[s]['permutation_importance'][f]['mean'] for s in M])} |")
    lines += ["", "Mean ± std over the 3 TAGCN seeds. Coefficients: sign = direction (positive = more likely that LKH "
              "improves); p_mean and disagreement are the same information (disagreement = 1 − p_mean), so their "
              "coefficients share it. Permutation importance: drop of the validation AP when the feature is shuffled "
              "(5 repeats)."]
    return lines


def main():
    M = load_metrics()
    notes = open(NOTES).read().strip() if os.path.exists(NOTES) else "(notes missing: " + NOTES + ")"
    commits = sorted({m["meta"]["git_commit"] for m in M.values()})
    lines = ["# Phase 3 — learned triage signal", "",
             f"Generated {datetime.date.today().isoformat()} by scripts/phase3_learned_triage_report.py (models at commit "
             f"{', '.join(commits) or '—'}). TAGCN K=3 (3 seeds), w = 20, LKH per window as in the triage experiment; "
             "sklearn on the CPU, network on the T4.", "", notes, "",
             "## Data", "", *table_data(), "",
             "Label: LKH improves the window (path shorter than the current one). Features: "
             "p mean / min / max on the segment's edges, disagreement, segment length / NN distances, longest edge / "
             "mean NN distance, density within 2 mean NN distances, position in the tour, edges outside the 5 nearest "
             "neighbours; no feature uses the LKH solution; n and the beam logp only as ablations. Training: half of "
             "the instances of tsp200 and of Fu-1k (same halves for the 3 seeds), each set with the same total "
             "weight; validation: the other halves; tests never used to fit or choose.", "",
             "## Model selection (validation)", "", *table_selection(M), "",
             "## Signal quality: average precision", "", *table_ap(M), "",
             "Mean ± std over the 3 TAGCN seeds (random, simple and the label do not depend on the seed). Validation "
             "rows: the validation halves; test rows: all windows of the set.", "",
             "## Policy: top x % of the windows to LKH (E2 protocol)", "", *table_policy(), "",
             "### Paired tests", "", *table_tests(), "",
             "### Same time", "", *table_time(), "",
             "## Feature importance", "", *table_importance(M), ""]
    os.makedirs(os.path.dirname(DOCS), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    shutil.copy(OUT, DOCS)
    print(f"wrote {OUT} and {DOCS}")


if __name__ == "__main__":
    main()
