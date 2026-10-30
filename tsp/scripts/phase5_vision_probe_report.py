"""Report of the phase 5 vision probe (scripts/vision_probe.py).

Reads results/phase5/vision_probe/ (cells.npz, images.json, tune_*.json, answers.jsonl, policy_*.npz, wait.log) and
the notes results/phase5/vision_probe_notes.md. Writes results/phase5/vision_probe.md and docs/phase5_vision_probe.md.

Per cell (pooled over the instances, cells with at least one window): target = the cell's share of windows that LKH
improves is above the mean share of its instance; AP of each ranking. Top-4 capture: share of the instance's total
LKH window gain that lies in the 4 cells a ranking puts first. Policy: LKH on the windows of the top-4 cells.

Example:
    python scripts/phase5_vision_probe_report.py --model gemma4:e4b-it-qat
"""

import argparse
import datetime
import importlib.util
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from tsp_gnn import vision_probe as V

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("summarize", os.path.join(_here, "summarize.py"))
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)

ROOT = "results/phase5/vision_probe"
OUT = "results/phase5/vision_probe.md"
DOCS = "docs/phase5_vision_probe.md"
NOTES = "results/phase5/vision_probe_notes.md"
KIND = {"a": "(a) tour only", "b": "(b) tour coloured by p"}


def fmt(v, d=3):
    v = [x for x in v if x == x]
    return "—" if not v else (f"{np.mean(v):.{d}f}" if len(v) == 1 else f"{np.mean(v):.{d}f} ± {np.std(v):.{d}f}")


def signal_rows(cells, answers):
    num = len(cells["label"])
    has = cells["windows"] > 0
    lab = cells["label"]
    target = np.array([lab[i] > np.nanmean(lab[i]) for i in range(num)])
    gain = np.nan_to_num(cells["gain"])

    def evaluate(scores):
        """scores: (num, 16) array (rows with nan = no ranking)."""
        ok = ~np.isnan(scores).any(axis=1)
        m = has & ok[:, None]
        ap = float(average_precision_score(target[m], scores[m]))
        rho = [spearmanr(scores[i][has[i]], lab[i][has[i]])[0] for i in range(num) if ok[i] and has[i].sum() > 2]
        cap = [gain[i][np.array(V.top_cells(scores[i])) - 1].sum() / max(gain[i].sum(), 1e-12)
               for i in range(num) if ok[i]]
        return ap, float(np.nanmean(rho)), float(np.mean(cap)), int(ok.sum())

    rows = {}
    rng = np.random.default_rng(0)
    tie = rng.random((num, 16)) * 1e-6
    rows["Disagreement (mean per cell)"] = [evaluate(np.nan_to_num(cells["disagreement"], nan=-1.0) + tie)]
    rows["Random"] = [evaluate(np.random.default_rng(r).random((num, 16))) for r in range(3)]
    rows["Oracle (LKH gain per cell)"] = [evaluate(gain + tie)]
    for kind in "ab":
        reps = []
        for rep in sorted({e["rep"] for e in answers}):
            sc = np.full((num, 16), np.nan)
            for e in answers:
                if e["kind"] == kind and e["rep"] == rep and e["ranking"] and e["instance"] < num:
                    sc[e["instance"]] = V.ranking_scores(e["ranking"], np.random.default_rng([rep, e["instance"]]))
            reps.append(evaluate(sc))
        rows[f"Vision {KIND[kind]}"] = reps
    base = float(target[has].mean())
    return rows, base


def table_signal(rows, base):
    lines = ["| Ranking | AP (target: cell above its instance's mean) | Spearman with the cell's improving share | "
             "Top-4 cells: share of the LKH gain | Instances ranked |", "|---|---|---|---|---|"]
    for name, reps in rows.items():
        lines.append(f"| {name} | {fmt([r[0] for r in reps])} | {fmt([r[1] for r in reps])} | "
                     f"{fmt([r[2] for r in reps])} | {fmt([r[3] for r in reps], 0)} |")
    lines += ["", f"Random AP = positive rate = {base:.3f}. Mean ± std over the 3 repetitions (vision: temperature 0.5, "
              "seeds 0-2; random: 3 draws). Top-4 capture of a random choice ≈ 0.25."]
    return lines


def table_answers(answers):
    lines = ["| Image | Calls | Unusable answers (%) | Cells listed (mean) | Latency per call (s) | "
             "Top-4 agreement between repetitions (Jaccard) |", "|---|---|---|---|---|---|"]
    for kind in "ab":
        es = [e for e in answers if e["kind"] == kind]
        if not es:
            continue
        bad = np.mean([not e["ranking"] for e in es])
        listed = np.mean([len(e["ranking"]) for e in es if e["ranking"]]) if any(e["ranking"] for e in es) else 0
        jac = []
        for i in sorted({e["instance"] for e in es}):
            tops = [set(V.top_cells(V.ranking_scores(e["ranking"], np.random.default_rng([e["rep"], i]))))
                    for e in es if e["instance"] == i and e["ranking"]]
            for a in range(len(tops)):
                for b in range(a + 1, len(tops)):
                    jac.append(len(tops[a] & tops[b]) / len(tops[a] | tops[b]))
        lines.append(f"| {KIND[kind]} | {len(es)} | {100 * bad:.1f} | {listed:.1f} | "
                     f"{np.mean([e['seconds'] for e in es]):.1f} | {np.mean(jac) if jac else float('nan'):.2f} |")
    return lines


def table_policy(P, answers, images):
    lat = {k: np.mean([e["seconds"] for e in answers if e["kind"] == k]) for k in "ab" if any(e["kind"] == k for e in answers)}
    extra = {"a": images["seconds_mean"]["a"], "b": images["seconds_mean"]["b"] + images["seconds_mean"]["edge_p"]}
    init = float(np.mean(P["initial__s"]))
    rows = [("P2 top 20 % by disagreement (batched)", ["p2_top20"], 0.0),
            ("Top-4 cells by disagreement", ["disagreement"], 0.0),
            ("Top-4 random cells (3 draws)", [f"random_r{r}" for r in range(3)], 0.0),
            ("Top-4 cells, oracle (LKH gain)", ["oracle"], 0.0)]
    for k in "ab":
        keys = [key[:-5] for key in P if key.startswith(f"vision_{k}_r") and key.endswith("__gap")]
        if keys:
            rows.append((f"Top-4 cells, vision {KIND[k]}", sorted(keys), lat.get(k, 0.0) + extra[k]))
    lines = ["| Policy | Gap (%) | Gain (pp) | Windows to LKH (%) | Time (s) | of which the model and the image (s) |",
             "|---|---|---|---|---|---|"]
    g0 = float(np.mean(P["initial__gap"]))
    per = {}
    for label, keys, add in rows:
        gaps = np.nanmean([P[f"{k}__gap"] for k in keys], axis=0)
        per[label] = gaps
        secs = init + np.nanmean([np.nanmean(P[f"{k}__s"]) for k in keys]) + add
        frac = np.nanmean([np.nanmean(P[f"{k}__lkh_frac"]) for k in keys])
        lines.append(f"| {label} | {np.nanmean(gaps):.3f} | {g0 - np.nanmean(gaps):.3f} | {100 * frac:.1f} | "
                     f"{secs:.2f} | {add:.2f} |")
    lines += ["", f"Initial gap {g0:.3f} % (greedy + 2-opt, {init:.2f} s per instance, included in every time). Up to 3 "
              "passes, w = 20. Vision time = mean model latency per call + drawing the image (+ the network's edge "
              "probabilities for (b))."]
    tests = ["", "| Vision policy | vs. | Better / equal / worse (% inst.) | Mean diff (pp) | p (Wilcoxon) |",
             "|---|---|---|---|---|"]
    for label in [l for l in per if l.startswith("Top-4 cells, vision")]:
        for other in ("Top-4 cells by disagreement", "Top-4 random cells (3 draws)"):
            ok = ~np.isnan(per[label]) & ~np.isnan(per[other])
            t = summarize.paired_compare(per[label][ok], per[other][ok])
            tests.append(f"| {label} | {other} | {t['better']:.0f} / {t['equal']:.0f} / {t['worse']:.0f} | "
                         f"{t['mean_diff']:+.3f} | {t['p']:.1e} |")
    return lines + tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4:e4b-it-qat")
    args = parser.parse_args()
    tag = args.model.replace(":", "_")
    cells = dict(np.load(os.path.join(ROOT, "cells.npz")))
    images = json.load(open(os.path.join(ROOT, "images.json")))
    answers = [e for e in map(json.loads, open(os.path.join(ROOT, "answers.jsonl"))) if e["model"] == args.model]
    tune = json.load(open(os.path.join(ROOT, f"tune_{tag}.json")))
    P = dict(np.load(os.path.join(ROOT, f"policy_{tag}.npz")))
    rows, base = signal_rows(cells, answers)
    notes = open(NOTES).read().strip() if os.path.exists(NOTES) else f"(notes missing: {NOTES})"
    wait = open(os.path.join(ROOT, "wait.log")).read().strip() if os.path.exists(os.path.join(ROOT, "wait.log")) else "—"
    layers = tune.get("layers")
    lines = ["# Phase 5 — vision probe: can a vision model see where LKH will gain?", "",
             f"Generated {datetime.date.today().isoformat()} by scripts/phase5_vision_probe_report.py. Model "
             f"{args.model} on the private Ollama (port 11435), num_gpu = {tune['chosen_num_gpu']} ("
             + (f"{layers[0]}/{layers[1]} layers on the GPU)" if layers else "layers on the GPU not logged)"),
             "", notes, "",
             "## Set-up", "",
             f"- {images['num']} Fu-1k instances (the learned-triage window logs, TAGCN seed 0), greedy + 2-opt tour; "
             f"images {images['size']} × {images['size']} with a numbered 4 × 4 grid: (a) tour only, (b) tour edges "
             "coloured by the network's p (green high, red low). At 768 px the 1000 cities, the tour and the cell "
             "numbers are legible (checked by eye).",
             f"- Windows per cell: {images['windows_per_cell']:.1f} on average (w = 20, offsets 0 and 10; a window "
             "belongs to the cell of its centroid).",
             f"- Model fitting: {json.dumps(tune['trials'])}; latency of 3 test calls: "
             f"{[round(x['seconds'], 1) for x in tune['latency']]} s; GPU used by other processes at tuning: "
             f"{tune['gpu_others_mib']} MiB.", "",
             "Start (waiting for the GPU):", "", "```", wait, "```", "",
             "## Answers", "", *table_answers(answers), "",
             "## Signal per cell", "", *table_signal(rows, base), "",
             "## Policy: LKH on the windows of the top-4 cells", "", *table_policy(P, answers, images), ""]
    os.makedirs("docs", exist_ok=True)
    text = "\n".join(lines) + "\n"
    open(OUT, "w").write(text)
    open(DOCS, "w").write(text)
    print(f"wrote {OUT} and {DOCS}")


if __name__ == "__main__":
    main()
