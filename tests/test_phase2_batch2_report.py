"""Quick tests for scripts/phase2_batch2_report.py on a fake results/ folder.  Run with: pytest tests/"""

import importlib.util
import json
import os

import numpy as np

# scripts/ is not a package: we load the file directly.
_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "phase2_batch2_report.py")
_spec = importlib.util.spec_from_file_location("phase2_batch2_report", _path)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

NUM = 100
# Shift of each configuration's gaps relative to phase 1 (negative = better).
SHIFT = {"knn_mean": -1.0, "kfrac": +1.0, "knn_mean_kfrac": 0.0, "bn_batch": -1.0,
         "layernorm": -1.0, "layernorm_kfrac": -1.0, "train50": -3.0}


def write_eval(folder, gaps):
    os.makedirs(folder, exist_ok=True)
    ev = {"meta": {}}
    for d, g in gaps.items():
        ev[d] = {m: {"gap_mean": float(g.mean())} for m in report.METHODS}
    with open(os.path.join(folder, "eval.json"), "w") as f:
        json.dump(ev, f)
    np.savez_compressed(os.path.join(folder, "eval_per_instance.npz"),
                        **{f"{d}__{m}": g for d, g in gaps.items() for m in report.METHODS})


def write_baselines(path, datasets, gap=10.0):
    entry = {m: {"gap_mean": gap} for k in report.BASELINES for m in (k, k + "+2opt")}
    with open(path, "w") as f:
        json.dump({"meta": {}, **{d: entry for d in datasets}}, f)


def fake_results(root, skip=()):
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(root, "phase2_exp1"))
    write_baselines(os.path.join(root, "baselines.json"), ["tsp20_test", "tsp50_test"])
    write_baselines(os.path.join(root, "phase2_exp1", "baselines_tsp100.json"), ["tsp100_test"])
    for m in report.MODELS:
        for s in report.SEEDS:
            # phase 1: 4% on TSP-20 (60% better than greedy), 9% on TSP-50/100
            base = {"tsp20_test": 4 + rng.random(NUM), "tsp50_test": 9 + rng.random(NUM),
                    "tsp100_test": 9 + rng.random(NUM)}
            folders = report.CONFIGS["phase1"][1](m, s)
            write_eval(os.path.join(root, folders[0]), {d: base[d] for d in ("tsp20_test", "tsp50_test")})
            write_eval(os.path.join(root, folders[1]), {"tsp100_test": base["tsp100_test"]})
            for c, shift in SHIFT.items():
                if c in skip:
                    continue
                datasets = report.expected_datasets(c)
                write_eval(os.path.join(root, report.CONFIGS[c][1](m, s)[0]),
                           {d: base[d] + shift for d in datasets})


def test_report_on_fake_results(tmp_path):
    fake_results(str(tmp_path))
    text = report.build_report(str(tmp_path))
    rows = text.split("## Table 2")[1].split("## Decisions")[0]
    # 3 models × 2 methods × datasets: knn_mean 3, kfrac 2, knn_mean_kfrac 2, bn_batch 3,
    # layernorm 3, layernorm_kfrac 3, train50 2  ->  18 × 6 = 108 tests
    assert "0.05/108 tests" in text
    assert rows.count("| knn_mean | ") == 18 and rows.count("| kfrac | ") == 12
    assert rows.count("**better**") == 4 * 18 + 12            # knn_mean, bn_batch, layernorm(+kfrac), train50
    assert rows.count("**worse**") == 12                      # kfrac is 1 pp worse everywhere
    assert "None: every configuration has all 3 seeds." in text
    # train50 on TSP-50: 6.5% vs. greedy 10% = 35% reduction < 0.75 × 55% on TSP-20 -> capacity
    assert "capacity/training" in text
    summary = text.split("## Summary")[1].split("## Table 1")[0]
    assert sum(line[:3].rstrip(".").strip().isdigit() for line in summary.splitlines() if line) <= 10


def test_report_with_failed_step(tmp_path):
    fake_results(str(tmp_path), skip=("layernorm", "layernorm_kfrac", "train50"))
    text = report.build_report(str(tmp_path))
    assert "**layernorm**: no results" in text and "**Control (train on TSP-50)**: no results" in text
    assert "| GCN | layernorm | missing |" in text
    assert "layernorm / GCN / TSP-20" in text.split("## Missing results")[1]
