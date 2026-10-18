"""Quick tests for scripts/phase2_exp1_table.py on a fake results/ folder.  Run with: pytest tests/"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

# scripts/ is not a package: we load the file directly.
_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "phase2_exp1_table.py")
_spec = importlib.util.spec_from_file_location("phase2_exp1_table", _path)
table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(table)

CONFIG = {"model": "gcn", "hidden": 64, "layers": 6, "k": 10, "epochs": 20, "seed": 0, "pos_weight": 5.0}


def write_eval(folder, gaps, config):
    """Write an eval.json + eval_per_instance.npz with the given per-instance gaps {dataset: array}."""
    os.makedirs(folder, exist_ok=True)
    ev = {"meta": {"config": config, "train_git_commit": "abc1234"}}
    per_instance = {}
    for d, g in gaps.items():
        ev[d] = {m: {"gap_mean": float(g.mean())} for m in table.METHODS}
        per_instance.update({f"{d}__{m}": g for m in table.METHODS})
    with open(os.path.join(folder, "eval.json"), "w") as f:
        json.dump(ev, f)
    np.savez_compressed(os.path.join(folder, "eval_per_instance.npz"), **per_instance)


def write_baselines(path, datasets, num=100):
    entry = {m: {"gap_mean": 1.0} for k in table.BASELINES for m in (k, k + "+2opt")}
    with open(path, "w") as f:
        json.dump({"meta": {}, **{d: entry for d in datasets}}, f)
    np.savez_compressed(os.path.splitext(path)[0] + "_per_instance.npz",
                        **{f"{d}__{m}": np.ones(num) for d in datasets for m in entry})


def fake_results(root, num=100):
    """knn_mean is 1 pp better than none on every instance, for every model and seed."""
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(root, "phase2_exp1"))
    write_baselines(os.path.join(root, "baselines.json"), ["tsp20_test", "tsp50_test"], num)
    write_baselines(os.path.join(root, "phase2_exp1", "baselines_tsp100.json"), ["tsp100_test"], num)
    for m in table.MODELS:
        for s in table.SEEDS:
            gaps = {d: 5 + rng.random(num) for d in table.DATASETS}
            cfg = {**CONFIG, "model": m, "seed": s}  # phase 1 configs have no edge_norm
            write_eval(os.path.join(root, "runs", f"{m}_seed{s}"),
                       {d: gaps[d] for d in ("tsp20_test", "tsp50_test")}, cfg)
            write_eval(os.path.join(root, "phase2_exp1", "phase1_tsp100", f"{m}_seed{s}"),
                       {"tsp100_test": gaps["tsp100_test"]}, cfg)
            write_eval(os.path.join(root, "runs", f"{m}_knnmean_seed{s}"),
                       {d: g - 1 for d, g in gaps.items()}, {**cfg, "edge_norm": "knn_mean"})


def test_report_on_fake_results(tmp_path, monkeypatch):
    fake_results(str(tmp_path))
    out = tmp_path / "table.md"
    monkeypatch.setattr(sys, "argv", ["x", "--results", str(tmp_path), "--out", str(out)])
    table.main()
    report = out.read_text()
    # Table 2: 3 models × 3 datasets × 2 methods = 18 rows, all significant in favour of knn_mean
    assert report.count("**knn_mean better**") == 18
    # Table 1: one row per baseline and per model × edge_norm
    for name in list(table.BASELINES.values()) + [f"{n} | {norm}" for n in table.MODELS.values()
                                                  for norm in table.NORMS]:
        assert f"| {name} |" in report
    assert "0.05/18 tests" in report


def test_missing_dataset_fails(tmp_path):
    fake_results(str(tmp_path))
    os.remove(tmp_path / "phase2_exp1" / "phase1_tsp100" / "gat_seed1" / "eval.json")
    with pytest.raises(FileNotFoundError):
        table.load_all(str(tmp_path))


def test_wrong_edge_norm_is_detected(tmp_path):
    fake_results(str(tmp_path))
    # pretend a knn_mean folder holds a model trained without normalization
    write_eval(str(tmp_path / "runs" / "gated_knnmean_seed2"), {d: np.ones(100) for d in table.DATASETS},
               {**CONFIG, "model": "gated", "seed": 2, "edge_norm": "none"})
    with pytest.raises(ValueError):
        table.check_consistency(table.load_all(str(tmp_path)))
