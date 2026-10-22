"""Quick tests for scripts/phase2_scale_report.py on a fake results/ folder.  Run with: pytest tests/"""

import importlib.util
import json
import os

import numpy as np

_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
_spec = importlib.util.spec_from_file_location("phase2_scale_report", os.path.join(_dir, "phase2_scale_report.py"))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def _write(path_no_ext, summary, per_inst):
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    np.savez(path_no_ext + ".npz", **per_inst)
    with open(path_no_ext + ".json", "w") as f:
        json.dump(summary, f)


def fake_results(root, skip=()):
    """Baselines at ~10 %; every model 2 pp better on every instance and method."""
    results, data = os.path.join(root, "results"), os.path.join(root, "data")
    os.makedirs(data)
    rng = np.random.default_rng(0)
    for n in report.SIZES:
        num = 25 if n <= 1000 else 5
        np.savez(os.path.join(data, f"tsp{n}_test.npz"), lengths=np.ones(num), seed=n,
                 **({"lkh_seconds": np.full(num, n / 100), "lkh_runs": 1} if n >= 200 else {}))
        base = 10 + rng.random(num)
        meta = {"git_commit": "abc1234", "num": num}
        b_keys = ["greedy"] + [f"greedy+2opt@{m}" for m in report.K_MODES]
        if n <= 1000:
            b_keys += ["christofides"] + [f"christofides+2opt@{m}" for m in report.K_MODES]
        b_inst = {**{k: base for k in b_keys}, **{f"{k}__ms": np.full(num, 5.0) for k in b_keys}}
        _write(os.path.join(results, "phase2_scale", "baselines", f"tsp{n}_test"),
               {"meta": meta, **{k: {"gap_mean": float(base.mean()), "ms_mean": 5.0} for k in b_keys}}, b_inst)
        methods = ["greedy", "greedy+2opt"] + (["beam10", "beam10+2opt"] if n <= 1000 else []) \
            + (["beam100", "beam100+2opt"] if n == 200 else [])
        for m in report.MODELS:
            for mode in report.K_MODES:
                for s in report.SEEDS:
                    if (m, mode, n, s) in skip:
                        continue
                    inst = {**{k: base - 2 for k in methods}, **{f"{k}__ms": np.full(num, 3.0) for k in methods},
                            "graph__ms": np.full(num, 1.0), "gnn__ms": np.full(num, 2.0)}
                    summ = {"meta": {**meta, "peak_gpu_gib": 0.5, "gnn_device": "cuda"},
                            **{k: {"gap_mean": float((base - 2).mean()), "ms_mean": 3.0} for k in methods}}
                    _write(os.path.join(results, "phase2_scale", f"{m}_seed{s}", mode, f"tsp{n}_test"), summ, inst)
    for s in report.SEEDS:      # earlier dense evaluation of GCN, identical greedy
        folder = os.path.join(results, report.OLD_DENSE["gcn"](s))
        os.makedirs(folder)
        z = {}
        for n in (20, 50, 100):
            with np.load(os.path.join(results, "phase2_scale", f"gcn_seed{s}", "khalf100", f"tsp{n}_test.npz")) as f:
                z.update({f"tsp{n}_test__{k}": f[k] for k in ("greedy", "beam10", "greedy+2opt")})
        np.savez(os.path.join(folder, "eval_per_instance.npz"), **z)
    return results, data


def test_scale_report(tmp_path):
    results, data = fake_results(str(tmp_path))
    text = report.build_report(results, data, figs_dir=str(tmp_path / "figs"))
    assert os.path.exists(tmp_path / "figs" / "phase2_scale_gap.png")
    assert os.path.exists(tmp_path / "figs" / "phase2_scale_time.png")
    # 2 k-modes × 4 models × 6 comparisons per size, sizes 20 … 1000 (≥ 20 instances)
    assert text.count("0.05/48 = ") == 6
    assert text.count("**better** -2.00") == 6 * 48 and "**worse**" not in text
    assert "| GCN | 20 | 75/75 | 75/75 | " in text
    assert "| 10000 | 5 | 10000 | 1 | 100.0 / 100.0 |" in text
    assert "-2.00 (15/15 better)" in text
    assert "| GCN | k = 50 | greedy | 10000 | — | 1000 | none |" in text
    assert "## Missing results\n\nNone." in text and "## 2-opt timeouts\n\nNone." in text


def test_scale_report_missing(tmp_path):
    results, data = fake_results(str(tmp_path), skip={("gat", "k50", 5000, 1)})
    (tmp_path / "results" / "phase2_scale_notes.md").write_text("1. A hand-written note.\n")
    text = report.build_report(results, data, figs_dir=str(tmp_path / "figs"))
    assert "- `gat_seed1/k50/tsp5000_test`" in text and "1. A hand-written note." in text
    assert "| GATv2 greedy | " in text and "missing" in text
