"""Quick tests for scripts/phase3_wsize_report.py on a fake results/ folder.  Run with: pytest tests/"""

import importlib.util
import json
import os

import numpy as np

_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
_spec = importlib.util.spec_from_file_location("phase3_wsize_report", os.path.join(_dir, "phase3_wsize_report.py"))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def _write(path_no_ext, summary, arrays):
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    np.savez(path_no_ext + ".npz", **arrays)
    with open(path_no_ext + ".json", "w") as f:
        json.dump(summary, f)


def fake_results(root, skip=()):
    rng = np.random.default_rng(0)
    for n, name in report.SETS.items():
        num = 20 if n == 1000 else 3
        base = 4.5 + rng.random(num)
        jobs = {"lkh": ("lkh", report.WIDTHS, lambda w: 0.01 * w)}
        for s in report.SEEDS:
            jobs[f"tagcn_k3_seed{s}"] = ("beam10", report.WIDTHS, lambda w: 0.5 if w == 20 else 0.1)
            jobs[f"tagcn_k3_tsp50_seed{s}"] = ("beam10", (50,), lambda w: 0.3)
            jobs[f"tagcn_k3_tsp100_seed{s}"] = ("beam10", (100, 200, 300), lambda w: 0.4)
        for folder, (method, widths, gain) in jobs.items():
            if (folder, n) in skip:
                continue
            arrays = {"initial__gap": base, "initial__ms": np.full(num, 100.0)}
            for w in widths:
                g = gain(w) * (1 + 0.01 * rng.random(num))
                arrays.update({f"{method}_w{w}__gap": base - g, f"{method}_w{w}__gain": g,
                               f"{method}_w{w}__ms": np.full(num, 10.0 * w), f"{method}_w{w}__ms_model": np.full(num, 1.0),
                               f"{method}_w{w}__windows": np.full(num, 10.0), f"{method}_w{w}__improved": np.full(num, 2.0),
                               f"{method}_w{w}__passes": np.full(num, 2.0), f"{method}_w{w}__limited": np.zeros(num)})
            limited = [300] if folder == "lkh" else []
            _write(os.path.join(root, "phase3_wsize", folder, name),
                   {"meta": {"git_commit": "abc1234", "limited_widths": limited}}, arrays)
    for t in ("tsp20", "tsp50", "tsp100"):
        for s in report.SEEDS:
            run = os.path.join(root, "runs", report.model_folder(t, s))
            os.makedirs(os.path.join(run, "kfrac05"), exist_ok=True)
            e = {"meta": {"best_epoch": 3, "best_val_gap": 5.0},
                 **{d: {"greedy": {"gap_mean": 10.0}, "greedy+2opt": {"gap_mean": 3.0}} for d in ("tsp50_test", "tsp100_test")}}
            for p in (os.path.join(run, "eval.json"), os.path.join(run, "kfrac05", "eval.json")):
                with open(p, "w") as f:
                    json.dump(e, f)


def test_matched():
    assert [report.matched(w) for w in report.WIDTHS] == ["tsp20", "tsp50", "tsp100", "tsp100", "tsp100"]


def test_wsize_report(tmp_path):
    fake_results(str(tmp_path))
    text = report.build_report(str(tmp_path), figs_dir=str(tmp_path / "figs"))
    for f in ("phase3_wsize_pareto.png", "phase3_wsize_cost.png"):
        assert os.path.exists(tmp_path / "figs" / f)
    assert "## Missing results\n\nNone." in text and "n = 1000, w = 300" in text
    assert "0.05/27 = " in text                     # 5 w × 3 + 4 w × 3 matched comparisons
    assert "| 1000 | 20 | TSP-20 model vs. greedy edge + 2-opt | **better** |" in text
    assert "(9/9 shorter)" in text
    assert "| TSP-100 | 2 | 3 | 5.00 | 10.00 / 3.00 | 10.00 / 3.00 |" in text


def test_wsize_report_missing(tmp_path):
    fake_results(str(tmp_path), skip={("tagcn_k3_tsp100_seed1", 10000)})
    (tmp_path / "phase3_wsize_notes.md").write_text("1. A note.\n")
    text = report.build_report(str(tmp_path), figs_dir=str(tmp_path / "figs"))
    assert "- `phase3_wsize/tagcn_k3_tsp100_seed1/tsp10000_test`" in text and "1. A note." in text
