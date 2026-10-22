"""Quick tests for scripts/phase3_window_report.py on a fake results/ folder.  Run with: pytest tests/"""

import importlib.util
import json
import os

import numpy as np

_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
_spec = importlib.util.spec_from_file_location("phase3_window_report", os.path.join(_dir, "phase3_window_report.py"))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

GAIN = {"greedy": 0.25, "beam10": 0.5, "lkh": 1.0, "2opt": 0.0}


def _write(path_no_ext, summary, per_inst):
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    np.savez(path_no_ext + ".npz", **per_inst)
    with open(path_no_ext + ".json", "w") as f:
        json.dump(summary, f)


def fake_results(root, skip=()):
    rng = np.random.default_rng(0)
    for n in report.SIZES:
        num = 25 if n <= 1000 else 5
        base = 4.5 + rng.random(num)
        _write(os.path.join(root, "phase2_scale", "baselines", f"tsp{n}_test"), {"meta": {"num": num}},
               {"greedy+2opt@khalf100": base})
        for s in report.SEEDS:
            _write(os.path.join(root, "phase2_scale", f"tagcn_k3_seed{s}", "khalf100", f"tsp{n}_test"), {"meta": {}},
                   {"greedy+2opt": base + 1})
        for folder in ["lkh", "2opt"] + [f"tagcn_k3_seed{s}" for s in report.SEEDS]:
            if (folder, n) in skip:
                continue
            methods = ["greedy", "beam10"] if folder.startswith("tagcn") else [folder]
            inst = {"initial__gap": base, "initial__ms": np.full(num, 5.0)}
            for m in methods:
                for w in report.WIDTHS:
                    g = GAIN[m] * (1 + rng.random(num) * 0.01) if GAIN[m] else np.zeros(num)
                    inst.update({f"{m}_w{w}__gap": base - g, f"{m}_w{w}__gain": g, f"{m}_w{w}__ms": np.full(num, 10.0),
                                 f"{m}_w{w}__ms_model": np.full(num, 2.0), f"{m}_w{w}__windows": np.full(num, 20.0),
                                 f"{m}_w{w}__improved": np.full(num, 5.0), f"{m}_w{w}__passes": np.full(num, 2.0)})
            _write(os.path.join(root, "phase3_window", folder, f"tsp{n}_test"),
                   {"meta": {"git_commit": "abc1234", "num": num}}, inst)


def test_window_report(tmp_path):
    fake_results(str(tmp_path))
    text = report.build_report(str(tmp_path), figs_dir=str(tmp_path / "figs"))
    assert os.path.exists(tmp_path / "figs" / "phase3_window_gap.png")
    assert text.count("0.05/12 = ") == 3
    assert text.count("**better**") == 3 * 9 and "**worse**" not in text     # 2opt: all ties -> n.s.
    assert "| 200 | 25 | 125/125 identical |" in text
    assert "| TAGCN K=3, beam B=10 paths | 20 | 50 % |" in text
    assert "25.0 % (2.0 passes)" in text and "(0/5 better)" in text and "(15/15 better)" in text
    assert "## Missing results\n\nNone." in text


def test_window_report_missing(tmp_path):
    fake_results(str(tmp_path), skip={("lkh", 10000)})
    (tmp_path / "phase3_window_notes.md").write_text("1. A note.\n")
    text = report.build_report(str(tmp_path), figs_dir=str(tmp_path / "figs"))
    assert "- `phase3_window/lkh/tsp10000_test`" in text and "1. A note." in text
