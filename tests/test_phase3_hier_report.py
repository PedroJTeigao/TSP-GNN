"""Quick tests for scripts/phase3_hier_report.py on a fake results/ folder.  Run with: pytest tests/"""

import importlib.util
import json
import os

import numpy as np

_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
_spec = importlib.util.spec_from_file_location("phase3_hier_report", os.path.join(_dir, "phase3_hier_report.py"))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def _write(path_no_ext, summary, per_inst):
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    np.savez(path_no_ext + ".npz", **per_inst)
    with open(path_no_ext + ".json", "w") as f:
        json.dump(summary, f)


def fake_results(root, skip=()):
    """Greedy edge + 2-opt at ~5 %, whole instance at ~6 %, every hierarchy variant 1 pp better than greedy edge."""
    rng = np.random.default_rng(0)
    for n in report.SIZES:
        num = 25 if n <= 1000 else 5
        base = 5 + rng.random(num)
        _write(os.path.join(root, "phase2_scale", "baselines", f"tsp{n}_test"),
               {"meta": {"num": num}, "greedy": {"gap_mean": 15.0}, "greedy+2opt@khalf100": {"gap_mean": float(base.mean())}},
               {"greedy": base + 10, "greedy__ms": np.full(num, 4.0), "greedy+2opt@khalf100": base,
                "greedy+2opt@khalf100__ms": np.full(num, 6.0)})
        for s in report.SEEDS:
            _write(os.path.join(root, "phase2_scale", f"tagcn_k3_seed{s}", "khalf100", f"tsp{n}_test"), {"meta": {}},
                   {"greedy": base + 9, "greedy+2opt": base + 1, **{k: np.full(num, 1.0) for k in
                    ("graph__ms", "gnn__ms", "greedy__ms", "greedy+2opt__ms")}})
            if (n, s) in skip:
                continue
            inst = {f"{p}__ms_partition": np.full(num, 3.0) for p in report.PARTITIONS}
            summ = {"meta": {"git_commit": "abc1234", "num": num, "hierarchy": {
                p: [{"points_per_level": [n, n // 20], "group_sizes": [[20] * (n // 20)]}] * num
                for p in report.PARTITIONS}}}
            for v in report.VARIANTS:
                inst.update({f"{v}__before": base + 5, f"{v}__after": base - 1, f"{v}__boundary_edges": np.full(num, n / 20),
                             f"{v}__boundary_wrong": np.full(num, n / 40), f"{v}__interior_edges": np.full(num, n - n / 20),
                             f"{v}__interior_wrong": np.full(num, n / 10), f"{v}__moved": np.full(num, 2.0),
                             f"{v}__c_used": np.full(num, 1.0),
                             **{f"{v}__ms_{x}": np.full(num, 2.0) for x in ("top", "stitch", "paths", "model", "2opt")}})
                summ[v] = {"gap_before": float((base + 5).mean()), "gap_after": float((base - 1).mean()), "timeouts": 0}
            _write(os.path.join(root, "phase3_hier", f"tagcn_k3_seed{s}", f"tsp{n}_test"), summ, inst)


def test_hier_report(tmp_path):
    fake_results(str(tmp_path))
    text = report.build_report(str(tmp_path), figs_dir=str(tmp_path / "figs"))
    assert os.path.exists(tmp_path / "figs" / "phase3_hier_gap.png")
    assert text.count("0.05/32 = ") == 3                       # n = 200, 500, 1000
    assert text.count("**better** -1.00") == 3 * 16 and text.count("**better** -2.00") == 3 * 16
    assert "| 200 | kmeans | 200 → 10 | L1: 20 / 20.0 / 20 |" in text
    assert "-1.00 (15/15 better)" in text
    assert "+0.00; same tour 100 %; c1 expanded 0 %" in text
    assert "| 200 | kmeans | distance | greedy | 5.0 | 50.0 % | 10.5 % | 20.0 % | 1.0 |" in text
    assert "## Missing results\n\nNone." in text


def test_hier_report_missing(tmp_path):
    fake_results(str(tmp_path), skip={(5000, 2)})
    (tmp_path / "phase3_hier_notes.md").write_text("1. A note.\n")
    text = report.build_report(str(tmp_path), figs_dir=str(tmp_path / "figs"))
    assert "- `phase3_hier/tagcn_k3_seed2/tsp5000_test`" in text and "1. A note." in text
