"""Quick tests for scripts/phase2_datasize_report.py on a fake results/ folder.  Run with: pytest tests/"""

import importlib.util
import json
import os

import numpy as np

# scripts/ is not a package: we load the file directly.
_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "phase2_datasize_report.py")
_spec = importlib.util.spec_from_file_location("phase2_datasize_report", _path)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

NUM = 100
SHIFT = {12000: 0.0, 25000: -0.5, 50000: -1.0, 100000: -1.5}  # more data -> lower gap


def write_eval(folder, gaps, epoch=20):
    os.makedirs(folder, exist_ok=True)
    ev = {"meta": {"best_epoch": epoch}}
    for d, g in gaps.items():
        ev[d] = {m: {"gap_mean": float(g.mean())} for m in report.METHODS}
    with open(os.path.join(folder, "eval.json"), "w") as f:
        json.dump(ev, f)
    np.savez_compressed(os.path.join(folder, "eval_per_instance.npz"),
                        **{f"{d}__{m}": g for d, g in gaps.items() for m in report.METHODS})


def fake_results(root, skip_sizes=()):
    rng = np.random.default_rng(0)
    entry = {m: {"gap_mean": 10.0} for k in report.summarize.BASELINES for m in (k, k + "+2opt")}
    os.makedirs(os.path.join(root, "phase2_exp1"))
    with open(os.path.join(root, "baselines.json"), "w") as f:
        json.dump({"meta": {}, "tsp20_test": entry, "tsp50_test": entry}, f)
    with open(os.path.join(root, "phase2_exp1", "baselines_tsp100.json"), "w") as f:
        json.dump({"meta": {}, "tsp100_test": entry}, f)
    base = {d: 5 + rng.random(NUM) for d in report.DATASETS}
    for m in report.MODELS:
        for n, seeds in report.SEEDS.items():
            if n in skip_sizes:
                continue
            for s in seeds:
                gaps = {d: g + SHIFT[n] + 0.01 * s for d, g in base.items()}
                k10, kfrac = report.folders(m, n, s, "k10"), report.folders(m, n, s, "kfrac")
                if n == 12000:  # phase 1 layout: TSP-20/50 in runs/, TSP-100 elsewhere, k = 0.5 n without TSP-20
                    write_eval(os.path.join(root, k10[0]), {d: gaps[d] for d in ("tsp20_test", "tsp50_test")})
                    write_eval(os.path.join(root, k10[1]), {"tsp100_test": gaps["tsp100_test"]})
                    write_eval(os.path.join(root, kfrac[1]), {d: gaps[d] - 2 for d in ("tsp50_test", "tsp100_test")})
                else:
                    write_eval(os.path.join(root, k10[0]), gaps)
                    write_eval(os.path.join(root, kfrac[0]), {d: gaps[d] - 2 * (d != "tsp20_test") for d in gaps})


def test_report_on_fake_results(tmp_path):
    fake_results(str(tmp_path))
    text = report.build_report(str(tmp_path))
    # 2 test graphs × 3 models × 3 datasets × 2 methods = 36 paired tests, all "better" (100k is 1.5 pp lower)
    assert "0.05/36 tests" in text
    table3 = text.split("## Table 3")[1].split("## Selected epochs")[0]
    assert table3.count("**better**") == 36 and "**worse**" not in table3
    assert "None." in text.split("## Missing results")[1]
    # seed 0 series of the greedy gap on TSP-20 decreases by 0.5 pp per size step
    t1 = text.split("## Table 1")[1].split("### Test graphs k = 0.5 n")[0]
    gcn = [line for line in t1.splitlines() if line.startswith("| GCN |")]
    assert len(gcn) == 4
    first = [float(line.split("|")[3]) for line in gcn]
    assert np.allclose(np.diff(first), -0.5, atol=0.011)
    # k = 0.5 n on TSP-20 reuses the k = 10 evaluation for phase 1
    data = report.load_all(str(tmp_path))
    assert data["kfrac", "gcn", 12000, 0]["eval"]["tsp20_test"] == data["k10", "gcn", 12000, 0]["eval"]["tsp20_test"]
    report.plot(data, report.load_baselines(str(tmp_path)), str(tmp_path / "figs"))
    assert (tmp_path / "figs" / "phase2_datasize_gap.png").exists()


def test_report_with_missing_runs(tmp_path):
    fake_results(str(tmp_path), skip_sizes=(50000, 100000))
    text = report.build_report(str(tmp_path))
    assert "no paired tests" in text and "| GCN | 100k | missing |" in text
    assert "gcn_data50k_seed0 / k = 10 / TSP-20" in text.split("## Missing results")[1]
    report.plot(report.load_all(str(tmp_path)), {}, str(tmp_path / "figs"))  # must not fail with gaps missing
