"""Quick tests for scripts/phase3_benchmark_report.py on a fake results/ folder.  Run with: pytest tests/"""

import importlib.util
import os

import numpy as np

_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
_spec = importlib.util.spec_from_file_location("phase3_benchmark_report",
                                               os.path.join(_dir, "phase3_benchmark_report.py"))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def fake_results(root, inst=8):
    rng = np.random.default_rng(0)
    bench = os.path.join(root, "phase3_benchmark")
    os.makedirs(os.path.join(bench, "e1"), exist_ok=True)
    ref = {}
    for name in report.REF_SETS:
        ref[f"{name}/lengths"] = np.full(inst, 10.0)
        ref[f"{name}/old_lengths"] = np.full(inst, 10.1)
        ref[f"{name}/improvement_pct"] = np.full(inst, 0.99)
        ref[f"{name}/wall_seconds"] = np.ones(inst)
        ref[f"{name}/has_reference"] = np.array(1)
        ref[f"{name}/params"] = np.array("RUNS=10")
    np.savez(os.path.join(bench, "ref_lkh3.npz"), **ref)
    name = report.FU[1000]
    budgets = report.BUDGETS[1000]
    configs = [(v, o, b) for v in ("quadrant5", "nn8") for o in (False, True) for b in budgets]
    c = len(configs)
    total = np.tile([b for _, _, b in configs], (inst, 1))
    total[:, 8] *= 1.5                                      # nn8 / LKH start / first budget: over budget
    np.savez(os.path.join(bench, "e1", f"{name}.npz"), length=np.full((inst, c), 10.01),
             first_trial_length=np.full((inst, c), 10.02), total_seconds=total, trials=np.full((inst, c), 3.0),
             config_variant=np.array([v for v, _, _ in configs]), config_ours=np.array([o for _, o, _ in configs]),
             config_budget=np.array([b for _, _, b in configs]),
             config_fits=np.array([not (v == "nn8" and o) for v, o, _ in configs]))
    for folder, keys in (("random", [f"random_s{s}_top{x}" for s in report.RANDOM_SEEDS for x in report.E2_FRACS]),
                         ("simple", [f"simple_top{x}" for x in report.E2_FRACS]),
                         ("oracle", [f"oracle_top{x}" for x in report.E2_FRACS]),
                         *[(f"disagreement_seed{s}", [f"disagreement_top{x}" for x in report.E2_FRACS])
                           for s in report.SEEDS]):
        better = {"random": 0.3, "simple": 0.4, "oracle": 0.9}.get(folder, 0.6)
        arrays = {"initial__gap": np.full(30, 5.0), "initial__ms": np.full(30, 100.0)}
        for k in keys:
            arrays.update({f"{k}__gap": 5.0 - better - 0.01 * rng.random(30), f"{k}__ms": np.full(30, 500.0),
                           f"{k}__to_lkh": np.full(30, 20.0), f"{k}__windows": np.full(30, 100.0)})
        os.makedirs(os.path.join(bench, "e2", folder), exist_ok=True)
        np.savez(os.path.join(bench, "e2", folder, f"{name}.npz"), **arrays)
    return name


def test_variant_label():
    assert report.variant_label("alpha_p100") == "ALPHA, INITIAL_PERIOD = 100"
    assert report.variant_label("quadrant5") == "QUADRANT 5"


def test_e1_rows_and_over_budget(tmp_path):
    name = fake_results(str(tmp_path))
    R = report.load_all(str(tmp_path))
    st = report.e1_get(R, name, "quadrant5", False, 0.72)
    assert np.isclose(st["gap"], 0.1) and np.isclose(st["gap1"], 0.2) and st["over"] == 0
    assert report.e1_get(R, name, "nn8", False, 0.72)["over"] == 1.0
    assert report.e1_get(R, name, "nn8", True, 0.72) is None              # does not fit
    over, main = report.over_budget(R)
    assert "NEAREST-NEIGHBOR 8 (LKH start; 0.72 s)" in over and main == "stays within its budgets"


def test_e2_policies_and_tests(tmp_path):
    name = fake_results(str(tmp_path))
    R = report.load_all(str(tmp_path))
    rnd = report.e2_policy(R, name, "random", 20)
    assert rnd["runs"] == 5 and np.isclose(rnd["s"], 0.6) and np.isclose(rnd["lkh"], 0.2)
    assert report.e2_policy(R, name, "disagreement", 50)["runs"] == 3
    tests = report.wilcoxon_e2(R, name)
    assert set(tests) == {(20, "random"), (20, "simple"), (50, "random"), (50, "simple")}
    assert all(t["verdict"] == "better" and t["mean_diff"] < 0 for t in tests.values())


def test_build_report_with_missing_results(tmp_path):
    fake_results(str(tmp_path))
    text = report.build_report(str(tmp_path), os.path.join(str(tmp_path), "figs"))
    assert "## Comparability caveats" in text and "missing" in text and "not reproduced" in text
    assert os.path.exists(os.path.join(str(tmp_path), "figs", "phase3_benchmark_pareto.png"))
