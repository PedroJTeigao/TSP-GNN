"""Quick tests for scripts/phase2_checks_report.py on a fake results/ folder.  Run with: pytest tests/"""

import importlib.util
import json
import os

import numpy as np

# scripts/ is not a package: we load the files directly.
_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
_spec = importlib.util.spec_from_file_location("phase2_checks_report", os.path.join(_dir, "phase2_checks_report.py"))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)
_spec = importlib.util.spec_from_file_location("test_b2", os.path.join(os.path.dirname(__file__),
                                                                      "test_phase2_batch2_report.py"))
b2test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b2test)

NUM = b2test.NUM


def fake_results(root):
    """Batch 2 fake results + the check folders: every new configuration is 2 pp better than phase 1."""
    b2test.fake_results(root)
    rng = np.random.default_rng(1)
    with open(os.path.join(root, "baselines.json")) as f:
        datasets = [d for d in json.load(f) if d != "meta"]
    np.savez(os.path.join(root, "baselines_per_instance.npz"),
             **{f"{d}__{b}": 10 + rng.random(NUM) for d in datasets for b in ("greedy+2opt", "christofides+2opt")})
    np.savez(os.path.join(root, "phase2_exp1", "baselines_tsp100_per_instance.npz"),
             **{f"tsp100_test__{b}": 10 + rng.random(NUM) for b in ("greedy+2opt", "christofides+2opt")})
    for m in report.MODELS:
        for s in report.SEEDS:
            phase1 = {}
            for folder in report.CONFIGS["phase1"][1](m, s):
                with np.load(os.path.join(root, folder, "eval_per_instance.npz")) as f:
                    phase1.update({k.split("__")[0]: f[k] for k in f.files if k.endswith("__greedy")})
            for c in report.NEW + ["train50"]:
                folder = report.CONFIGS[c][1](m, s)[-1]
                if not os.path.exists(os.path.join(root, folder, "eval.json")):
                    b2test.write_eval(os.path.join(root, folder), {d: g - 2 for d, g in phase1.items()})
            # Validation history: gap best at epoch 2, loss best at the last epoch.
            hist = [{"epoch": e, "train_loss": 1 / e, "val_loss": 1 / e, "val_gap_greedy": abs(e - 2) + 5.0}
                    for e in range(1, 21)]
            with open(os.path.join(root, "runs", report.ln_run(m, s), "history.json"), "w") as f:
                json.dump(hist, f)


def test_checks_report(tmp_path):
    fake_results(str(tmp_path))
    text = report.build_report(str(tmp_path))
    assert "None: every configuration has all 3 seeds." in text
    t1 = text.split("## Table 1")[1].split("## Table 2")[0]
    assert "| missing" not in t1 and "| — |" not in t1                    # TSP-20 for every configuration
    assert "best val gap at epochs [2, 2, 2, 2, 2, 2, 2, 2, 2] (9 at epoch ≤ 5)" in text
    assert "best val loss at epochs [20, 20, 20, 20, 20, 20, 20, 20, 20]" in text
    t5 = text.split("## Table 5")[1].split("## Decisions")[0]
    # 15 configurations × 2 baselines × 3 models × 2 datasets
    assert f"0.05/{len(report.CONFIGS) * 2 * 3 * 2} tests" in t5
    histories = report.load_histories(str(tmp_path))
    report.plot_curves(histories, str(tmp_path / "figs"))
    assert (tmp_path / "figs" / "phase2_checks_ln_curves.png").exists()
