"""Quick tests for scripts/phase2_beam_report.py on a fake results/ folder.  Run with: pytest tests/"""

import importlib.util
import json
import os

import numpy as np

# scripts/ is not a package: we load the file directly.
_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
_spec = importlib.util.spec_from_file_location("phase2_beam_report", os.path.join(_dir, "phase2_beam_report.py"))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

NUM = 200
METHODS = [k for dec in report.DECODERS for k in (dec, dec + "+2opt")]


def fake_results(root, skip=()):
    """Every beam method is 1 pp (beam 10) or 2 pp (beam 100) better than greedy on every instance."""
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(root, "phase2_exp1"))
    base_json, base_pi = {}, {}
    for d in report.DATASETS:
        base_json[d] = {m: {"gap_mean": 10.0} for m in ("greedy", "greedy+2opt", "christofides", "christofides+2opt")}
        base_pi.update({f"{d}__{b}": 10 + rng.random(NUM) for b in ("greedy+2opt", "christofides+2opt")})
    with open(os.path.join(root, "baselines.json"), "w") as f:
        json.dump({k: v for k, v in base_json.items() if k != "tsp100_test"}, f)
    with open(os.path.join(root, "phase2_exp1", "baselines_tsp100.json"), "w") as f:
        json.dump({"tsp100_test": base_json["tsp100_test"]}, f)
    np.savez(os.path.join(root, "baselines_per_instance.npz"),
             **{k: v for k, v in base_pi.items() if not k.startswith("tsp100")})
    np.savez(os.path.join(root, "phase2_exp1", "baselines_tsp100_per_instance.npz"),
             **{k: v for k, v in base_pi.items() if k.startswith("tsp100")})
    for c, (_, models, run, _) in report.CONFIGS.items():
        for m in models:
            for s in report.SEEDS:
                if (c, m, s) in skip:
                    continue
                folder = os.path.join(root, report.BEAM, c, run(m, s))
                os.makedirs(folder)
                ev, pi = {"meta": {}}, {}
                for d in report.DATASETS:
                    greedy = 12 + rng.random(NUM)
                    gaps = {"greedy": greedy, "beam10": greedy - 1, "beam100": greedy - 2}
                    gaps.update({k + "+2opt": v - 8 for k, v in list(gaps.items())})  # 2 - 4 %: below the baselines
                    ev[d] = {k: {"gap_mean": float(v.mean()), "decode_ms_per_inst": 1.0} for k, v in gaps.items()}
                    pi.update({f"{d}__{k}": v for k, v in gaps.items()})
                with open(os.path.join(folder, "eval.json"), "w") as f:
                    json.dump(ev, f)
                np.savez(os.path.join(folder, "eval_per_instance.npz"), **pi)


def test_beam_report(tmp_path):
    fake_results(str(tmp_path))
    text = report.build_report(str(tmp_path))
    n_tests = 6 * 3 * 3   # 6 configuration/model pairs, 3 datasets, 3 references
    assert f"{report.ALPHA}/{n_tests} tests" in text
    assert text.count("| **better** |") == n_tests and "| **worse** |" not in text
    assert "| 0.0% | 0.0% | 0.0% |" in text                  # beam never worse in the fake data
    assert "no earlier evaluation" in text                   # the consistency check has nothing to compare
    assert "## Missing results\n\nNone." in text


def test_beam_report_missing_run(tmp_path):
    fake_results(str(tmp_path), skip={("ln_last", "gcn", 1)})
    text = report.build_report(str(tmp_path))
    assert "ln_last/gcn_layernorm_ckpt_seed1" in text
    assert "| GCN | ln_last | missing |" in text


def test_zero_median_is_neutral():
    rng = np.random.default_rng(0)
    ref = [rng.random(NUM) for _ in range(3)]
    # Seeds 0 and 2: 60% ties (median 0, up to rounding), the rest better. Seed 1: all better.
    tie_then_better = [np.where(np.arange(NUM) < 0.6 * NUM, r - 1e-14, r - 1) for r in ref]
    a = [tie_then_better[0], ref[1] - 1, tie_then_better[2]]
    assert report.paired(a, ref, 0.05)["verdict"] == "better"
    # Neutral seeds do not count: if the only non-zero median says "worse", the verdict is "worse".
    a[1] = ref[1] + 1
    assert report.paired(a, ref, 0.05)["verdict"] == "worse"
    # Two non-zero medians with opposite signs: no verdict.
    assert report.paired([ref[0] - 1, ref[1] + 1, tie_then_better[2]], ref, 0.05)["verdict"] == "n.s."
    # Every median zero: the direction comes from the mean difference.
    assert report.paired(tie_then_better, ref, 0.05)["verdict"] == "better"
