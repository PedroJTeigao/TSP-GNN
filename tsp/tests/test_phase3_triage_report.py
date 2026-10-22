"""Quick tests for scripts/phase3_triage_report.py on a fake results/ folder.  Run with: pytest tests/"""

import importlib.util
import json
import os

import numpy as np

_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
_spec = importlib.util.spec_from_file_location("phase3_triage_report", os.path.join(_dir, "phase3_triage_report.py"))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

GAIN = {"P0": 0.5, "P1": 1.0, "P2": 0.6, "P3": 0.8}


def _write(path_no_ext, summary, arrays):
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    np.savez(path_no_ext + ".npz", **arrays)
    with open(path_no_ext + ".json", "w") as f:
        json.dump(summary, f)


def fake_log(rng, windows=400):
    """Disagreement predicts LKH's gain; logp predicts whether the model catches it."""
    dis = rng.random(windows)
    lkh_gain = np.where(rng.random(windows) < dis, rng.random(windows) * 0.1, 0.0)
    logp = -rng.random(windows)
    catch = rng.random(windows) < np.exp(logp)
    cur = np.ones(windows)
    return {"cur_len": cur, "lkh_len": cur - lkh_gain, "model_len": cur - np.where(catch, lkh_gain, 0.2 * lkh_gain),
            "disagreement": dis, "logp": logp, "margin": rng.random(windows), "start": np.arange(windows, dtype=float),
            "ms_net": np.ones(windows), "ms_beam": np.ones(windows), "ms_lkh": np.ones(windows)}


def fake_results(root, skip=()):
    rng = np.random.default_rng(0)
    th = {"dis": {str(q): 1 - q / 100 for q in report.P2_TOP}, "conf_signal": "logp", "conf": -0.5,
          "conf_ap": {"logp": 0.6, "margin": 0.3}}
    for n in report.SIZES:
        num = 25 if n <= 1000 else 3
        base = 4.5 + rng.random(num)
        for s in [*report.SEEDS, "lkh"]:
            folder = "lkh" if s == "lkh" else f"tagcn_k3_seed{s}"
            if s != "lkh":
                _write(os.path.join(root, "phase3_triage", "log", folder, f"tsp{n}_test"), {"meta": {"git_commit": "abc1234"}},
                       fake_log(rng))
            if (folder, n) in skip:
                continue
            arrays = {"initial__gap": base, "initial__ms": np.full(num, 1000.0)}
            for p in (["P1"] if s == "lkh" else report.POLICIES):
                g = GAIN[p[:2]] * (1 + rng.random(num) * 0.01)
                arrays.update({f"{p}__gap": base - g, f"{p}__gain": g, f"{p}__ms": np.full(num, 2000.0),
                               f"{p}__windows": np.full(num, 100.0), f"{p}__to_lkh": np.full(num, 20.0),
                               f"{p}__to_model": np.full(num, 50.0 if p[:2] in ("P0", "P3") else 0.0)})
            _write(os.path.join(root, "phase3_triage", "policies", folder, f"tsp{n}_test"),
                   {"meta": {"git_commit": "abc1234", "thresholds": None if s == "lkh" else th}}, arrays)


def test_triage_report(tmp_path):
    fake_results(str(tmp_path))
    text = report.build_report(str(tmp_path), figs_dir=str(tmp_path / "figs"))
    for f in ("phase3_triage_pr.png", "phase3_triage_pareto.png"):
        assert os.path.exists(tmp_path / "figs" / f)
    assert "## Missing results\n\nNone." in text
    assert "| P2: LKH on top 10 % disagreement |" in text and "0.05/15 = " in text
    p0_vs_p1 = [line for line in text.splitlines() if line.startswith("| P0: model everywhere |") and "**worse**" in line]
    assert p0_vs_p1                                                     # P0 gains 0.5 pp, P1 1.0 pp
    assert "(0/9 shorter)" in text                                      # n = 10000: P0 never beats P1


def test_triage_report_missing(tmp_path):
    fake_results(str(tmp_path), skip={("lkh", 10000)})
    (tmp_path / "phase3_triage_notes.md").write_text("1. A note.\n")
    text = report.build_report(str(tmp_path), figs_dir=str(tmp_path / "figs"))
    assert "- `phase3_triage/policies/lkh/tsp10000_test`" in text and "1. A note." in text
