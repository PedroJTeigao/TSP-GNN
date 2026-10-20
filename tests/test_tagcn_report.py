"""Quick tests for scripts/tagcn_report.py on a fake results/ folder.  Run with: pytest tests/"""

import importlib.util
import json
import os

import numpy as np

# scripts/ is not a package: we load the file directly.
_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
_spec = importlib.util.spec_from_file_location("tagcn_report", os.path.join(_dir, "tagcn_report.py"))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

NUM = 200


def fake_results(root, skip=()):
    """Baselines at ~10 %; every model 2 pp better on every instance (TAGCN: 4 pp), every method."""
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(root, "phase2_exp1"))
    base_json, base_pi = {}, {}
    for d in report.DATASETS:
        base_json[d] = {m: {"gap_mean": 10.0} for m in ("greedy", "greedy+2opt", "christofides", "christofides+2opt")}
        base_pi.update({f"{d}__{b}": 10 + rng.random(NUM)
                        for b in ("greedy", "greedy+2opt", "christofides", "christofides+2opt")})
    with open(os.path.join(root, "baselines.json"), "w") as f:
        json.dump({k: v for k, v in base_json.items() if k != "tsp100_test"}, f)
    with open(os.path.join(root, "phase2_exp1", "baselines_tsp100.json"), "w") as f:
        json.dump({"tsp100_test": base_json["tsp100_test"]}, f)
    np.savez(os.path.join(root, "baselines_per_instance.npz"),
             **{k: v for k, v in base_pi.items() if not k.startswith("tsp100")})
    np.savez(os.path.join(root, "phase2_exp1", "baselines_tsp100_per_instance.npz"),
             **{k: v for k, v in base_pi.items() if k.startswith("tsp100")})
    for blk, (_, datasets, methods, folder_of) in report.BLOCKS.items():
        for a in report.ARCHS:
            for s in report.SEEDS:
                if (blk, a, s) in skip:
                    continue
                folder = os.path.join(root, folder_of(a, s))
                os.makedirs(folder, exist_ok=True)
                ev, pi = {"meta": {"best_epoch": 20, "best_val_gap": 6.0, "config": {}}}, {}
                for d in datasets:
                    shift = 4 if a in report.TAGCN else 2
                    for m in methods:
                        g = base_pi[f"{d}__greedy"] - shift   # same instances in every block: consistency holds
                        ev.setdefault(d, {})[m] = {"gap_mean": float(g.mean())}
                        pi[f"{d}__{m}"] = g
                with open(os.path.join(folder, "eval.json"), "w") as f:
                    json.dump(ev, f)
                np.savez(os.path.join(folder, "eval_per_instance.npz"), **pi)
    for a in report.TAGCN:                                   # BatchNorm reruns: best.pt (k = 10) and last.pt
        for s in report.SEEDS:
            if ("rerun", a, s) in skip:
                continue
            run = os.path.join(root, report.CKPT_RUN(a, s))
            for folder, ds, methods, meta in ((run, ("tsp20_test", "tsp50_test"), ("greedy", "greedy+2opt"),
                                               {"best_epoch": 5, "best_val_gap": 6.0}),
                                              (os.path.join(run, "last_beam"), report.DATASETS, report.LAST_METHODS,
                                               {"epoch": 20})):
                os.makedirs(folder, exist_ok=True)
                ev = {"meta": meta, **{d: {m: {"gap_mean": float((base_pi[f"{d}__greedy"] - 4).mean())}
                                           for m in methods} for d in ds}}
                with open(os.path.join(folder, "eval.json"), "w") as f:
                    json.dump(ev, f)
                np.savez(os.path.join(folder, "eval_per_instance.npz"),
                         **{f"{d}__{m}": base_pi[f"{d}__greedy"] - 4 for d in ds for m in methods})
            with open(os.path.join(run, "history.json"), "w") as f:
                json.dump([{"epoch": e, "val_gap_greedy": 7.0, "val_loss": 0.6 - 0.01 * e + 1e-3,
                            "seconds": 3.0, "peak_gpu_mb": 100.0} for e in range(1, 21)], f)
    for ln in (False, True):
        for a in report.ARCHS:
            for s in report.SEEDS:
                path = report.history_path(root, a, s, ln)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    json.dump([{"epoch": e, "val_gap_greedy": 10 - 0.1 * e, "val_loss": 0.6 - 0.01 * e,
                                "seconds": 3.0, "peak_gpu_mb": 100.0} for e in range(1, 21)], f)


def test_tagcn_report(tmp_path):
    fake_results(str(tmp_path))
    text = report.build_report(str(tmp_path), figs_dir=str(tmp_path / "figs"))
    assert os.path.exists(tmp_path / "figs" / "tagcn_curves.png")
    assert "0.05/16 tests" in text and "0.05/12 tests" in text   # 2 K × 2 datasets × 2 decoders × 2 baselines; beam 2 × 3 × 2
    assert text.count("| **better** |") == 16 + 12 and "| **worse** |" not in text
    assert "| TAGCN K=2 | 3 | 2 | 6 | 54,337 |" in text and "| TAGCN K=3 | 2 | 3 | 6 | 50,049 |" in text
    assert "| 600 | 0 | 0.00e+00 |" in text                     # consistency check: 3 seeds × 200, all identical
    assert "## Missing results\n\nNone." in text
    assert "(to be written after reading the results)" in text
    assert "| TAGCN K=3 (last.pt, rerun) |" in text and "| TAGCN K=2 last.pt |" in text
    assert "| tagcn_k2_seed0 | 20 / 5 | " in text and "| 1.0e-03 |" in text and "| 400/400 |" in text


def test_tagcn_report_missing_and_notes(tmp_path):
    fake_results(str(tmp_path), skip={("ln", "tagcn_k3", 1), ("rerun", "tagcn_k2", 0)})
    (tmp_path / "tagcn_notes.md").write_text("1. A hand-written note.\n")
    text = report.build_report(str(tmp_path))
    assert "runs/tagcn_k3_layernorm_seed1/last_kfrac05" in text
    assert "| TAGCN K=3 | missing | missing | missing | missing | missing | missing |" in text
    assert "1. A hand-written note." in text
    assert "runs/tagcn_k2_ckpt_seed0/last_beam" in text and "| tagcn_k2_seed0 | missing |" in text
