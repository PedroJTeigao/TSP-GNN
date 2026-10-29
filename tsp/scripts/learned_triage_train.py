"""Learned triage, step 2: train the window signal ("will LKH improve this window?") and measure its AP.

For one TAGCN K=3 seed (the features with p come from that seed's network):
    train = half of the instances of tsp200 and of Fu-1k (learned_triage.split_instances, seed 0), validation = the
    other half; test = tsp1000, tsp10000 and Fu-10k (never used to fit or to choose anything).
    Each training set gets the same total weight (Fu-1k has 6.4× more windows than tsp200).
Models (sklearn, CPU), main features learned_triage.FEATURES (scale-free, no n, no beam):
    logreg   standardisation + logistic regression, C chosen on validation
    hgb      histogram gradient boosting classifier (label: LKH improves), depth / rate chosen on validation
    hgb_reg  histogram gradient boosting regressor on the relative gain in % (ranked by predicted gain)
The model with the highest validation AP (mean of the two validation sets) is the "chosen" one, used by the policy.
Ablations of the chosen type: + log10_n, + logp (beam; AP only, the policy does not run the beam).
AP of the raw disagreement and of random ranking (= positive rate) on the same windows, for comparison.
Importance: logistic coefficients (standardised features) and permutation importance (AP drop) of the chosen
model on the validation windows.

Writes results/phase3_learned_triage/models/tagcn_k3_seed<s>/{chosen.joblib, metrics.json}.

Example:
    python scripts/learned_triage_train.py --seed 0
"""

import argparse
import datetime
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tsp_gnn import learned_triage as L
from tsp_gnn.utils import git_info

TRAIN_SETS = ("tsp200_test", "tsp1000_test_concorde")
TEST_SETS = ("tsp1000_test", "tsp10000_test", "tsp10000_test_concorde")
SPLIT_SEED = 0
GRIDS = {"logreg": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
         "hgb": [{"max_depth": d, "learning_rate": r} for d in (3, 6) for r in (0.05, 0.1)],
         "hgb_reg": [{"max_depth": d, "learning_rate": r} for d in (3, 6) for r in (0.05, 0.1)]}


def make(kind, params):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], max_iter=2000))
    common = dict(max_iter=200, min_samples_leaf=50, early_stopping=False, random_state=0, **params)
    return HistGradientBoostingClassifier(**common) if kind == "hgb" else HistGradientBoostingRegressor(**common)


def score(model, kind, X):
    """Higher = more likely that LKH improves the window."""
    return model.predict(X) if kind == "hgb_reg" else model.predict_proba(X)[:, 1]


def ap(y, s):
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def load(root, seed, name):
    return dict(np.load(os.path.join(root, f"tagcn_k3_seed{seed}", f"{name}.npz")))


def matrix(d, cols, mask=None):
    X = np.stack([d[c] for c in cols], axis=1).astype(float)
    return X if mask is None else X[mask]


def main():
    parser = argparse.ArgumentParser(description="Learned triage: train and evaluate the window signal.")
    parser.add_argument("--seed", type=int, required=True, help="TAGCN K=3 seed")
    parser.add_argument("--features-root", default="results/phase3_learned_triage/features")
    parser.add_argument("--out-root", default="results/phase3_learned_triage/models")
    args = parser.parse_args()
    data = {name: load(args.features_root, args.seed, name) for name in TRAIN_SETS + TEST_SETS}

    # instance split: the same instances on each side for every seed
    tr, va = {}, {}
    for name in TRAIN_SETS:
        inst = data[name]["instance"].astype(int)
        side = L.split_instances(int(inst.max()) + 1, seed=SPLIT_SEED)[inst]
        tr[name], va[name] = side, ~side
        assert set(inst[side]).isdisjoint(set(inst[~side]))

    def fit_eval(kind, params, cols):
        X = np.concatenate([matrix(data[nm], cols, tr[nm]) for nm in TRAIN_SETS])
        target = "gain_rel" if kind == "hgb_reg" else "improves"
        y = np.concatenate([data[nm][target][tr[nm]] for nm in TRAIN_SETS])
        w = np.concatenate([np.full(tr[nm].sum(), 1.0 / tr[nm].sum()) for nm in TRAIN_SETS])
        w *= len(w) / w.sum()                                  # equal total weight per training set
        model = make(kind, params)
        if kind == "logreg":
            model.fit(X, y.astype(int), logisticregression__sample_weight=w)
        else:
            model.fit(X, y if kind == "hgb_reg" else y.astype(int), sample_weight=w)
        val = {nm: ap(data[nm]["improves"][va[nm]], score(model, kind, matrix(data[nm], cols, va[nm])))
               for nm in TRAIN_SETS}
        return model, val

    results, best = {}, {}
    for kind, grid in GRIDS.items():
        trials = []
        for params in grid:
            model, val = fit_eval(kind, params, L.FEATURES)
            trials.append((float(np.mean(list(val.values()))), params, model, val))
            print(f"  {kind} {params}: validation AP {val}", flush=True)
        mean_ap, params, model, val = max(trials, key=lambda t: t[0])
        best[kind] = (model, params)
        results[kind] = {"params": params, "val_ap": val, "val_ap_mean": mean_ap,
                         "test_ap": {nm: ap(data[nm]["improves"], score(model, kind, matrix(data[nm], L.FEATURES)))
                                     for nm in TEST_SETS},
                         "grid": [{"params": p, "val_ap_mean": m} for m, p, _, _ in trials]}
    chosen = max(results, key=lambda k: results[k]["val_ap_mean"])
    model, params = best[chosen]

    ablations = {}
    for extra in L.EXTRA:
        cols = L.FEATURES + (extra,)
        m, val = fit_eval(chosen, params, cols)
        ablations[extra] = {"val_ap": val, "test_ap": {nm: ap(data[nm]["improves"], score(m, chosen,
                                                          matrix(data[nm], cols))) for nm in TEST_SETS}}

    # raw signals on the same windows (validation halves and full test sets)
    raw = {}
    for nm in TRAIN_SETS + TEST_SETS:
        mask = va[nm] if nm in TRAIN_SETS else np.ones(len(data[nm]["improves"]), dtype=bool)
        y = data[nm]["improves"][mask]
        raw[nm] = {"disagreement": ap(y, data[nm]["disagreement"][mask]),
                   "simple": ap(y, data[nm]["seg_len_nn"][mask]), "random": float(y.mean()),
                   "windows": int(mask.sum()), "instances": int(len(np.unique(data[nm]["instance"][mask])))}

    # importance
    coef = dict(zip(L.FEATURES, best["logreg"][0][-1].coef_[0].tolist()))
    Xv = np.concatenate([matrix(data[nm], L.FEATURES, va[nm]) for nm in TRAIN_SETS])
    yv = np.concatenate([data[nm]["improves"][va[nm]] for nm in TRAIN_SETS]).astype(int)
    wv = np.concatenate([np.full(va[nm].sum(), 1.0 / va[nm].sum()) for nm in TRAIN_SETS])
    scorer = (lambda est, X, y, sample_weight=None: average_precision_score(y, score(est, chosen, X),
                                                                            sample_weight=sample_weight))
    perm = permutation_importance(model, Xv, yv, scoring=scorer, n_repeats=5, random_state=0, sample_weight=wv)
    importance = {f: {"mean": float(m), "std": float(s)} for f, m, s in
                  zip(L.FEATURES, perm.importances_mean, perm.importances_std)}

    commit, dirty = git_info()
    out_dir = os.path.join(args.out_root, f"tagcn_k3_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"model": model, "kind": chosen, "features": L.FEATURES, "params": params}, os.path.join(out_dir, "chosen.joblib"))
    metrics = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                        "seed": args.seed, "train_sets": TRAIN_SETS, "test_sets": TEST_SETS,
                        "split_seed": SPLIT_SEED, "features": L.FEATURES,
                        "train_windows": {nm: int(tr[nm].sum()) for nm in TRAIN_SETS}},
               "chosen": chosen, "models": results, "ablations": ablations, "raw": raw,
               "logreg_coef": coef, "permutation_importance": importance}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in ("chosen", "raw")}, indent=2))
    print({k: (v["val_ap"], v["test_ap"]) for k, v in results.items()})
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out_dir}")


if __name__ == "__main__":
    main()
