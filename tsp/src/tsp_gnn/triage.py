"""Window triage (phase 3): for each window of the tour, do nothing, use the model, or call LKH.

Builds on window.py (windows of w consecutive tour cities, ends fixed, replaced only if shorter).

Signals of a window, all computed on the window rescaled to [0, 1]² (so they do not depend on n):
    disagreement   1 − mean model probability of the w − 1 edges of the current segment
                   (an edge outside the model's k-NN graph counts as p = 0)
    logp           mean log-probability per edge of the beam B=10 path that is returned (the shortest)
    margin         (highest − second highest) final beam score, per edge
Gains (in the instance's units): current length − beam path length (model, can be <= 0) and
current length − LKH path length (the best a window can give).

Policies (one action per window, then the same "replace only if shorter" rule):
    model      beam path everywhere                              (P0)
    lkh        LKH everywhere                                    (P1)
    dis        LKH if disagreement >= dis, else nothing          (P2; needs only the network)
    mixed      LKH if disagreement >= dis and confidence < conf,
               else beam path if confidence >= conf, else nothing (P3; confidence needs the beam
               on every window, so its decoding time is paid everywhere)
"""

import time

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from tsp_gnn.hier import beam_path
from tsp_gnn.window import lkh_path, path_length, window_starts

IMPROVE_TOL = 1e-9     # same rule as window.refine: a window improves if shorter by more than this
CATCH = 0.8            # "the model catches the window": model gain >= CATCH × LKH gain


def segment_disagreement(heat):
    """1 − mean probability of the edges (i, i+1) of the window in its current order."""
    m = len(heat)
    return float(1.0 - heat[np.arange(m - 1), np.arange(1, m)].mean())


def beam_with_confidence(heat, pts, beam_width=10):
    """Beam path between the window's ends plus (logp, margin), both per edge."""
    m = len(pts)
    order, info = beam_path(heat, pts, 0, m - 1, beam_width, info=True)
    scores = info["scores"]
    margin = (scores[0] - scores[1]) / (m - 1) if len(scores) > 1 else np.nan
    return order, info["logp"] / (m - 1), float(margin)


def window_log(coords, tour, w, heatmaps, beam_width=10):
    """Record every window of `tour` at offsets 0 and w/2 (all on the same tour, each window
    evaluated alone, nothing replaced). Returns a dict of per-window arrays."""
    t = np.asarray(tour)
    n = len(t)
    rows = {k: [] for k in ("offset", "start", "size", "cur_len", "disagreement", "logp", "margin",
                            "model_len", "lkh_len", "ms_net", "ms_beam", "ms_lkh")}
    for offset in (0, w // 2):
        wins = window_starts(n, w, offset)
        sets = [coords[t[(s + np.arange(length)) % n]] for s, length in wins]
        before = heatmaps.seconds
        heats = heatmaps(sets)
        ms_net = 1000 * (heatmaps.seconds - before) / len(sets)        # batched: shared equally
        for (s, length), pts, heat in zip(wins, sets, heats):
            start = time.perf_counter()
            order, logp, margin = beam_with_confidence(heat, pts, beam_width)
            ms_beam = 1000 * (time.perf_counter() - start)
            start = time.perf_counter()
            lkh_order = lkh_path(pts)
            ms_lkh = 1000 * (time.perf_counter() - start)
            for k, v in (("offset", offset), ("start", s), ("size", length), ("cur_len", path_length(pts)),
                         ("disagreement", segment_disagreement(heat)), ("logp", logp), ("margin", margin),
                         ("model_len", path_length(pts[order])), ("lkh_len", path_length(pts[lkh_order])),
                         ("ms_net", ms_net), ("ms_beam", ms_beam), ("ms_lkh", ms_lkh)):
                rows[k].append(v)
    return {k: np.array(v, dtype=float) for k, v in rows.items()}


# ---------------------------------------------------------------- targets and signal quality

def targets(log):
    """(i) LKH improves the window; (ii) the model catches >= 80 % of LKH's gain (defined only where (i))."""
    lkh_gain = log["cur_len"] - log["lkh_len"]
    model_gain = log["cur_len"] - log["model_len"]
    lkh_improves = lkh_gain > IMPROVE_TOL
    catches = model_gain >= CATCH * lkh_gain
    return lkh_improves, catches


def signals(log):
    """Signals oriented so that higher = more likely positive, for targets (i) and (ii)."""
    rel_gain = (log["cur_len"] - log["model_len"]) / log["cur_len"]
    return {"i": {"disagreement": log["disagreement"], "-logp": -log["logp"], "-margin": -log["margin"],
                  "model gain": rel_gain},
            "ii": {"logp": log["logp"], "margin": log["margin"], "-disagreement": -log["disagreement"],
                   "model gain": rel_gain}}


def average_precision(score, y):
    return float(average_precision_score(y, score)) if 0 < y.sum() < len(y) else np.nan


def pr_curve(score, y):
    precision, recall, _ = precision_recall_curve(y, score)
    return precision, recall


def best_f1_threshold(score, y):
    """Threshold on score (predict positive if score >= threshold) with the highest F1."""
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def choose_thresholds(log, quantiles=(10, 20, 30, 50)):
    """Thresholds from a validation log: disagreement of the top q % windows, and for P3 the confidence
    signal (logp or margin, the one with the higher average precision for (ii)) at its best F1."""
    lkh_improves, catches = targets(log)
    dis = {str(q): float(np.quantile(log["disagreement"], 1 - q / 100)) for q in quantiles}
    ap = {s: average_precision(log[s][lkh_improves], catches[lkh_improves]) for s in ("logp", "margin")}
    signal = max(ap, key=lambda s: -np.inf if np.isnan(ap[s]) else ap[s])
    conf = best_f1_threshold(log[signal][lkh_improves], catches[lkh_improves])
    return {"dis": dis, "conf_signal": signal, "conf": conf, "conf_ap": ap}


# ---------------------------------------------------------------- policies

def triage_refine(coords, tour, w, policy, heatmaps=None, max_passes=3, beam_width=10):
    """Window refinement where each window goes to nothing / the model / LKH according to `policy`
    ({"kind": "model" | "lkh" | "dis" | "mixed", "dis": threshold, "conf": threshold, "conf_signal":
    "logp" | "margin"}). Returns (new tour, stats) with times split into network, beam and LKH."""
    kind = policy["kind"]
    t = np.array(tour, dtype=np.int64)
    n = len(t)
    stats = {"windows": 0, "to_model": 0, "to_lkh": 0, "improved": 0, "passes": 0, "gain": 0.0,
             "ms_net": 0.0, "ms_beam": 0.0, "ms_lkh": 0.0}
    for _ in range(max_passes):
        stats["passes"] += 1
        pass_gain = 0.0
        for offset in (0, w // 2):
            wins = window_starts(n, w, offset)
            idx = [(s + np.arange(length)) % n for s, length in wins]
            sets = [coords[t[i]] for i in idx]
            heats = [None] * len(sets)
            if kind != "lkh":
                before = heatmaps.seconds
                heats = heatmaps(sets)
                stats["ms_net"] += 1000 * (heatmaps.seconds - before)
            for i, pts, heat in zip(idx, sets, heats):
                stats["windows"] += 1
                order, action = None, "none"
                if kind in ("model", "mixed"):
                    start = time.perf_counter()
                    beam_order, logp, margin = beam_with_confidence(heat, pts, beam_width)
                    stats["ms_beam"] += 1000 * (time.perf_counter() - start)
                if kind == "model":
                    action = "model"
                elif kind == "lkh":
                    action = "lkh"
                elif kind == "dis":
                    action = "lkh" if segment_disagreement(heat) >= policy["dis"] else "none"
                else:
                    conf = logp if policy["conf_signal"] == "logp" else margin
                    if segment_disagreement(heat) >= policy["dis"] and conf < policy["conf"]:
                        action = "lkh"
                    elif conf >= policy["conf"]:
                        action = "model"
                if action == "model":
                    stats["to_model"] += 1
                    order = beam_order
                elif action == "lkh":
                    stats["to_lkh"] += 1
                    start = time.perf_counter()
                    order = lkh_path(pts)
                    stats["ms_lkh"] += 1000 * (time.perf_counter() - start)
                if order is None:
                    continue
                old, new = path_length(pts), path_length(pts[order])
                if new < old - IMPROVE_TOL:
                    t[i] = t[i][order]
                    stats["improved"] += 1
                    pass_gain += old - new
        stats["gain"] += pass_gain
        if pass_gain <= 0:
            break
    return t, stats
