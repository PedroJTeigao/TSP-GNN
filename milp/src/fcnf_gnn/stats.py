"""Paired comparisons across instances and seeds (copied from the TSP project: scripts/summarize.py
paired_compare and scripts/phase2_beam_report.py paired).

Lower is better for every metric compared here (cost gap, seconds).
"""

import numpy as np
from scipy.stats import wilcoxon

TIE_TOL = 1e-6   # difference below which two instances count as a tie


def paired_compare(a, ref, tol=TIE_TOL):
    """a − ref instance by instance: % better / equal / worse, mean and median difference, two-sided
    Wilcoxon p (exact ties dropped; p = 1 if everything ties)."""
    diff = np.asarray(a, float) - np.asarray(ref, float)
    nonzero = np.abs(diff) > tol
    p = wilcoxon(diff[nonzero]).pvalue if nonzero.sum() > 0 else 1.0
    return {"better": 100 * np.mean(diff < -tol), "equal": 100 * np.mean(~nonzero),
            "worse": 100 * np.mean(diff > tol), "mean_diff": float(diff.mean()),
            "median_diff": float(np.median(diff)), "p": float(p)}


def paired(a_list, ref_list, threshold):
    """One paired comparison per seed; p = max over seeds. A seed whose median difference is zero is
    neutral; the other seeds' medians must agree in sign (if all are zero, the mean decides).
    verdict: "better" (a lower), "worse" or "n.s."."""
    res = [paired_compare(a, r) for a, r in zip(a_list, ref_list)]
    avg = {k: float(np.mean([x[k] for x in res])) for k in res[0]}
    p_max = max(x["p"] for x in res)
    signs = {np.sign(x["median_diff"]) for x in res if abs(x["median_diff"]) > TIE_TOL}
    if len(signs) > 1:
        verdict = "n.s."
    else:
        direction = signs.pop() if signs else np.sign(avg["mean_diff"])
        verdict = ("better" if direction < 0 else "worse") if p_max < threshold and direction != 0 else "n.s."
    return {**avg, "p_max": p_max, "seeds": len(res), "verdict": verdict}
