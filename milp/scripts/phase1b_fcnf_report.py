"""FCNF phase 1b report: the full MILP at equal time, the trust region (predict-and-search) and feasible warm starts.

Reads data/fcnf*_test.npz (+ data/fcnf100_test_ref600.npz), results/phase1_decode/<run>/<set>__guided.npz,
results/phase1b/decode[_tr120]/<run>/<set>__trust.npz, results/phase1b/traj/<tag>/<set>.npz and
results/phase1b/equal_time_check/<run>/<set>.npz; writes results/phase1b_fcnf_report.md and
results/figs/phase1b_fcnf_equal_time.png. Missing pieces are listed, not fatal. The three summary lines come from
results/phase1b_fcnf_notes.md (written by hand after reading the results).

The MILP at time t on an instance = the best solution of its improving-solution trajectory (scripts/
milp_trajectory.py, no warm start) found by t; no solution yet = the decoder wins that instance. gap = 100 (cost /
reference − 1), reference as in phase 1 (MILP optimum; on fcnf100 the best of the 120 s and 600 s runs).

Example:
    python scripts/phase1b_fcnf_report.py
"""

import argparse
import os

import numpy as np
from scipy.stats import wilcoxon

from fcnf_gnn.data import load_dataset
from fcnf_gnn.stats import paired

SETS = {"fcnf20_test": "fcnf20", "fcnf50_test": "fcnf50", "fcnf100_test": "fcnf100"}
ENCODERS = {"gcn": "GCN", "tagcn_k3": "TAGCN K=3"}
SEEDS = (0, 1, 2)
QS = (20, 30, 50)
DELTAS = (5, 10, 20)
FIXED = (1, 2, 5, 10, 25, 50)
TR = [f"tr_q{q}_d{d}" for q in QS for d in DELTAS]
PHASE1 = [f"prune_q{q}" for q in QS] + ["delegate"]
LABELS = {**{f"prune_q{q}": f"Pruning q = {q} %" for q in QS}, "delegate": "Delegation",
          **{f"tr_q{q}_d{d}": f"Trust region q = {q} %, Δ = {d}" for q in QS for d in DELTAS}}
THRESHOLDS = (1.0, 0.1)
ALPHA = 0.05
NO_SOLUTION = 1e6        # gap of "no solution yet" in the paired tests (the worst rank)


def fmt(values, digits=2):
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return "missing"
    return f"{np.mean(v):.{digits}f}" + (f" ± {np.std(v, ddof=1):.{digits}f}" if len(v) > 1 else "")


def load_npz(path, missing):
    if os.path.exists(path):
        return dict(np.load(path))
    missing.append(path)
    return None


# ---------------------------------------------------------------- loading

def load_all(args):
    missing = []
    data, refs = {}, {}
    for s in SETS:
        d = load_dataset(os.path.join(args.data_dir, f"{s}.npz"))
        ref = d["cost"].copy()
        rerun = os.path.join(args.data_dir, f"{s}_ref600.npz")
        if os.path.exists(rerun):
            ref = np.load(rerun)["ref_cost"]
        data[s], refs[s] = d, ref
    traj, phase1, trust, checks = {}, {}, {}, {}
    for s in SETS:
        tags = ["none", "slope"] + [f"trust_tagcn_k3_seed{seed}" for seed in (SEEDS if s != "fcnf100_test" else (0,))]
        for tag in tags:
            traj[(tag, s)] = load_npz(os.path.join(args.phase1b, "traj", tag, f"{s}.npz"), missing)
        for enc in ENCODERS:
            for seed in SEEDS:
                run = f"{enc}_seed{seed}"
                phase1[(run, s)] = load_npz(os.path.join(args.phase1_decode, run, f"{s}__guided.npz"), missing)
        for seed in (SEEDS if s != "fcnf100_test" else (0,)):
            run = f"tagcn_k3_seed{seed}"
            trust[(run, s)] = load_npz(os.path.join(args.phase1b, "decode", run, f"{s}__trust.npz"), missing)
        checks[s] = load_npz(os.path.join(args.phase1b, "equal_time_check", "tagcn_k3_seed0", f"{s}.npz"), missing)
    tr120 = load_npz(os.path.join(args.phase1b, "decode_tr120", "tagcn_k3_seed0", "fcnf100_test__trust.npz"), missing)
    return data, refs, traj, phase1, trust, tr120, checks, missing


# ---------------------------------------------------------------- trajectory helpers

def instance_traj(t, i):
    return t["traj"][int(t["traj_ptr"][i]):int(t["traj_ptr"][i + 1])]


def cost_at(t, i, seconds):
    """Best cost of instance i found by `seconds` (nan if none yet)."""
    tr = instance_traj(t, i)
    ok = tr[:, 0] <= seconds
    return float(tr[ok, 1].min()) if ok.any() else np.nan


def milp_gap_at(t, ref, seconds):
    """Per-instance gap (%) of the MILP stopped at seconds[i] (nan = no solution yet)."""
    return np.array([100 * (cost_at(t, i, seconds[i]) / ref[i] - 1) for i in range(len(seconds))])


def time_to(t, ref, i, thr):
    """MILP seconds until the first solution with gap <= thr % (nan if never)."""
    tr = instance_traj(t, i)
    ok = tr[:, 1] <= ref[i] * (1 + thr / 100) + 1e-9 * ref[i]
    return float(tr[ok, 0].min()) if ok.any() else np.nan


def decoder_gap(a, method, ref):
    cost = a[f"{method}__cost"]
    return 100 * (cost / ref[:len(cost)] - 1), a[f"{method}__seconds"]


# ---------------------------------------------------------------- sections

def setup_section(data, traj):
    lines = ["## 1. What was run", "",
             "- **Equal-time baseline**: one run of the full MILP per instance (no warm start; time limit as the "
             "reference: none on fcnf20, 120 s on fcnf50 / fcnf100) records every improving solution. The MILP "
             "stopped at time t is read from it, for every decoder and instance (t = the time the decoder used "
             "there) and for the fixed limits 1, 2, 5, 10, 25 and 50 s. Section 2.3 checks this against runs "
             "with the time limit itself.",
             "- **Trust region** (Han et al. 2023, predict-and-search): the top q % arcs by p are free and at "
             "most Δ of the others may open (Σ y ≤ Δ on them); no repair. TAGCN K=3 (the better encoder of "
             "phase 1): 3 seeds on fcnf20 / fcnf50 with the usual MILP limit; on fcnf100 seed 0 only, the first "
             "100 instances, with a 25 s limit (about the time of pruning q50 there), plus q50 / Δ20 with the "
             "usual 120 s as a reference point.",
             "- **Feasible warm start**: the full MILP started from (a) the trust-region solution q50 / Δ20 "
             "(on fcnf100 the 25 s one) or (b) the slope scaling solution; the heuristic's time is added to the "
             "MILP's. On fcnf100 the first 100 instances.",
             "- Pruning and delegation are the phase 1 runs (both encoders, 3 seeds), times as measured there "
             "(some fcnf20 / fcnf50 results with 2 processes in parallel instead of 8, see the phase 1 report). "
             "Everything else here: 8 processes, one HiGHS thread each.", ""]
    rows = []
    for s in SETS:
        t = traj.get(("none", s))
        if t is not None:
            rows.append(f"| {SETS[s]} | {len(t['seconds'])} | {t['seconds'].mean():.2f} | {data[s]['seconds'][:len(t['seconds'])].mean():.2f} "
                        f"| {(t['status'] == 0).sum()} | {np.diff(t['traj_ptr']).mean():.1f} |")
    if rows:
        lines += ["Full MILP of this phase (trajectory run) vs the phase 1 reference run:", "",
                  "| set | instances | MILP s (this run) | MILP s (phase 1) | proven | improving solutions per instance |",
                  "|---|---|---|---|---|---|"] + rows + [""]
    return lines


def milp_curve(t, ref, limits):
    """Per limit: mean gap over the instances with a solution, % with a solution."""
    out = {}
    n = len(t["seconds"])
    for L in limits:
        g = milp_gap_at(t, ref, np.full(n, float(L)))
        out[L] = (np.nanmean(g) if np.isfinite(g).any() else np.nan, 100 * np.isfinite(g).mean())
    return out


def equal_time_section(data, refs, traj, phase1, trust, tr120, checks):
    lines = ["## 2. Decoders vs the full MILP at equal time", "",
             "### 2.1 The full MILP stopped at a fixed time", "",
             "Mean gap (%) over the instances that have a solution by then (in brackets: % of instances with a "
             "solution, when below 100).", "",
             "| set | " + " | ".join(f"{L} s" for L in FIXED) + " | end (limit) |", "|---|" + "---|" * (len(FIXED) + 1)]
    for s in SETS:
        t = traj.get(("none", s))
        if t is None:
            continue
        ref = refs[s][:len(t["seconds"])]
        curve = milp_curve(t, ref, FIXED + (1e9,))
        cells = [f"{g:.2f}" + (f" ({pct:.0f} %)" if pct < 99.95 else "") for g, pct in curve.values()]
        lines.append(f"| {SETS[s]} | " + " | ".join(cells) + " |")
    lines += ["", "### 2.2 Each decoder vs the MILP with the decoder's time", "",
              "Per instance, the MILP gets exactly the time the decoder used there. *MILP no sol.* = % instances "
              "where the MILP has no solution yet (counted as a decoder win). Wilcoxon signed-rank on the gap, "
              "one test per seed, p = max over seeds; Bonferroni within each size. Δ = median of decoder gap − "
              "MILP gap (% points), averaged over seeds. *better* = the decoder has the lower gap.", ""]
    for s in SETS:
        t = traj.get(("none", s))
        if t is None:
            continue
        n_milp = len(t["seconds"])
        ref = refs[s]
        rows = []
        for label, method, runs, store in (
                [(f"{LABELS[m]} ({ENCODERS[e]})", m, [f"{e}_seed{k}" for k in SEEDS], phase1) for e in ENCODERS for m in PHASE1]
                + [(f"{LABELS[m]} (TAGCN K=3)", m, [f"tagcn_k3_seed{k}" for k in SEEDS], trust) for m in TR]):
            dec, mil, secs, nosol = [], [], [], []
            for run in runs:
                a = store.get((run, s))
                if a is None or f"{method}__cost" not in a:
                    continue
                g, sec = decoder_gap(a, method, ref)
                n = min(len(g), n_milp)
                g, sec = g[:n], sec[:n]
                mg = milp_gap_at(t, ref, sec)
                ok = np.isfinite(g)                      # trust region: instances where it found a solution
                dec.append(g[ok])
                mil.append(np.where(np.isfinite(mg), mg, NO_SOLUTION)[ok])
                secs.append(sec.mean())
                nosol.append(100 * (~np.isfinite(mg[ok])).mean())
            if not dec:
                continue
            rows.append((label, dec, mil, secs, nosol))
        threshold = ALPHA / max(1, len(rows))
        lines += [f"### {SETS[s]} ({len(rows)} tests, threshold {threshold:.1e})", "",
                  "| decoder | instances | s | decoder gap | MILP gap at that time | MILP no sol. (%) | Δ median "
                  "| better / equal / worse (%) | p max | seeds | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for label, dec, mil, secs, nosol in rows:
            r = paired(dec, mil, threshold)
            milp_mean = [np.mean(m[m < NO_SOLUTION]) if (m < NO_SOLUTION).any() else np.nan for m in mil]
            lines.append(f"| {label} | {len(dec[0])} | {fmt(secs)} | {fmt([np.mean(d) for d in dec])} | {fmt(milp_mean)} "
                         f"| {fmt(nosol, 0)} | {r['median_diff']:+.3f} | {r['better']:.0f} / {r['equal']:.0f} / "
                         f"{r['worse']:.0f} | {r['p_max']:.1e} | {r['seeds']} | {r['verdict']} |")
        lines.append("")
    lines += ["### 2.3 Check: trajectory vs a run with the time limit itself", "",
              "tagcn_k3_seed0, first instances of each set: the MILP re-run with time_limit = the decoder's time "
              "(or the fixed limit) against the cost read from the trajectory. The two differ only by timing noise "
              "(an improving solution found near the limit can fall on either side). *same* = equal cost "
              "(rel. 1e-6); p = Wilcoxon on the gap, run vs trajectory.", "",
              "| set | limit | instances | gap, run with the limit | gap, trajectory | no sol. run / traj. | same (%) | p |",
              "|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        c, t = checks.get(s), traj.get(("none", s))
        if c is None or t is None:
            continue
        idx = c["index"].astype(int)
        for key in sorted({k[:-len("__cost")] for k in c if k.endswith("__cost")},
                          key=lambda k: (k.startswith("fixed"), float(k.split("_")[1]) if k.startswith("fixed") else 0, k)):
            lim, run_cost = c[f"{key}__limit"], c[f"{key}__cost"]
            tr_cost = np.array([cost_at(t, i, lim[k]) for k, i in enumerate(idx)])
            ref = refs[s][idx]
            g_run, g_tr = 100 * (run_cost / ref - 1), 100 * (tr_cost / ref - 1)
            same = np.isclose(run_cost, tr_cost, rtol=1e-6) | (np.isnan(run_cost) & np.isnan(tr_cost))
            a, b = np.where(np.isfinite(g_run), g_run, NO_SOLUTION), np.where(np.isfinite(g_tr), g_tr, NO_SOLUTION)
            diff = a - b
            p = wilcoxon(diff[np.abs(diff) > 1e-9]).pvalue if (np.abs(diff) > 1e-9).sum() > 0 else 1.0
            name = f"{key.split('_')[1]} s" if key.startswith("fixed") else f"time of {LABELS.get(key, key)}"
            lines.append(f"| {SETS[s]} | {name} | {len(idx)} | {np.nanmean(g_run):.2f} | {np.nanmean(g_tr):.2f} "
                         f"| {np.isnan(g_run).sum()} / {np.isnan(g_tr).sum()} | {100 * same.mean():.0f} | {p:.2g} |")
    return lines + [""]


def trust_section(data, refs, phase1, trust, tr120):
    lines = ["## 3. Trust region vs pruning with repair", "",
             "TAGCN K=3. gap over the instances with a solution, mean ± std over seeds. *no sol.* = % instances where "
             "the restricted MILP found no solution (infeasible, or none by the time limit); *proven* = % where it was "
             "solved to optimality; *outside* = open arcs outside the top q % in the solution (mean); *binding* = % of "
             "solved instances with outside = Δ. The pruning row of the same q is the phase 1 result (with repair) on "
             "the same instances. Tests: trust region vs pruning, same q and seed, instances where the trust region "
             "has a solution; Bonferroni within each size.", ""]
    for s in SETS:
        runs = [r for r in (f"tagcn_k3_seed{k}" for k in SEEDS) if trust.get((r, s)) is not None]
        if not runs:
            continue
        ref = refs[s]
        n = len(trust[(runs[0], s)]["tr_q20_d5__cost"])
        limit = {"fcnf20_test": "no limit", "fcnf50_test": "120 s", "fcnf100_test": "25 s"}[s]
        lines += [f"### {SETS[s]} ({n} instances, {len(runs)} seed(s), restricted MILP limit: {limit})", "",
                  "| method | gap (%) | s | no sol. (%) | proven (%) | outside | binding (%) |", "|---|---|---|---|---|---|---|"]
        tests = []
        for q in QS:
            pr = [phase1[(r, s)] for r in runs if phase1.get((r, s)) is not None]
            if pr:
                g = [decoder_gap(a, f"prune_q{q}", ref)[0][:n] for a in pr]
                sec = [a[f"prune_q{q}__seconds"][:n] for a in pr]
                lines.append(f"| Pruning q = {q} % (with repair) | {fmt([np.nanmean(x) for x in g])} | "
                             f"{fmt([x.mean() for x in sec])} | 0 | — | — | — |")
            for d in DELTAS:
                m = f"tr_q{q}_d{d}"
                arrs = [trust[(r, s)] for r in runs]
                g = [decoder_gap(a, m, ref)[0] for a in arrs]
                sec = [a[f"{m}__seconds"] for a in arrs]
                used = [a[f"{m}__outside_used"] for a in arrs]
                solved = [np.isfinite(x) for x in g]
                proven = [100 * a[f"{m}__proven"].mean() if f"{m}__proven" in a else
                          (100 * np.isfinite(gg).mean() if s == "fcnf20_test" else np.nan) for a, gg in zip(arrs, g)]
                lines.append(f"| {LABELS[m]} | {fmt([np.nanmean(x) for x in g])} | {fmt([x.mean() for x in sec])} | "
                             f"{fmt([100 * (~ok).mean() for ok in solved], 1)} | {fmt(proven, 0)} | "
                             f"{fmt([u[ok].mean() for u, ok in zip(used, solved)], 1)} | "
                             f"{fmt([100 * (u[ok] == d).mean() for u, ok in zip(used, solved)], 0)} |")
                if pr:
                    pg = [decoder_gap(a, f"prune_q{q}", ref)[0][:n] for a in pr]
                    ps = [a[f"prune_q{q}__seconds"][:n] for a in pr]
                    tests.append((f"{LABELS[m]} vs pruning q{q}", "gap", [x[ok] for x, ok in zip(g, solved)],
                                  [x[ok] for x, ok in zip(pg, solved)]))
                    tests.append((f"{LABELS[m]} vs pruning q{q}", "seconds", sec, ps))
        if s == "fcnf100_test" and tr120 is not None:
            m = "tr_q50_d20"
            g = decoder_gap(tr120, m, ref)[0]
            ok = np.isfinite(g)
            lines.append(f"| {LABELS[m]}, 120 s limit (reference point) | {np.nanmean(g):.2f} | "
                         f"{tr120[f'{m}__seconds'].mean():.2f} | {100 * (~ok).mean():.1f} | "
                         f"{100 * tr120[f'{m}__proven'].mean():.0f} | {tr120[f'{m}__outside_used'][ok].mean():.1f} | "
                         f"{100 * (tr120[f'{m}__outside_used'][ok] == 20).mean():.0f} |")
        threshold = ALPHA / max(1, len(tests))
        lines += ["", f"Paired tests ({len(tests)}, threshold {threshold:.1e}); Δ = trust region − pruning (mean and median over instances, averaged over seeds):",
                  "", "| comparison | metric | Δ mean | Δ median | better / equal / worse (%) | p max | seeds | verdict |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, metric, a, b in tests:
            r = paired(a, b, threshold)
            lines.append(f"| {name} | {metric} | {r['mean_diff']:+.3f} | {r['median_diff']:+.3f} | {r['better']:.0f} / {r['equal']:.0f} / "
                         f"{r['worse']:.0f} | {r['p_max']:.1e} | {r['seeds']} | {r['verdict']} |")
        lines.append("")
    return lines


def warm_section(refs, traj):
    lines = ["## 4. Feasible warm start", "",
             "Time until the first solution with gap ≤ 1 % and ≤ 0.1 % (against the reference), and until the "
             "optimality proof. *total* = heuristic + MILP (the time to a threshold is the heuristic's time alone if "
             "the warm start is already within it). Means over instances, with an instance that never got there "
             "counted at the limit (so the mean is a lower bound when *reached* < 100 %); median in brackets. Trust "
             "region rows: mean ± std over seeds of those means.", ""]
    for s in SETS:
        base = traj.get(("none", s))
        if base is None:
            continue
        n = len(traj[("slope", s)]["seconds"]) if traj.get(("slope", s)) is not None else len(base["seconds"])
        ref = refs[s]
        limit = 120.0 if s != "fcnf20_test" else np.inf
        variants = [("No warm start", ["none"]), ("Slope scaling", ["slope"]),
                    ("Trust region q50 / Δ20 (TAGCN K=3)", [k[0] for k in traj if k[1] == s and k[0].startswith("trust_")])]
        lines += [f"### {SETS[s]} ({n} instances)", "",
                  "| warm start | warm-start gap (%) | heuristic s | ≤ 1 %: total s | reached (%) | ≤ 0.1 %: total s "
                  "| reached (%) | proof: total s | proven (%) |", "|---|---|---|---|---|---|---|---|---|"]
        per = {}
        for label, tags in variants:
            stats = []
            for tag in sorted(tags):
                t = traj.get((tag, s))
                if t is None:
                    continue
                heur = t["heur_seconds"][:n]
                row = {"warm_gap": np.nanmean(100 * (t["warm_cost"][:n] / ref[:n] - 1)) if t["warm_used"][:n].any() else np.nan,
                       "heur": heur.mean()}
                for thr in THRESHOLDS:
                    tt = np.array([time_to(t, ref, i, thr) for i in range(n)])
                    total = np.where(np.isfinite(tt), tt + heur, limit + heur)
                    row[thr] = (total, np.isfinite(tt).mean() * 100)
                proof = np.where(t["status"][:n] == 0, t["seconds"][:n] + heur, limit + heur)
                row["proof"] = (proof, (t["status"][:n] == 0).mean() * 100)
                stats.append(row)
            if not stats:
                continue
            per[label] = stats

            def cell(key):
                means = [st[key][0].mean() for st in stats]
                med = [np.median(st[key][0]) for st in stats]
                return f"{fmt(means)} ({np.mean(med):.2f})", fmt([st[key][1] for st in stats], 0)
            c1, c2, cp = cell(1.0), cell(0.1), cell("proof")
            lines.append(f"| {label} | {fmt([st['warm_gap'] for st in stats])} | {fmt([st['heur'] for st in stats])} "
                         f"| {c1[0]} | {c1[1]} | {c2[0]} | {c2[1]} | {cp[0]} | {cp[1]} |")
        tests = []
        if "No warm start" in per:
            ref_stats = per["No warm start"][0]
            for label in ("Slope scaling", "Trust region q50 / Δ20 (TAGCN K=3)"):
                for key, name in ((1.0, "time to ≤ 1 %"), (0.1, "time to ≤ 0.1 %"), ("proof", "time to proof")):
                    if label in per:
                        tests.append((f"{label} vs no warm start", name, [st[key][0] for st in per[label]],
                                      [ref_stats[key][0]] * len(per[label])))
        threshold = ALPHA / max(1, len(tests))
        lines += ["", f"Paired tests on the total time ({len(tests)}, threshold {threshold:.1e}); Δ = warm start − none (s; mean and median over instances, averaged over seeds; the verdict follows the median):", "",
                  "| comparison | metric | Δ mean | Δ median | better / equal / worse (%) | p max | seeds | verdict |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, metric, a, b in tests:
            r = paired(a, b, threshold)
            lines.append(f"| {name} | {metric} | {r['mean_diff']:+.3f} | {r['median_diff']:+.3f} | {r['better']:.0f} / {r['equal']:.0f} / "
                         f"{r['worse']:.0f} | {r['p_max']:.1e} | {r['seeds']} | {r['verdict']} |")
        lines.append("")
    return lines


def figure(refs, traj, phase1, trust, tr120, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

    sets = [s for s in SETS if traj.get(("none", s)) is not None]
    if not sets:
        return False
    colors = {"milp": "#222222", "prune": "#1f77b4", "delegate": "#2ca02c", "tr": "#d62728"}
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4.4), squeeze=False)
    for ax, s in zip(axes[0], sets):
        t = traj[("none", s)]
        n = len(t["seconds"])
        ref = refs[s][:n]
        grid = np.logspace(-2.5, np.log10(max(t["seconds"].max(), 1.0)), 60)
        curve = milp_curve(t, ref, grid)
        xs = [L for L, (g, pct) in curve.items() if pct >= 99]           # only where (almost) every instance has one
        ax.plot(xs, [curve[L][0] for L in xs], color=colors["milp"], lw=1.6, label="Full MILP stopped at t", zorder=2)
        fixed = milp_curve(t, ref, [L for L in FIXED if L <= max(t["seconds"].max(), 1.0)])
        fx = [L for L, (g, pct) in fixed.items() if pct >= 99]
        ax.scatter(fx, [fixed[L][0] for L in fx], color=colors["milp"], s=22, zorder=3, label="MILP at 1, 2, 5, 10, 25, 50 s")
        for enc, marker in (("gcn", "o"), ("tagcn_k3", "^")):
            for m in PHASE1:
                arr = [phase1.get((f"{enc}_seed{k}", s)) for k in SEEDS]
                arr = [a for a in arr if a is not None]
                if arr:
                    x = np.mean([a[f"{m}__seconds"].mean() for a in arr])
                    y = np.mean([np.nanmean(decoder_gap(a, m, refs[s])[0]) for a in arr])
                    ax.scatter(x, y, s=50, marker=marker, color=colors["prune" if m.startswith("prune") else "delegate"],
                               edgecolor="white", linewidth=1.2, zorder=4,
                               label=f"{'Pruning q = 20/30/50 %' if m.startswith('prune') else 'Delegation'} ({ENCODERS[enc]})"
                               if m in ("prune_q20", "delegate") else None)
        arr = [trust.get((f"tagcn_k3_seed{k}", s)) for k in SEEDS]
        arr = [a for a in arr if a is not None]
        for k, m in enumerate(TR):
            if arr:
                x = np.mean([a[f"{m}__seconds"].mean() for a in arr])
                y = np.mean([np.nanmean(decoder_gap(a, m, refs[s])[0]) for a in arr])
                ax.scatter(x, y, s=50, marker="D", color=colors["tr"], edgecolor="white", linewidth=1.2, zorder=4,
                           label="Trust region, 9 (q, Δ) (TAGCN K=3)" if k == 0 else None)
        if s == "fcnf100_test" and tr120 is not None:
            ax.scatter(tr120["tr_q50_d20__seconds"].mean(), np.nanmean(decoder_gap(tr120, "tr_q50_d20", refs[s])[0]),
                       s=70, marker="D", facecolors="none", edgecolors=colors["tr"], linewidths=1.6, zorder=4,
                       label="Trust region q50 / Δ20, 120 s")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("time per instance (s, log)")
        ax.set_ylabel("cost gap to the reference (%, symlog)")
        ax.set_title(SETS[s])
        ax.grid(alpha=0.25)
    handles, labels = [], []
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="FCNF phase 1b report.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--phase1-decode", default="results/phase1_decode")
    parser.add_argument("--phase1b", default="results/phase1b")
    parser.add_argument("--notes", default="results/phase1b_fcnf_notes.md")
    parser.add_argument("--out", default="results/phase1b_fcnf_report.md")
    parser.add_argument("--fig", default="results/figs/phase1b_fcnf_equal_time.png")
    args = parser.parse_args()
    data, refs, traj, phase1, trust, tr120, checks, missing = load_all(args)

    notes = open(args.notes).read().strip() if os.path.exists(args.notes) else "(to be written after reading the results)"
    lines = ["# FCNF phase 1b: equal-time baseline, trust region, feasible warm start", "",
             "Same instances, models and reference solutions as phase 1 (`docs/phase1_fcnf_report.md`).", "",
             "## Summary", "", notes, ""]
    lines += setup_section(data, traj)
    lines += equal_time_section(data, refs, traj, phase1, trust, tr120, checks)
    lines += trust_section(data, refs, phase1, trust, tr120)
    lines += warm_section(refs, traj)
    if figure(refs, traj, phase1, trust, tr120, args.fig):
        lines += ["## 5. Figure", "", f"![Gap vs time]({os.path.relpath(args.fig, os.path.dirname(args.out))})", ""]
    lines += ["## Missing results", ""] + ([f"- `{m}`" for m in missing] if missing else ["None."])
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"written {args.out}; {len(missing)} missing")


if __name__ == "__main__":
    main()
