"""Table of ablation A4 (SoftDist heatmaps against the TAGCN network), from results/ablations/a4 (scripts/ablation_a4.py).

Windows: SoftDist greedy / beam B = 10 against the network's E3 windows (TAGCN K = 3, seeds 0-2, results/phase3_benchmark/e3),
paired Wilcoxon against seed 0 per instance. Triage: SoftDist scores at top 10 / 20 / 50 % against the network's scores of
A1 (seed 0, no memo, same protocol) and random (A1, rng seed 0), same x, so same share of windows to LKH: gaps only, since the
A4 runs shared the machine with other runs. Fu-1k triage rows use the 64 instances neither learned selector was trained on.

Writes results/ablations/a4.md and a4.json.

    python scripts/ablation_a4_report.py
"""

import json

import numpy as np
from scipy.stats import wilcoxon

from tsp_gnn.learned_triage import split_instances

A4 = "results/ablations/a4"
A1 = "results/ablations/a1/w20"
E3 = "results/phase3_benchmark/e3/window"
NAMES = {"fu1k": "tsp1000_test_concorde", "fu10k": "tsp10000_test_concorde"}


def npz(path):
    return dict(np.load(path))


def test(a, b):
    d = a - b
    return float(d.mean()), float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, int(np.sum(d < 0)), len(d)


def fmt(t):
    diff, p, wins, n = t
    return f"{diff:+.3f} | {p:.2g} | {wins}/{n}"


def main():
    tau = json.load(open(f"{A4}/tau.json"))
    rec = {"tau": tau, "window": {}, "triage": {}}
    lines = [f"# Ablation A4: SoftDist against the network (τ = {tau['tau']}, chosen on 20 new instances, n = 1000)", "",
             "τ grid (mean length after beam B = 10 windows; initial {:.4f}):".format(tau["initial_mean_length"]), "",
             "| τ | " + " | ".join(tau["grid"]) + " |", "|---|" + "---:|" * len(tau["grid"]),
             "| length | " + " | ".join(f"{v:.4f}" for v in tau["grid"].values()) + " |", "",
             "## Windows (w = 20, ≤ 5 passes, all instances)", "",
             "| set | decoder | initial gap % | network (TAGCN, 3 seeds) gap % | SoftDist gap % | 2-opt control gap % | "
             "SoftDist − network seed 0 (pp) | Wilcoxon p | instances SoftDist better |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s, name in NAMES.items():
        two = npz(f"{A4}/window/2opt/{name}.npz")["2opt_w20__gap"]
        for dec in ("greedy", "beam10"):
            soft = npz(f"{A4}/window/{dec}/{name}.npz")
            net = [npz(f"{E3}/tagcn_k3_seed{k}/{name}.npz")[f"{dec}_w20__gap"] for k in range(3)]
            t = test(soft[f"{dec}_w20__gap"], net[0])
            means = [x.mean() for x in net]
            rec["window"][f"{s}_{dec}"] = {"softdist": float(soft[f"{dec}_w20__gap"].mean()), "network": means, "test_seed0": t}
            lines.append(f"| {s} | {dec} | {soft['initial__gap'].mean():.3f} | {np.mean(means):.3f} ± {np.std(means):.3f} | "
                         f"{soft[f'{dec}_w20__gap'].mean():.3f} | {two.mean():.3f} | {fmt(t)} |")
    lines += ["", "## Triage (w = 20, ≤ 3 passes, same share of windows to LKH; gaps only)", "",
              "| set | x % | random | network: disagreement | SoftDist: disagreement | network: learned | "
              "SoftDist into the network's selector | selector retrained on SoftDist | retrained − network learned (pp) | Wilcoxon p | instances retrained better |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, name in NAMES.items():
        r = npz(f"{A1}/random_s0/{name}.npz")
        nd, nl = npz(f"{A1}/fast_disagreement_seed0/{name}.npz"), npz(f"{A1}/fast_learned_seed0/{name}.npz")
        sd, so, sn = (npz(f"{A4}/triage/{k}/{name}.npz") for k in ("dis", "learned_old", "learned_new"))
        m = ~split_instances(128, seed=0) if s == "fu1k" else np.ones(len(r["initial__gap"]), dtype=bool)
        for x in (10, 20, 50):
            g = {"random": r[f"random_s0_top{x}__gap"], "net_dis": nd[f"fast_disagreement_top{x}__gap"],
                 "soft_dis": sd[f"softdist_dis_top{x}__gap"], "net_learned": nl[f"fast_learned_top{x}__gap"],
                 "soft_old": so[f"softdist_learned_old_top{x}__gap"], "soft_new": sn[f"softdist_learned_new_top{x}__gap"]}
            g = {k: v[m] for k, v in g.items()}
            t = test(g["soft_new"], g["net_learned"])
            rec["triage"][f"{s}_{x}"] = {**{k: float(v.mean()) for k, v in g.items()}, "test_new_vs_net": t}
            lines.append(f"| {s} | {x} | " + " | ".join(f"{g[k].mean():.3f}" for k in
                         ("random", "net_dis", "soft_dis", "net_learned", "soft_old", "soft_new")) + f" | {fmt(t)} |")
    open("results/ablations/a4.md", "w").write("\n".join(lines) + "\n")
    json.dump(rec, open("results/ablations/a4.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
