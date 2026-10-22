"""Generator statistics of the labelled FCNF sets: size, MILP time, open-arc fraction (base rate),
instances without a proof of optimality. Writes a markdown table.

Example:
    python scripts/data_stats.py --data data/fcnf20_train.npz data/fcnf20_test.npz --out results/phase1_fcnf_data_stats.md
"""

import argparse
import os

import numpy as np

from fcnf_gnn.data import load_dataset, num_instances


def stats(d):
    ptr = d["arc_ptr"]
    open_frac = np.array([d["y"][ptr[i]:ptr[i + 1]].mean() for i in range(num_instances(d))])
    status = d["status"].astype(int)
    unproven = status == 1
    row = {"num": num_instances(d), "n": int(d["n"]), "arcs": float(np.diff(ptr).mean()),
           "sec_mean": float(d["seconds"].mean()), "sec_median": float(np.median(d["seconds"])),
           "sec_max": float(d["seconds"].max()), "open": float(d["y"].mean()),
           "open_inst_std": float(open_frac.std()), "optimal": int((status == 0).sum()),
           "unproven": int(unproven.sum()), "no_solution": int((status == 2).sum()),
           "gap_unproven": float(100 * d["mip_gap"][unproven].mean()) if unproven.any() else None,
           "rescaled": int((d["cap_rescaled"] > 0).sum())}
    row["by_ratio"] = {float(r): (int((d["ratio"] == r).sum()), float(d["seconds"][d["ratio"] == r].mean()),
                                  float(open_frac[d["ratio"] == r].mean()))
                       for r in np.unique(d["ratio"])}
    return row


def main():
    parser = argparse.ArgumentParser(description="Statistics of labelled FCNF sets.")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", default="results/phase1_fcnf_data_stats.md")
    args = parser.parse_args()
    rows = {os.path.splitext(os.path.basename(p))[0]: stats(load_dataset(p)) for p in args.data if os.path.exists(p)}

    lines = ["# FCNF phase 1: generator statistics", "",
             "MILP = HiGHS, 1 thread per instance, 8 instances in parallel. *open* = fraction of arcs open in "
             "the best solution (the base rate of the label). *unproven* = stopped at the time limit with a "
             "feasible solution; *gap* = HiGHS relative MIP gap of those.", "",
             "| set | n | instances | arcs (mean) | MILP s mean | median | max | open | std open / inst. "
             "| optimal | unproven | gap unproven (%) | no solution | capacities rescaled |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        gap = "—" if r["gap_unproven"] is None else f"{r['gap_unproven']:.2f}"
        lines.append(f"| {name} | {r['n']} | {r['num']} | {r['arcs']:.1f} | {r['sec_mean']:.2f} | {r['sec_median']:.2f} "
                     f"| {r['sec_max']:.1f} | {r['open']:.3f} | {r['open_inst_std']:.3f} | {r['optimal']} | {r['unproven']} "
                     f"| {gap} | {r['no_solution']} | {r['rescaled']} |")
    lines += ["", "## By fixed-cost ratio F / (c u)", "",
              "| set | ratio | instances | MILP s mean | open |", "|---|---|---|---|---|"]
    for name, r in rows.items():
        for ratio, (num, sec, op) in r["by_ratio"].items():
            lines.append(f"| {name} | {ratio:g} | {num} | {sec:.2f} | {op:.3f} |")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
