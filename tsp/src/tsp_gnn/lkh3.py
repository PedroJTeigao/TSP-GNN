"""Run the stand-alone LKH-3 binary (third_party/LKH-3/LKH) on one instance (phase 3 benchmark).

Unlike elkai (a library), the binary reads a parameter file and a problem file, writes the best tour to a
file and prints its timings. That lets us set any LKH parameter (TIME_LIMIT, INITIAL_TOUR_FILE,
CANDIDATE_SET_TYPE, ...) and measure:
    wall_seconds           total wall-clock time of the process (files, preprocessing, search)
    preprocessing_seconds  LKH's "Preprocessing time" (reading the problem, ascent, candidate sets)
    search_seconds         sum of the "Run i: ... Time = t sec." lines (the runs themselves)
The problem is the same as data.solve_lkh_coords: coordinates multiplied by SCALE, EUC_2D (each distance
rounded to the nearest integer). Lengths are always recomputed with float distances (data.tour_length).

LKH version: 3.0.6 (May 2019), built from the GitHub mirror cerebis/LKH3 (the official site cannot be
reached from the server). In this version TIME_LIMIT is per run and checked only between trials, so one
long trial can overshoot it: always compare budgets with wall_seconds, never with TIME_LIMIT.
"""

import os
import re
import subprocess
import time

import numpy as np

from tsp_gnn.data import SCALE, is_valid_tour, tour_length

LKH_BIN = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "LKH-3", "LKH")


def write_problem(path, coords):
    """TSPLIB file with the coordinates scaled by SCALE and rounded (EUC_2D)."""
    xy = np.rint(np.asarray(coords) * SCALE).astype(np.int64)
    with open(path, "w") as f:
        f.write(f"NAME : instance\nTYPE : TSP\nDIMENSION : {len(xy)}\nEDGE_WEIGHT_TYPE : EUC_2D\n"
                "NODE_COORD_SECTION\n")
        f.write("".join(f"{i + 1} {x} {y}\n" for i, (x, y) in enumerate(xy.tolist())))
        f.write("EOF\n")


def write_tour(path, tour):
    """TSPLIB tour file (1-indexed, ends with -1), e.g. for INITIAL_TOUR_FILE."""
    with open(path, "w") as f:
        f.write(f"TYPE : TOUR\nDIMENSION : {len(tour)}\nTOUR_SECTION\n")
        f.write("".join(f"{c + 1}\n" for c in np.asarray(tour).tolist()))
        f.write("-1\nEOF\n")


def read_tour(path):
    """Tour of a TSPLIB tour file as an (n,) 0-indexed array that starts at city 0."""
    with open(path) as f:
        lines = f.read().split("TOUR_SECTION", 1)[1].split()
    tour = []
    for token in lines:
        if token == "-1":
            break
        tour.append(int(token) - 1)
    tour = np.array(tour, dtype=np.int64)
    return np.roll(tour, -int(np.where(tour == 0)[0][0]))


def parse_output(text):
    """Timings and costs printed by LKH (TRACE_LEVEL = 1)."""
    pre = re.search(r"Preprocessing time = ([\d.]+) sec", text)
    runs = re.findall(r"^Run \d+: Cost = (-?\d+).*?Time = ([\d.]+) sec", text, flags=re.M)
    return {
        "preprocessing_seconds": float(pre.group(1)) if pre else float("nan"),
        "run_costs": [int(c) for c, _ in runs],
        "run_seconds": [float(t) for _, t in runs],
    }


def solve(coords, workdir, params=None, initial_tour=None, timeout=None):
    """Run LKH on one instance inside workdir (created if needed; its files are overwritten, never removed).

    params: extra LKH parameters as {"KEY": value}, e.g. {"RUNS": 1, "TIME_LIMIT": 8.4}.
    initial_tour: (n,) tour written to a file and passed as INITIAL_TOUR_FILE.
    timeout: hard wall-clock limit for the process (seconds); if it is hit, there is no tour.
    Returns a dict: tour, length (float), wall_seconds, preprocessing_seconds, search_seconds,
    run_costs, run_seconds, timed_out, log (LKH's output).
    """
    os.makedirs(workdir, exist_ok=True)
    problem, par = os.path.join(workdir, "problem.tsp"), os.path.join(workdir, "run.par")
    out_tour, init = os.path.join(workdir, "best.tour"), os.path.join(workdir, "initial.tour")
    n = len(coords)
    # The files of workdir are overwritten on every call: a tour file whose (ns) modification time did not
    # change was left by an earlier call, not written by this one.
    old_mtime = os.stat(out_tour).st_mtime_ns if os.path.exists(out_tour) else None
    start = time.perf_counter()        # the clock includes writing the files
    write_problem(problem, coords)
    lines = [f"PROBLEM_FILE = {problem}", f"OUTPUT_TOUR_FILE = {out_tour}"]
    if initial_tour is not None:
        write_tour(init, initial_tour)
        lines.append(f"INITIAL_TOUR_FILE = {init}")
    lines += [f"{k} = {v}" for k, v in (params or {}).items()]
    with open(par, "w") as f:
        f.write("\n".join(lines) + "\n")

    timed_out = False
    try:
        proc = subprocess.run([LKH_BIN, par], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=timeout)
        log = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as err:
        timed_out = True
        log = (err.stdout or b"").decode() if isinstance(err.stdout, bytes) else (err.stdout or "")
    wall = time.perf_counter() - start

    info = parse_output(log)
    tour = None
    if not timed_out and os.path.exists(out_tour) and os.stat(out_tour).st_mtime_ns != old_mtime:
        tour = read_tour(out_tour)
        assert is_valid_tour(tour, n), "LKH wrote an invalid tour"
    return {
        "tour": tour,
        "length": tour_length(coords, tour) if tour is not None else float("nan"),
        "wall_seconds": wall,
        "preprocessing_seconds": info["preprocessing_seconds"],
        "search_seconds": float(sum(info["run_seconds"])),
        "run_costs": info["run_costs"],
        "run_seconds": info["run_seconds"],
        "timed_out": timed_out,
        "log": log,
    }
