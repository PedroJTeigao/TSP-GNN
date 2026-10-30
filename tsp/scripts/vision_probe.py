"""Phase 5 vision probe: can a vision model (Gemma 4 on the private Ollama, port 11435) point at the grid cells of a
tour image where LKH will gain? Steps (--step):

gpu      print the GPU memory (MiB) used by processes other than the private Ollama (ollama_pc) and ours (started
         from this project)
images   first --num Fu-1k instances: greedy + 2-opt tour, tour-edge probabilities of the TAGCN (seed 0, windows
         w = 20 as in the triage), images (a) tour only and (b) tour coloured by p, with the numbered 4 × 4 grid;
         per-cell labels from the learned-triage window log (Fu-1k, seed 0). Writes results/phase5/vision_probe/
         {images/<a|b>/<i>.png, cells.npz, images.json}
tune     load the model with decreasing num_gpu until the ollama_pc runner uses <= --gpu-budget MiB; records the
         layers offloaded (from the server log) and three call latencies. Writes .../tune.json
ask      for each image type, repetition (temperature 0.5, seed = repetition) and instance: the ranking of the cells
         as JSON; latency, raw answer. Writes .../answers.jsonl (resumes)
policy   LKH on the windows of the top-4 cells (vision a / b per repetition, disagreement, oracle, 3 random draws)
         and P2 top 20 % by disagreement (batched scorer), ≤ 3 passes, gap and time. Writes .../policy.npz

Example:
    python scripts/vision_probe.py --step images
"""

import argparse
import base64
import datetime
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np

ROOT = "results/phase5/vision_probe"
DATA = "results/phase3_benchmark/data/tsp1000_test_concorde.npz"
LOG = "results/phase3_learned_triage/log/tagcn_k3_seed0/tsp1000_test_concorde.npz"
FEATS = "results/phase3_learned_triage/features/tagcn_k3_seed0/tsp1000_test_concorde.npz"
HOST = "127.0.0.1:11435"
SERVER_LOG = "results/phase5/ollama_pc.log"
PROMPT = {
    "a": "The image shows a travelling-salesman tour through 1000 cities (black dots joined by black lines) "
         "in the unit square, with a 4 x 4 grid whose cells are numbered 1 to 16 (blue numbers, row by row from the "
         "top left). A local optimiser will try to shorten the tour piece by piece. Rank the grid cells from the "
         "one where the tour can MOST likely be improved (crossings, long detours, zig-zags, edges that jump over "
         "nearby cities) to the least likely. Answer only with JSON: {\"cells\": [16 distinct cell numbers, most "
         "likely first]}.",
    "b": "The image shows a travelling-salesman tour through 1000 cities in the unit square, with a 4 x 4 grid whose "
         "cells are numbered 1 to 16 (blue numbers, row by row from the top left). Each tour edge is coloured by "
         "how confident a neural network is that the edge belongs to a good tour: green = confident, red = not "
         "confident. A local optimiser will try to shorten the tour piece by piece. Rank the grid cells from the one "
         "where the tour can MOST likely be improved to the least likely. Answer only with JSON: {\"cells\": [16 "
         "distinct cell numbers, most likely first]}.",
}


def gpu_others():
    """MiB of GPU memory used by processes other than ollama_pc (binary under /scratch/pcanaste/tools/ollama) and our
    .venv python processes."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    total = 0
    for line in out.strip().splitlines():
        pid, mem = [x.strip() for x in line.split(",")]
        try:
            exe = os.path.realpath(f"/proc/{pid}/exe")
            cmd = open(f"/proc/{pid}/cmdline").read()
            cwd = os.path.realpath(f"/proc/{pid}/cwd")
        except OSError:
            exe, cmd, cwd = "", "", ""
        # ours: the private Ollama, or a process started from this project (the .venv python resolves to the
        # system binary, so the working directory is what identifies it)
        if "/scratch/pcanaste/tools/ollama" in exe + cmd or cwd.startswith("/scratch/pcanaste/projeto/gnn-tsp"):
            continue
        total += int(mem)
    return total


def ollama_pc_gpu():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    total = 0
    for line in out.strip().splitlines():
        pid, mem = [x.strip() for x in line.split(",")]
        try:
            if "/scratch/pcanaste/tools/ollama" in os.path.realpath(f"/proc/{pid}/exe") + open(f"/proc/{pid}/cmdline").read():
                total += int(mem)
        except OSError:
            pass
    return total


def chat(model, prompt, image_path, options, timeout=900):
    body = {"model": model, "stream": False, "format": "json", "keep_alive": "30m", "options": options,
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(open(image_path, "rb").read()).decode()]}]}
    req = urllib.request.Request(f"http://{HOST}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return None, time.perf_counter() - start, {"error": e.read().decode(errors="replace")}
    return out["message"]["content"], time.perf_counter() - start, out


def offloaded_layers():
    """Last 'offloaded X/Y layers to GPU' line of the ollama_pc server log."""
    if not os.path.exists(SERVER_LOG):
        return None
    hits = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", open(SERVER_LOG, errors="replace").read())
    return [int(hits[-1][0]), int(hits[-1][1])] if hits else None


def step_images(args):
    import torch

    from tsp_gnn import heatcand
    from tsp_gnn import vision_probe as V
    from tsp_gnn.data import load_dataset
    from tsp_gnn.learned_triage import labels
    from tsp_gnn.window import initial_tour

    torch.set_num_threads(2)
    model, device = heatcand.load_model("tagcn_k3_seed0")
    d = load_dataset(DATA)
    log, feats = dict(np.load(LOG)), dict(np.load(FEATS))
    n = d["coords"].shape[1]
    k = min(n - 1, round(0.5 * n), 100)
    improves, gain_rel = labels(log["cur_len"], log["lkh_len"])
    gain = log["cur_len"] - log["lkh_len"]
    rows = {key: [] for key in ("label", "gain", "disagreement", "windows")}
    secs = {"a": [], "b": [], "edge_p": []}
    for kind in "ab":
        os.makedirs(os.path.join(ROOT, "images", kind), exist_ok=True)
    for i in range(args.num):
        coords = d["coords"][i].astype(np.float32).astype(np.float64)
        tour = initial_tour(coords, k)
        m = log["instance"].astype(int) == i
        cells = V.window_cells(coords, tour, log["start"][m], log["size"][m])
        st = V.cell_stats(cells, improves[m], gain[m], feats["disagreement"][m])
        for key in rows:
            rows[key].append(st[key])
        t0 = time.perf_counter()
        V.render(os.path.join(ROOT, "images", "a", f"{i}.png"), coords, tour, args.size)
        t1 = time.perf_counter()
        edge_p = V.tour_edge_p(model, device, coords, tour)
        t2 = time.perf_counter()
        V.render(os.path.join(ROOT, "images", "b", f"{i}.png"), coords, tour, args.size, edge_p)
        t3 = time.perf_counter()
        secs["a"].append(t1 - t0)
        secs["edge_p"].append(t2 - t1)
        secs["b"].append(t3 - t2)
        print(f"  instance {i}: {int(st['windows'].sum())} windows, {int((st['windows'] > 0).sum())} cells with "
              f"windows", flush=True)
    np.savez_compressed(os.path.join(ROOT, "cells.npz"), **{k2: np.array(v) for k2, v in rows.items()})
    json.dump({"num": args.num, "size": args.size, "seconds_mean": {k2: float(np.mean(v)) for k2, v in secs.items()},
               "windows_per_cell": float(np.mean(np.array(rows["windows"]))),
               "date": datetime.datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(ROOT, "images.json"), "w"), indent=2)


def step_tune(args):
    image = os.path.join(ROOT, "images", "b", "0.png")
    trials = []
    chosen = None
    for num_gpu in args.num_gpu_list:
        content, sec, raw = chat(args.model, PROMPT["b"], image, {"num_gpu": num_gpu, "num_ctx": 8192,
                                                                  "temperature": 0.5, "seed": 0})
        time.sleep(2)
        mem = ollama_pc_gpu()
        trial = {"num_gpu": num_gpu, "ollama_pc_gpu_mib": mem, "seconds": sec, "ok": content is not None,
                 "layers": offloaded_layers(), "error": raw.get("error") if content is None else None}
        trials.append(trial)
        print(trial, flush=True)
        if content is not None and mem <= args.gpu_budget:
            chosen = num_gpu
            break
    lat = []
    if chosen is not None:
        for s in range(3):
            content, sec, raw = chat(args.model, PROMPT["b"], image, {"num_gpu": chosen, "num_ctx": 8192,
                                                                      "temperature": 0.5, "seed": s})
            lat.append({"seconds": sec, "ok": content is not None, "eval_count": raw.get("eval_count"),
                        "prompt_eval_count": raw.get("prompt_eval_count")})
    out = {"model": args.model, "budget_mib": args.gpu_budget, "trials": trials, "chosen_num_gpu": chosen,
           "layers": offloaded_layers(), "latency": lat, "gpu_others_mib": gpu_others(),
           "date": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(out, open(os.path.join(ROOT, f"tune_{args.model.replace(':', '_')}.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    if chosen is None:
        raise SystemExit("no num_gpu fits the budget")


def step_ask(args):
    from tsp_gnn.llm import extract_json
    from tsp_gnn import vision_probe as V

    tune = json.load(open(os.path.join(ROOT, f"tune_{args.model.replace(':', '_')}.json")))
    num_gpu = tune["chosen_num_gpu"]
    path = os.path.join(ROOT, "answers.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(e["model"], e["kind"], e["rep"], e["instance"]) for e in map(json.loads, open(path))}
    with open(path, "a") as f:
        for rep in range(args.reps):
            for kind in "ab":
                for i in range(args.num):
                    if (args.model, kind, rep, i) in done:
                        continue
                    content, sec, raw = chat(args.model, PROMPT[kind], os.path.join(ROOT, "images", kind, f"{i}.png"),
                                             {"num_gpu": num_gpu, "num_ctx": 8192, "temperature": 0.5, "seed": rep})
                    ranking = V.parse_ranking(extract_json(content)) if content is not None else None
                    f.write(json.dumps({"model": args.model, "kind": kind, "rep": rep, "instance": i,
                                        "ranking": ranking, "seconds": sec, "raw": content,
                                        "error": raw.get("error") if content is None else None,
                                        "eval_count": raw.get("eval_count"),
                                        "prompt_eval_count": raw.get("prompt_eval_count"), "num_gpu": num_gpu}) + "\n")
                    f.flush()
                print(f"  rep {rep} image {kind}: done", flush=True)


def step_policy(args):
    import torch

    from tsp_gnn import heatcand
    from tsp_gnn import vision_probe as V
    from tsp_gnn.data import load_dataset, tour_length
    from tsp_gnn.decode import optimality_gap
    from tsp_gnn.fast_triage import WindowBatchScorer
    from tsp_gnn.triage import ranked_refine
    from tsp_gnn.window import initial_tour

    torch.set_num_threads(2)
    model, device = heatcand.load_model("tagcn_k3_seed0")
    d = load_dataset(DATA)
    n = d["coords"].shape[1]
    k = min(n - 1, round(0.5 * n), 100)
    cells = dict(np.load(os.path.join(ROOT, "cells.npz")))
    answers = [e for e in map(json.loads, open(os.path.join(ROOT, "answers.jsonl"))) if e["model"] == args.model]
    out = {}
    add = lambda key, v: out.setdefault(key, []).append(v)
    for i in range(args.num):
        coords = d["coords"][i].astype(np.float32).astype(np.float64)
        opt = float(d["lengths"][i])
        start = time.perf_counter()
        t0 = initial_tour(coords, k)
        init_s = time.perf_counter() - start
        g0 = optimality_gap(tour_length(coords, t0), opt)
        add("initial__gap", g0)
        add("initial__s", init_s)
        choices = {"disagreement": V.top_cells(np.nan_to_num(cells["disagreement"][i], nan=-1.0)),
                   "oracle": V.top_cells(np.nan_to_num(cells["gain"][i], nan=-1.0))}
        for r in range(3):
            choices[f"random_r{r}"] = [int(c) for c in np.random.default_rng([r, i]).choice(16, 4, replace=False) + 1]
        for e in answers:
            if e["instance"] == i and e["ranking"]:
                rank = V.ranking_scores(e["ranking"], np.random.default_rng([e["rep"], i]))
                choices[f"vision_{e['kind']}_r{e['rep']}"] = V.top_cells(rank)
        for name, chosen in choices.items():
            start = time.perf_counter()
            tour, st = V.cell_refine(coords, t0, 20, chosen)
            sec = time.perf_counter() - start
            add(f"{name}__gap", optimality_gap(tour_length(coords, tour), opt))
            add(f"{name}__s", sec)
            add(f"{name}__lkh_frac", st["to_lkh"] / st["windows"])
        start = time.perf_counter()
        tour, st = ranked_refine(coords, t0, 20, "scorer", 0.2, scorer=WindowBatchScorer(model, device))
        add("p2_top20__s", time.perf_counter() - start)
        add("p2_top20__gap", optimality_gap(tour_length(coords, tour), opt))
        add("p2_top20__lkh_frac", st["to_lkh"] / st["windows"])
        print(f"  instance {i}: done", flush=True)
    keys = set(out)
    for key in keys:                              # policies missing for some instances (no answer) -> nan
        if len(out[key]) != args.num:
            out[key] = [np.nan] * args.num
    np.savez_compressed(os.path.join(ROOT, f"policy_{args.model.replace(':', '_')}.npz"),
                        **{key: np.array(v, dtype=float) for key, v in out.items()})


def main():
    parser = argparse.ArgumentParser(description="Phase 5 vision probe.")
    parser.add_argument("--step", choices=("gpu", "images", "tune", "ask", "policy"), required=True)
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--model", default="gemma4:e4b-it-qat")
    parser.add_argument("--gpu-budget", type=int, default=5376, help="MiB for the model (5.5 GiB − 0.25 for the GNN)")
    parser.add_argument("--num-gpu-list", type=int, nargs="+", default=[999, 36, 32, 28, 24, 20, 16, 12, 8])
    args = parser.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    if args.step == "gpu":
        print(gpu_others())
    else:
        {"images": step_images, "tune": step_tune, "ask": step_ask, "policy": step_policy}[args.step](args)


if __name__ == "__main__":
    main()
