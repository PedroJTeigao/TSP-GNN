"""Training and evaluation of the EdgeGNNs.

Training: binary edge classification with BCEWithLogitsLoss(pos_weight).
Evaluation: probabilities -> decode from step 3 (greedy, with or without 2-opt) -> gap to LKH.
"""

import datetime
import json
import os
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tsp_gnn.decode import decode
from tsp_gnn.graph import load_graphs
from tsp_gnn.models import with_batch_stats
from tsp_gnn.utils import git_info


def compute_pos_weight(graphs):
    """No. of negative edges / no. of positive edges (gives more weight to the rare, positive class)."""
    pos = sum(g.y.sum().item() for g in graphs)
    total = sum(g.num_edges for g in graphs)
    return (total - pos) / pos


def train_epoch(model, loader, optimizer, loss_fn, device):
    """One training epoch. Returns the mean loss per edge."""
    model.train()
    total_loss, total_edges = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(batch), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_edges
        total_edges += batch.num_edges
    return total_loss / total_edges


@torch.no_grad()
def eval_loss(model, loader, loss_fn, device):
    """Mean loss per edge, without training (eval mode: BatchNorm uses its stored statistics)."""
    model.eval()
    total_loss, total_edges = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        total_loss += loss_fn(model(batch), batch.y).item() * batch.num_edges
        total_edges += batch.num_edges
    return total_loss / total_edges


@torch.no_grad()
def predict_probs(model, graphs, batch_size=256):
    """Edge probabilities of each graph (list of CPU tensors) and the time taken (s)."""
    model.eval()
    device = next(model.parameters()).device
    probs = []
    start = time.perf_counter()
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        p = torch.sigmoid(model(batch)).cpu()
        # The batch concatenates the edges of all graphs, in order: split them back.
        edges_per_graph = torch.bincount(batch.batch[batch.edge_index[0]], minlength=batch.num_graphs)
        probs.extend(p.split(edges_per_graph.tolist()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return probs, time.perf_counter() - start


def decode_all(graphs, probs, use_2opt):
    """Decode all graphs. Returns the array of gaps (%) and the time taken (s).

    decode() calls verify_tour, which raises an error if any tour is invalid.
    """
    start = time.perf_counter()
    gaps = np.array([decode(g, p, use_2opt)["gap"] for g, p in zip(graphs, probs)])
    return gaps, time.perf_counter() - start


def evaluate_model(model, datasets, batch_size=256):
    """Evaluate the model on several datasets ({name: graphs}).

    Returns (summary per dataset and method, per-instance gaps).
    """
    summary, per_instance = {}, {}
    for name, graphs in datasets.items():
        probs, t_inf = predict_probs(model, graphs, batch_size)
        opt_len = np.array([float(g.opt_len) for g in graphs])
        entry = {"n": graphs[0].num_nodes, "num": len(graphs), "lkh_mean_len": float(opt_len.mean()),
                 "inference_ms_per_inst": 1000 * t_inf / len(graphs)}
        for method, use_2opt in [("greedy", False), ("greedy+2opt", True)]:
            gaps, t_dec = decode_all(graphs, probs, use_2opt)
            entry[method] = {
                "mean_len": float((opt_len * (1 + gaps / 100)).mean()),
                "gap_mean": float(gaps.mean()), "gap_std": float(gaps.std(ddof=1)),
                "gap_max": float(gaps.max()),
                "ms_per_inst": 1000 * (t_inf + t_dec) / len(graphs),  # network + decoding
                "all_valid": len(gaps) == len(graphs),  # decode() would have raised on an invalid tour
            }
            per_instance[f"{name}__{method}"] = gaps
        summary[name] = entry
    return summary, per_instance


def evaluate_and_save(model, checkpoint, test_paths, run_dir, k_frac=None, batch_size=256,
                      bn_batch_stats=False):
    """Evaluate the model on the test sets and save eval.json + eval_per_instance.npz.

    By default the graphs are built as in training (the checkpoint's k, k_frac and edge_norm).
    k_frac overrides the graph size used at evaluation; bn_batch_stats evaluates a copy of the
    model whose BatchNorms use the statistics of each batch (see models.with_batch_stats).
    """
    cfg = checkpoint["config"]
    edge_norm = cfg.get("edge_norm", "none")  # phase 1 checkpoints have no edge_norm: they used "none"
    k_frac = k_frac if k_frac is not None else cfg.get("k_frac")
    datasets = {os.path.splitext(os.path.basename(p))[0]: load_graphs(p, cfg["k"], edge_norm, k_frac)
                for p in test_paths}
    if bn_batch_stats:
        model = with_batch_stats(model)
    summary, per_instance = evaluate_model(model, datasets, batch_size)

    commit, dirty = git_info()
    result = {"meta": {"git_commit": commit, "git_dirty": dirty,
                       "date": datetime.date.today().isoformat(),
                       "train_git_commit": checkpoint["git_commit"],
                       "config": checkpoint["config"],
                       "best_epoch": checkpoint["epoch"], "best_val_gap": checkpoint["val_gap"],
                       "eval_options": {"k": cfg["k"], "k_frac": k_frac, "edge_norm": edge_norm,
                                        "bn_batch_stats": bn_batch_stats, "batch_size": batch_size}},
              **summary}
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(os.path.join(run_dir, "eval_per_instance.npz"), **per_instance)

    print(f"{'dataset':<12}{'method':<14}{'mean gap':>9}{'max gap':>9}{'ms/inst':>9}  valid tours")
    for name, entry in summary.items():
        for method in ["greedy", "greedy+2opt"]:
            m = entry[method]
            print(f"{name:<12}{method:<14}{m['gap_mean']:>8.2f}%{m['gap_max']:>8.2f}%"
                  f"{m['ms_per_inst']:>9.2f}  {entry['num'] if m['all_valid'] else 'ERROR'}/{entry['num']}")
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  "
          f"saved to {run_dir}/eval.json and eval_per_instance.npz")
    return result
