"""LKH candidate edges from the network's heatmap (phase 3 benchmark, E1 variant "heat").

The trained network runs once on the whole instance (sparse k-NN graph, as in the scale experiment,
scale.knn_graph); for each city we keep the c edges of highest probability p (p of an undirected edge =
mean of both directions, scale.edge_pairs). LKH reads them from a CANDIDATE_FILE; its "alpha" values must
be integers and LKH keeps the MAX_CANDIDATES smallest, so alpha = round(1000 · (1 − p)) (order of p).
"""

import os
import time

import numpy as np
import torch

from tsp_gnn.models import build_model
from tsp_gnn.scale import edge_pairs, knn_graph
from tsp_gnn.utils import limit_gpu_memory


def load_model(run, checkpoint="best.pt", runs_root="results/runs"):
    """(model in eval mode, device) of a trained run; GPU capped at 5.5 GiB if there is one."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        limit_gpu_memory(device)
    ckpt = torch.load(os.path.join(runs_root, run, checkpoint), map_location=device)
    cfg = ckpt["config"]
    assert cfg.get("edge_norm", "none") == "none", "whole-instance inference uses raw distances (phase 1)"
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, device


@torch.no_grad()
def instance_heatmap(model, device, coords, k):
    """Undirected edges (pi, pj, p) of the k-NN graph and the times (s) of the graph and of the network."""
    start = time.perf_counter()
    g = knn_graph(coords, k)
    graph_s = time.perf_counter() - start
    start = time.perf_counter()
    data = g.to(device)
    p = torch.sigmoid(model(data))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    net_s = time.perf_counter() - start
    pi, pj, prob = edge_pairs(g.edge_index, p.cpu(), len(coords))
    return pi, pj, prob, graph_s, net_s


def top_candidates(pi, pj, prob, n, c):
    """(n, c) candidate cities of each city (highest p first) and their integer alphas round(1000 (1 − p))."""
    src = np.concatenate([pi, pj])
    dst = np.concatenate([pj, pi])
    p = np.concatenate([prob, prob])
    order = np.lexsort((-p, src))           # by city, then by decreasing p
    src, dst, p = src[order], dst[order], p[order]
    first = np.searchsorted(src, np.arange(n))
    counts = np.bincount(src, minlength=n)
    assert counts.min() >= c, "every city needs at least c graph edges"
    idx = first[:, None] + np.arange(c)[None, :]
    return dst[idx], np.rint(1000 * (1 - p[idx])).astype(np.int64)


def write_candidates(path, cand, alpha):
    """LKH CANDIDATE_FILE: n, then "city dad count (to alpha)*" per city (1-indexed; dad 0 = none), -1, EOF.

    cand, alpha: (n, c) arrays, or lists of n arrays when the number of candidates varies per city.
    """
    n = len(cand)
    with open(path, "w") as f:
        f.write(f"{n}\n")
        for i in range(n):
            pairs = " ".join(f"{int(t) + 1} {int(a)}" for t, a in zip(cand[i], alpha[i]))
            f.write(f"{i + 1} 0 {len(cand[i])} {pairs}\n")
        f.write("-1\nEOF\n")


def hybrid_candidates(coords, heat_cand, nn_cand):
    """Union of the network's candidates and the nearest neighbours of each city (E1 variant "nnheat5").

    heat_cand, nn_cand: (n, c) arrays. Returns two lists of n arrays (up to 2c candidates per city, no
    duplicates), ordered by distance; the alphas are the ranks 0, 1, ... in that order (with MAX_CANDIDATES
    >= 2c nothing is trimmed, the alphas only set the order in which LKH tries the candidates).
    """
    cands, alphas = [], []
    for i in range(len(coords)):
        union = np.union1d(heat_cand[i], nn_cand[i])
        union = union[union != i]
        dist = np.sqrt(((coords[union] - coords[i]) ** 2).sum(axis=1))
        cands.append(union[np.argsort(dist, kind="stable")])
        alphas.append(np.arange(len(union), dtype=np.int64))
    return cands, alphas
