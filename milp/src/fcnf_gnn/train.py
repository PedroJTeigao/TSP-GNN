"""Training and evaluation of the arc classifier (adapted from tsp_gnn.train).

Training: BCEWithLogitsLoss(pos_weight = negatives / positives of the training arcs).
Evaluation of the heatmap: average precision (AP) for "arc open" against the base rate, and
precision / recall / F1 at p >= 0.5.
"""

import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader


def compute_pos_weight(graphs):
    pos = sum(g.y.sum().item() for g in graphs)
    total = sum(len(g.y) for g in graphs)
    return (total - pos) / pos


def train_epoch(model, loader, optimizer, loss_fn, device):
    """One epoch. Returns the mean loss per arc."""
    model.train()
    total_loss, total_arcs = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(batch), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch.y)
        total_arcs += len(batch.y)
    return total_loss / total_arcs


@torch.no_grad()
def predict_probs(model, graphs, batch_size=256):
    """Arc probabilities of each graph (list of numpy arrays) and the time taken (s)."""
    model.eval()
    device = next(model.parameters()).device
    probs = []
    start = time.perf_counter()
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        p = torch.sigmoid(model(batch)).cpu()
        arcs_per_graph = torch.bincount(batch.batch[batch.arc_index[0]], minlength=batch.num_graphs)
        probs.extend(t.numpy() for t in p.split(arcs_per_graph.tolist()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return probs, time.perf_counter() - start


def heatmap_metrics(graphs, probs, loss_fn=None):
    """AP (all arcs pooled and mean per instance), base rate, precision / recall / F1 at 0.5."""
    y = np.concatenate([g.y.numpy() for g in graphs])
    p = np.concatenate(probs)
    pred = p >= 0.5
    tp = float((pred & (y == 1)).sum())
    precision = float(tp / max(pred.sum(), 1))
    recall = float(tp / max((y == 1).sum(), 1))
    per_inst = [average_precision_score(g.y.numpy(), q) for g, q in zip(graphs, probs) if 0 < g.y.sum() < len(g.y)]
    out = {"ap": float(average_precision_score(y, p)), "ap_inst_mean": float(np.mean(per_inst)),
           "base_rate": float(y.mean()), "precision": precision, "recall": recall,
           "f1": float(2 * precision * recall / max(precision + recall, 1e-12)), "pred_open_frac": float(pred.mean())}
    if loss_fn is not None:
        logits = torch.logit(torch.as_tensor(p).clamp(1e-7, 1 - 1e-7))
        out["loss"] = float(loss_fn(logits, torch.as_tensor(y)).item())
    return out
