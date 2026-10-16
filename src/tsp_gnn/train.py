"""Treino e avaliação das EdgeGNN.

Treino: classificação binária das arestas com BCEWithLogitsLoss(pos_weight).
Avaliação: probabilidades -> decode do passo 3 (gulosa, com ou sem 2-opt) -> gap face ao LKH.
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
from tsp_gnn.utils import git_info


def compute_pos_weight(graphs):
    """Nº de arestas negativas / nº de positivas (dá mais peso à classe rara, a positiva)."""
    pos = sum(g.y.sum().item() for g in graphs)
    total = sum(g.num_edges for g in graphs)
    return (total - pos) / pos


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Uma época de treino. Devolve a perda média por aresta."""
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
    """Perda média por aresta, sem treinar (modo eval: BatchNorm usa as estatísticas guardadas)."""
    model.eval()
    total_loss, total_edges = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        total_loss += loss_fn(model(batch), batch.y).item() * batch.num_edges
        total_edges += batch.num_edges
    return total_loss / total_edges


@torch.no_grad()
def predict_probs(model, graphs, batch_size=256):
    """Probabilidades das arestas de cada grafo (lista de tensores no CPU) e o tempo gasto (s)."""
    model.eval()
    device = next(model.parameters()).device
    probs = []
    start = time.perf_counter()
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        p = torch.sigmoid(model(batch)).cpu()
        # O batch junta as arestas de todos os grafos, pela ordem: separamo-las de volta.
        edges_per_graph = torch.bincount(batch.batch[batch.edge_index[0]], minlength=batch.num_graphs)
        probs.extend(p.split(edges_per_graph.tolist()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return probs, time.perf_counter() - start


def decode_all(graphs, probs, use_2opt):
    """Decodifica todos os grafos. Devolve o array de gaps (%) e o tempo gasto (s).

    decode() chama verify_tour, que lança um erro se algum tour for inválido.
    """
    start = time.perf_counter()
    gaps = np.array([decode(g, p, use_2opt)["gap"] for g, p in zip(graphs, probs)])
    return gaps, time.perf_counter() - start


def evaluate_model(model, datasets):
    """Avalia o modelo em vários conjuntos ({nome: grafos}).

    Devolve (resumo por conjunto e método, gaps por instância).
    """
    summary, per_instance = {}, {}
    for name, graphs in datasets.items():
        probs, t_inf = predict_probs(model, graphs)
        opt_len = np.array([float(g.opt_len) for g in graphs])
        entry = {"n": graphs[0].num_nodes, "num": len(graphs), "lkh_mean_len": float(opt_len.mean()),
                 "inference_ms_per_inst": 1000 * t_inf / len(graphs)}
        for method, use_2opt in [("greedy", False), ("greedy+2opt", True)]:
            gaps, t_dec = decode_all(graphs, probs, use_2opt)
            entry[method] = {
                "mean_len": float((opt_len * (1 + gaps / 100)).mean()),
                "gap_mean": float(gaps.mean()), "gap_std": float(gaps.std(ddof=1)),
                "gap_max": float(gaps.max()),
                "ms_per_inst": 1000 * (t_inf + t_dec) / len(graphs),  # rede + decodificação
                "all_valid": len(gaps) == len(graphs),  # decode() teria lançado erro num tour inválido
            }
            per_instance[f"{name}__{method}"] = gaps
        summary[name] = entry
    return summary, per_instance


def evaluate_and_save(model, checkpoint, test_paths, run_dir):
    """Avalia o modelo nos conjuntos de teste e grava eval.json + eval_per_instance.npz."""
    datasets = {os.path.splitext(os.path.basename(p))[0]: load_graphs(p, checkpoint["config"]["k"])
                for p in test_paths}
    summary, per_instance = evaluate_model(model, datasets)

    commit, dirty = git_info()
    result = {"meta": {"git_commit": commit, "git_dirty": dirty,
                       "date": datetime.date.today().isoformat(),
                       "train_git_commit": checkpoint["git_commit"],
                       "config": checkpoint["config"],
                       "best_epoch": checkpoint["epoch"], "best_val_gap": checkpoint["val_gap"]},
              **summary}
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(os.path.join(run_dir, "eval_per_instance.npz"), **per_instance)

    print(f"{'conjunto':<12}{'método':<14}{'gap méd':>9}{'gap máx':>9}{'ms/inst':>9}  tours válidos")
    for name, entry in summary.items():
        for method in ["greedy", "greedy+2opt"]:
            m = entry[method]
            print(f"{name:<12}{method:<14}{m['gap_mean']:>8.2f}%{m['gap_max']:>8.2f}%"
                  f"{m['ms_per_inst']:>9.2f}  {entry['num'] if m['all_valid'] else 'ERRO'}/{entry['num']}")
    print(f"commit {commit}{' (com alterações por gravar)' if dirty else ''}  |  "
          f"gravado em {run_dir}/eval.json e eval_per_instance.npz")
    return result
