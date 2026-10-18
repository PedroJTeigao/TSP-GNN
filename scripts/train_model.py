"""Train an EdgeGNN, keep the best model (by validation gap) and evaluate it on the test sets.

Examples:
    python scripts/train_model.py --model gated
    python scripts/train_model.py --model gcn --epochs 2 --num-train 1000 --run-name quick_gcn
    python scripts/train_model.py --model gated --edge-norm knn_mean --run-name gated_knnmean_seed0
    python scripts/train_model.py --model gated --norm layernorm --run-name gated_layernorm_seed0
"""

import argparse
import json
import os
import time

import torch
from torch_geometric.loader import DataLoader

from tsp_gnn.graph import EDGE_NORMS, load_graphs
from tsp_gnn.models import NORMS, EdgeGNN, count_parameters
from tsp_gnn.train import (compute_pos_weight, decode_all, eval_loss, evaluate_and_save,
                           predict_probs, train_epoch)
from tsp_gnn.utils import git_info, limit_gpu_memory, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train an EdgeGNN for the TSP.")
    parser.add_argument("--model", choices=["gcn", "gat", "gated"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--k", type=int, default=10, help="neighbours in the k-NN graph")
    parser.add_argument("--edge-norm", choices=EDGE_NORMS, default="none",
                        help="rescaling of the edge lengths (none = phase 1)")
    parser.add_argument("--k-frac", type=float, default=None,
                        help="k = round(k_frac × n) instead of the fixed --k (default: fixed k, as in phase 1)")
    parser.add_argument("--norm", choices=NORMS, default="batchnorm", help="normalization layer of the encoder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-train", type=int, default=None, help="use only the first N instances")
    parser.add_argument("--train", default="data/tsp20_train.npz")
    parser.add_argument("--val", default="data/tsp20_val.npz")
    parser.add_argument("--test", nargs="+", default=["data/tsp20_test.npz", "data/tsp50_test.npz"])
    parser.add_argument("--run-name", default=None, help="default: {model}_seed{seed}")
    args = parser.parse_args()

    run_dir = os.path.join("results", "runs", args.run_name or f"{args.model}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda")
    fraction = limit_gpu_memory(device)

    # Data: training and validation only. The test sets are loaded only after training ends.
    train_graphs = load_graphs(args.train, args.k, args.edge_norm, args.k_frac)[:args.num_train]
    val_graphs = load_graphs(args.val, args.k, args.edge_norm, args.k_frac)
    pos_weight = compute_pos_weight(train_graphs)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    shuffle_gen = torch.Generator().manual_seed(args.seed)  # reproducible batch order
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, generator=shuffle_gen)
    val_loader = DataLoader(val_graphs, batch_size=256, shuffle=False)

    model = EdgeGNN(args.model, args.hidden, args.layers, norm=args.norm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    commit, dirty = git_info()
    config = {"model": args.model, "hidden": args.hidden, "layers": args.layers, "k": args.k,
              "edge_norm": args.edge_norm, "k_frac": args.k_frac, "norm": args.norm,
              "train": args.train, "val": args.val,
              "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
              "num_train": len(train_graphs), "pos_weight": pos_weight}
    print(f"{run_dir}  |  edge_norm {args.edge_norm}, norm {args.norm}, k_frac {args.k_frac}  |  {count_parameters(model):,} parameters  |  train {len(train_graphs)}, "
          f"validation {len(val_graphs)}  |  pos_weight {pos_weight:.3f}  |  GPU ≤ {fraction:.2f} of memory")
    print(f"{'epoch':>5}{'lr':>10}{'train loss':>14}{'val loss':>11}{'val gap':>9}{'time':>8}{'GPU peak':>10}")

    history, best_gap = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_loss(model, val_loader, loss_fn, device)
        probs, _ = predict_probs(model, val_graphs)
        val_gap = float(decode_all(val_graphs, probs, use_2opt=False)[0].mean())  # greedy, no 2-opt
        scheduler.step()
        elapsed = time.time() - start
        peak_mb = torch.cuda.max_memory_allocated(device) / 2**20

        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val_loss,
                        "val_gap_greedy": val_gap, "seconds": elapsed, "peak_gpu_mb": peak_mb})
        is_best = val_gap < best_gap
        if is_best:  # model selection uses the validation set only
            best_gap = val_gap
            torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch,
                        "val_gap": val_gap, "git_commit": commit, "git_dirty": dirty},
                       os.path.join(run_dir, "best.pt"))
        print(f"{epoch:>5}{lr:>10.2e}{train_loss:>14.4f}{val_loss:>11.4f}{val_gap:>8.2f}%"
              f"{elapsed:>7.1f}s{peak_mb:>8.0f}MB{'  *' if is_best else ''}")

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation of the best model (now, and only now, on the test sets).
    checkpoint = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"\nbest epoch: {checkpoint['epoch']} (val gap {checkpoint['val_gap']:.2f}%) — test evaluation:")
    evaluate_and_save(model, checkpoint, args.test, run_dir)


if __name__ == "__main__":
    main()
