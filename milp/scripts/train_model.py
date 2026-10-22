"""Train an ArcGNN on fcnf20, keep the best epoch by validation F1 (p >= 0.5) and evaluate the heatmap
on the test sets.

Writes to results/runs/<run name>/: best.pt, history.json, train.log (from the runner), eval.json
(AP, base rate, precision / recall / F1 per test set) and eval_probs.npz ("<set>" = all arc
probabilities of the set, concatenated in the order of the .npz arcs; the decoders reuse them).

Examples:
    python scripts/train_model.py --model gcn --seed 0
    python scripts/train_model.py --model tagcn --seed 0 --epochs 2 --num-train 500 --run-name quick_tagcn
"""

import argparse
import datetime
import json
import os
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from fcnf_gnn.graph import load_graphs
from fcnf_gnn.models import KINDS, ArcGNN, count_parameters
from fcnf_gnn.train import compute_pos_weight, heatmap_metrics, predict_probs, train_epoch
from fcnf_gnn.utils import git_info, limit_gpu_memory, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train an ArcGNN for the FCNF.")
    parser.add_argument("--model", choices=KINDS, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--tag-k", type=int, default=3, help="TAGCN only: hops K of each layer's filter")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-train", type=int, default=None, help="use only the first N instances")
    parser.add_argument("--train", default="data/fcnf20_train.npz")
    parser.add_argument("--val", default="data/fcnf20_val.npz")
    parser.add_argument("--test", nargs="+",
                        default=["data/fcnf20_test.npz", "data/fcnf50_test.npz", "data/fcnf100_test.npz"])
    parser.add_argument("--run-name", default=None, help="default: {model}_seed{seed}")
    args = parser.parse_args()

    run_dir = os.path.join("results", "runs", args.run_name or f"{args.model}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda")
    fraction = limit_gpu_memory(device)

    train_graphs = load_graphs(args.train, args.num_train)
    val_graphs = load_graphs(args.val)
    pos_weight = compute_pos_weight(train_graphs)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    cpu_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    shuffle_gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, generator=shuffle_gen)

    model = ArcGNN(args.model, args.hidden, args.layers, args.tag_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    commit, dirty = git_info()
    config = {"model": args.model, "hidden": args.hidden, "layers": args.layers,
              **({"tag_k": args.tag_k} if args.model == "tagcn" else {}),
              "train": args.train, "val": args.val, "epochs": args.epochs, "batch_size": args.batch_size,
              "lr": args.lr, "seed": args.seed, "num_train": len(train_graphs), "pos_weight": pos_weight,
              "num_parameters": count_parameters(model), "selection": "val F1 at p >= 0.5"}
    print(f"{run_dir}  |  {args.model}, {args.layers} layers  |  {config['num_parameters']:,} parameters  |  "
          f"train {len(train_graphs)}, validation {len(val_graphs)}  |  pos_weight {pos_weight:.3f}  |  "
          f"GPU ≤ {fraction:.2f} of memory", flush=True)
    print(f"{'epoch':>5}{'lr':>10}{'train loss':>12}{'val loss':>10}{'val AP':>8}{'val F1':>8}{'P':>7}{'R':>7}"
          f"{'time':>7}{'GPU peak':>10}", flush=True)

    history, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        probs, _ = predict_probs(model, val_graphs)
        val = heatmap_metrics(val_graphs, probs, cpu_loss)
        scheduler.step()
        elapsed = time.time() - start
        peak_mb = torch.cuda.max_memory_allocated(device) / 2**20
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()},
                        "seconds": elapsed, "peak_gpu_mb": peak_mb})
        is_best = val["f1"] > best_f1
        if is_best:
            best_f1 = val["f1"]
            torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch, "val": val,
                        "git_commit": commit, "git_dirty": dirty}, os.path.join(run_dir, "best.pt"))
        print(f"{epoch:>5}{lr:>10.2e}{train_loss:>12.4f}{val['loss']:>10.4f}{val['ap']:>8.3f}{val['f1']:>8.3f}"
              f"{val['precision']:>7.3f}{val['recall']:>7.3f}{elapsed:>6.1f}s{peak_mb:>8.0f}MB{'  *' if is_best else ''}",
              flush=True)

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation of the best epoch on the test sets (loaded only now).
    checkpoint = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"\nbest epoch: {checkpoint['epoch']} (val F1 {checkpoint['val']['f1']:.3f}) — test sets:")
    result = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                       "config": config, "best_epoch": checkpoint["epoch"], "best_val": checkpoint["val"]}}
    all_probs = {}
    for path in args.test:
        name = os.path.splitext(os.path.basename(path))[0]
        graphs = load_graphs(path)
        probs, seconds = predict_probs(model, graphs)
        result[name] = {**heatmap_metrics(graphs, probs), "num": len(graphs),
                        "inference_ms_per_inst": 1000 * seconds / len(graphs)}
        all_probs[name] = np.concatenate(probs).astype(np.float32)
        r = result[name]
        print(f"  {name:<14} AP {r['ap']:.3f} (base rate {r['base_rate']:.3f}, per instance {r['ap_inst_mean']:.3f})  "
              f"F1 {r['f1']:.3f}  P {r['precision']:.3f}  R {r['recall']:.3f}  {r['inference_ms_per_inst']:.2f} ms/inst",
              flush=True)
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(os.path.join(run_dir, "eval_probs.npz"), **all_probs)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {run_dir}/eval.json")


if __name__ == "__main__":
    main()
