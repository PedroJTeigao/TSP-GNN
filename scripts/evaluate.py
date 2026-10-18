"""Evaluate an already trained model (best.pt) on the test sets, without retraining.

Examples:
    python scripts/evaluate.py --run-dir results/runs/gated_seed0
    python scripts/evaluate.py --run-dir results/runs/gated_seed0 --test data/tsp100_test.npz \
        --out-dir results/phase2_exp1/phase1_tsp100/gated_seed0   # leaves the original eval.json untouched
    python scripts/evaluate.py --run-dir results/runs/gated_seed0 --k-frac 0.5 --out-dir <new folder>
    python scripts/evaluate.py --run-dir results/runs/gated_seed0 --bn-batch-stats --out-dir <new folder>
"""

import argparse
import os

import torch

from tsp_gnn.models import EdgeGNN
from tsp_gnn.train import evaluate_and_save
from tsp_gnn.utils import limit_gpu_memory


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained EdgeGNN.")
    parser.add_argument("--run-dir", required=True, help="folder containing best.pt")
    parser.add_argument("--test", nargs="+", default=["data/tsp20_test.npz", "data/tsp50_test.npz"])
    parser.add_argument("--out-dir", default=None,
                        help="where to write eval.json and eval_per_instance.npz (default: --run-dir)")
    parser.add_argument("--k-frac", type=float, default=None,
                        help="evaluate on graphs with k = round(k_frac × n) (default: as in training)")
    parser.add_argument("--bn-batch-stats", action="store_true",
                        help="BatchNorm uses the statistics of each batch instead of the stored ones")
    parser.add_argument("--batch-size", type=int, default=256, help="graphs per forward pass")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        limit_gpu_memory(device)  # shared GPU: same 5.5 GiB cap as training
    checkpoint = torch.load(os.path.join(args.run_dir, "best.pt"), map_location=device)
    cfg = checkpoint["config"]
    model = EdgeGNN(cfg["model"], cfg["hidden"], cfg["layers"], norm=cfg.get("norm", "batchnorm")).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"{args.run_dir}: {cfg['model']}, edge_norm {cfg.get('edge_norm', 'none')}, "
          f"norm {cfg.get('norm', 'batchnorm')}, best epoch {checkpoint['epoch']} "
          f"(val gap {checkpoint['val_gap']:.2f}%)  |  evaluation: k_frac {args.k_frac}, "
          f"bn_batch_stats {args.bn_batch_stats}, batch size {args.batch_size}")
    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)
    evaluate_and_save(model, checkpoint, args.test, out_dir, k_frac=args.k_frac,
                      batch_size=args.batch_size, bn_batch_stats=args.bn_batch_stats)


if __name__ == "__main__":
    main()
