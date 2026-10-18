"""Evaluate an already trained model (best.pt) on the test sets, without retraining.

Examples:
    python scripts/evaluate.py --run-dir results/runs/gated_seed0
    python scripts/evaluate.py --run-dir results/runs/gated_seed0 --test data/tsp100_test.npz \
        --out-dir results/phase2_exp1/phase1_tsp100/gated_seed0   # leaves the original eval.json untouched
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        limit_gpu_memory(device)  # shared GPU: same 5.5 GiB cap as training
    checkpoint = torch.load(os.path.join(args.run_dir, "best.pt"), map_location=device)
    cfg = checkpoint["config"]
    model = EdgeGNN(cfg["model"], cfg["hidden"], cfg["layers"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"{args.run_dir}: {cfg['model']}, edge_norm {cfg.get('edge_norm', 'none')}, "
          f"best epoch {checkpoint['epoch']} (val gap {checkpoint['val_gap']:.2f}%)")
    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)
    evaluate_and_save(model, checkpoint, args.test, out_dir)


if __name__ == "__main__":
    main()
