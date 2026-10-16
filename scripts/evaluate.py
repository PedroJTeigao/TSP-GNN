"""Avalia um modelo já treinado (best.pt) nos conjuntos de teste, sem voltar a treinar.

Exemplo:
    python scripts/evaluate.py --run-dir results/runs/gated_seed0
"""

import argparse
import os

import torch

from tsp_gnn.models import EdgeGNN
from tsp_gnn.train import evaluate_and_save


def main():
    parser = argparse.ArgumentParser(description="Avaliação de uma EdgeGNN treinada.")
    parser.add_argument("--run-dir", required=True, help="pasta com o best.pt")
    parser.add_argument("--test", nargs="+", default=["data/tsp20_test.npz", "data/tsp50_test.npz"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(os.path.join(args.run_dir, "best.pt"), map_location=device)
    cfg = checkpoint["config"]
    model = EdgeGNN(cfg["model"], cfg["hidden"], cfg["layers"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"{args.run_dir}: {cfg['model']}, melhor época {checkpoint['epoch']} "
          f"(gap val {checkpoint['val_gap']:.2f}%)")
    evaluate_and_save(model, checkpoint, args.test, args.run_dir)


if __name__ == "__main__":
    main()
