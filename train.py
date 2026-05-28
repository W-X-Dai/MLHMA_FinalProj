"""
Final model training script.

Trains a chosen model on the full 106-sample dataset.
80/20 random split is used for early stopping only — it is NOT a
generalization estimate. For held-out generalization metrics, see
compare_methods.py (Leave-One-Patient-Out CV).

Usage:
  python train.py                     # default: CNN1D (best from LOOCV)
  python train.py --model LSTM
  python train.py --model ANN --epochs 200
  python train.py --model CNN1D --seed 0
"""

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader, random_split

from dataset import GaitDataset
from model import MODEL_REGISTRY

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="CNN1D", choices=list(MODEL_REGISTRY),
                    help="Model architecture (default: CNN1D)")
parser.add_argument("--epochs", type=int, default=None,
                    help="Override epochs from config.yaml")
parser.add_argument("--seed",   type=int, default=42)
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
EPOCHS = args.epochs or config["train"]["epochs"]
LR     = config["train"]["lr"]
BATCH  = config["train"]["batch_size"]

# ── Data ──────────────────────────────────────────────────────────────────────
dataset = GaitDataset(config["train"]["data_path"])
n_val   = max(1, int(len(dataset) * 0.2))
n_fit   = len(dataset) - n_val

torch.manual_seed(args.seed)
fit_set, val_set = random_split(dataset, [n_fit, n_val])

fit_loader = DataLoader(fit_set, batch_size=BATCH, shuffle=True,  drop_last=False)
val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False, drop_last=False)

# ── Model ─────────────────────────────────────────────────────────────────────
model     = MODEL_REGISTRY[args.model](**config["model"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()
n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model   : {args.model}  ({n_params:,} params)")
print(f"Device  : {device}")
print(f"Epochs  : {EPOCHS}  |  LR: {LR}  |  Batch: {BATCH}")
print(f"Samples : fit={n_fit}  val={n_val}  (seed={args.seed})")
print()

# ── Training loop ─────────────────────────────────────────────────────────────
os.makedirs("weights", exist_ok=True)
out_path = f"weights/final_{args.model.lower()}.pth"

best_val  = float("inf")
best_state = None

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for inputs, targets in fit_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            val_loss += criterion(model(inputs), targets).item()

    avg_train = train_loss / len(fit_loader)
    avg_val   = val_loss   / len(val_loader)

    if avg_val < best_val:
        best_val   = avg_val
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        saved_mark = " *"
    else:
        saved_mark = ""

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}]  "
              f"train={avg_train:.5f}  val={avg_val:.5f}{saved_mark}")

# ── Save ──────────────────────────────────────────────────────────────────────
model.load_state_dict(best_state)
torch.save(best_state, out_path)
print(f"\nBest val MSE : {best_val:.5f}")
print(f"Weights saved: {out_path}")
