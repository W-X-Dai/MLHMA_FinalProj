"""
Single-sample prediction visualiser.

Loads a trained model from weights/final_{model}.pth and plots ground
truth vs prediction for one randomly chosen trial.

Usage:
  python eval.py                  # default: CNN1D
  python eval.py --model LSTM
  python eval.py --model CNN1D --seed 7
"""

import argparse
import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from dataset import GaitDataset
from model import MODEL_REGISTRY

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="CNN1D", choices=list(MODEL_REGISTRY))
parser.add_argument("--seed",  type=int, default=42)
args = parser.parse_args()

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ── Load model ────────────────────────────────────────────────────────────────
weight_path = f"weights/final_{args.model.lower()}.pth"
model = MODEL_REGISTRY[args.model](**config["model"])
model.load_state_dict(torch.load(weight_path, map_location="cpu"))
model.eval()

# ── Use the same val split as train.py ────────────────────────────────────────
dataset = GaitDataset(config["train"]["data_path"])
n_val   = max(1, int(len(dataset) * 0.2))
n_fit   = len(dataset) - n_val
torch.manual_seed(args.seed)
_, val_set = random_split(dataset, [n_fit, n_val])

rng   = np.random.default_rng(args.seed)
idx   = rng.integers(len(val_set))
sample_input, sample_target = val_set[idx]

# Retrieve subject height via original dataset index
original_idx = val_set.indices[int(idx)]
height_m     = dataset.data[original_idx]["height_m"]

with torch.no_grad():
    prediction = model(sample_input.unsqueeze(0)).squeeze(0).numpy()
target = sample_target.numpy()

# ── Plot ──────────────────────────────────────────────────────────────────────
gait_pct = np.linspace(0, 100, target.shape[0])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle(f"{args.model} — Prediction vs Ground Truth", fontsize=13)

ax1.plot(gait_pct, target[:, 0],     color="gray", label="Ground Truth")
ax1.plot(gait_pct, prediction[:, 0], "r--",        label=f"{args.model} Prediction")
ax1.set_title("Direction X (ML)")
ax1.set_ylabel("COM–COP (m)")
ax1.legend()

ax2.plot(gait_pct, target[:, 1],     color="gray", label="Ground Truth")
ax2.plot(gait_pct, prediction[:, 1], "b--",        label=f"{args.model} Prediction")
ax2.set_title("Direction Y (AP)")
ax2.set_xlabel("Gait Cycle (%)")
ax2.set_ylabel("COM–COP (m)")
ax2.legend()

plt.tight_layout()
os.makedirs("results", exist_ok=True)
out = f"results/eval_{args.model.lower()}.png"
plt.savefig(out, dpi=150)
plt.show()

# ── Metrics ───────────────────────────────────────────────────────────────────
rmse_x = float(np.sqrt(np.mean((prediction[:, 0] - target[:, 0]) ** 2)) * 1000)
rmse_y = float(np.sqrt(np.mean((prediction[:, 1] - target[:, 1]) ** 2)) * 1000)

angle_pred = np.arctan2(prediction[:, 0], prediction[:, 1])
angle_true = np.arctan2(target[:, 0],     target[:, 1])
diff       = (angle_pred - angle_true + np.pi) % (2 * np.pi) - np.pi
angle_err  = float(np.mean(np.abs(diff)) * 180.0 / np.pi)

rmse_x_norm = rmse_x / 1000 / height_m * 100
rmse_y_norm = rmse_y / 1000 / height_m * 100

print(f"\nSaved → {out}")
print(f"Subject est. height : {height_m:.3f} m")
print(f"RMSE X (ML)         : {rmse_x:.2f} mm  ({rmse_x_norm:.3f}% height)")
print(f"RMSE Y (AP)         : {rmse_y:.2f} mm  ({rmse_y_norm:.3f}% height)")
print(f"Angular error       : {angle_err:.2f}°")
