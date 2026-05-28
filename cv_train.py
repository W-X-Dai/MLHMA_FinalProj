"""
Leave-One-Patient-Out CV for the LSTM model (single-model training script).

All 13 patients in data/ are distinct individuals.
One patient is held out as test per fold; the remaining 12 train.
Within each fold, 20% of training data is reserved as a validation set
for early stopping so the test fold is never used during model selection.

Weights saved to:  weights/lstm_fold<N>.pth
"""

import csv
import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader, Subset

from dataset import GaitDataset
from model import GRFtoCOMModel   # LSTM alias

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("weights", exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
full_dataset = GaitDataset(config["train"]["data_path"])
patient_ids  = sorted(set(d["patient_id"] for d in full_dataset.data))
N_FOLDS      = len(patient_ids)

print(f"\nTotal samples: {len(full_dataset)}  |  Patients: {N_FOLDS}")
for pid in patient_ids:
    n = sum(1 for d in full_dataset.data if d["patient_id"] == pid)
    print(f"  {pid}: {n} trials")

# ── Helpers ───────────────────────────────────────────────────────────────────
def eval_metrics(model, loader, test_idx):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            preds.append(out.cpu().numpy().reshape(-1, 2))
            trues.append(targets.cpu().numpy().reshape(-1, 2))
    pred = np.concatenate(preds)
    true = np.concatenate(trues)

    mse    = float(np.mean((pred - true) ** 2))
    rmse_x = float(np.sqrt(np.mean((pred[:, 0] - true[:, 0]) ** 2)) * 1000)
    rmse_y = float(np.sqrt(np.mean((pred[:, 1] - true[:, 1]) ** 2)) * 1000)
    r_x    = float(np.corrcoef(pred[:, 0], true[:, 0])[0, 1])
    r_y    = float(np.corrcoef(pred[:, 1], true[:, 1])[0, 1])

    angle_pred = np.arctan2(pred[:, 0], pred[:, 1])
    angle_true = np.arctan2(true[:, 0], true[:, 1])
    diff       = (angle_pred - angle_true + np.pi) % (2 * np.pi) - np.pi
    angle_err  = float(np.mean(np.abs(diff)) * 180.0 / np.pi)

    heights          = np.array([full_dataset.data[i]["height_m"] for i in test_idx])
    heights_per_step = np.repeat(heights, 100)
    rmse_x_norm = float(np.sqrt(np.mean(((pred[:, 0] - true[:, 0]) / heights_per_step) ** 2)) * 100)
    rmse_y_norm = float(np.sqrt(np.mean(((pred[:, 1] - true[:, 1]) / heights_per_step) ** 2)) * 100)

    return mse, rmse_x, rmse_y, r_x, r_y, angle_err, rmse_x_norm, rmse_y_norm

# ── LOOCV ─────────────────────────────────────────────────────────────────────
fold_results = []

for fold_i, test_pid in enumerate(patient_ids):
    print(f"\n{'='*60}")
    print(f"Fold {fold_i+1}/{N_FOLDS}: test = {test_pid}")

    all_train_idx = [i for i, d in enumerate(full_dataset.data) if d["patient_id"] != test_pid]
    test_idx      = [i for i, d in enumerate(full_dataset.data) if d["patient_id"] == test_pid]

    # 80/20 val split within training data
    rng   = np.random.default_rng(42)
    perm  = rng.permutation(len(all_train_idx))
    n_val = max(1, int(len(all_train_idx) * 0.2))
    val_idx = [all_train_idx[i] for i in perm[:n_val]]
    fit_idx = [all_train_idx[i] for i in perm[n_val:]]

    print(f"  fit={len(fit_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    fit_loader  = DataLoader(Subset(full_dataset, fit_idx),  batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader  = DataLoader(Subset(full_dataset, val_idx),  batch_size=config["train"]["batch_size"])
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=config["train"]["batch_size"])

    model     = GRFtoCOMModel(**config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(config["train"]["epochs"]):
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
        avg_val = val_loss / len(val_loader)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1:3d}/{config['train']['epochs']}] "
                  f"train={train_loss/len(fit_loader):.5f}  val={avg_val:.5f}")

    model.load_state_dict(best_state)
    weight_path = f"weights/lstm_fold{fold_i+1:02d}.pth"
    torch.save(best_state, weight_path)

    mse, rmse_x, rmse_y, r_x, r_y, angle_err, nx, ny = eval_metrics(model, test_loader, test_idx)
    fold_results.append((mse, rmse_x, rmse_y, r_x, r_y, angle_err, nx, ny))
    print(f"\n  [Fold {fold_i+1} result]  "
          f"MSE={mse:.5f}  RMSE_X={rmse_x:.1f}mm  RMSE_Y={rmse_y:.1f}mm  "
          f"r_X={r_x:.4f}  r_Y={r_y:.4f}  Angle={angle_err:.2f}°  "
          f"normX={nx:.3f}%  normY={ny:.3f}%")
    print(f"  Weights saved → {weight_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"LOOCV Summary  ({N_FOLDS}-fold, mean ± std)")
print(f"{'='*70}")
print(f"{'Patient':<14} {'MSE(m²)':<12} {'RMSE_X(mm)':<12} {'RMSE_Y(mm)':<12} "
      f"{'r_X':<8} {'r_Y':<8} {'Angle(°)':<10} {'normX(%)':<10} {'normY(%)'}")
for pid, row in zip(patient_ids, fold_results):
    mse, rx, ry, r_x, r_y, ang, nx, ny = row
    print(f"{pid:<14} {mse:<12.5f} {rx:<12.1f} {ry:<12.1f} "
          f"{r_x:<8.4f} {r_y:<8.4f} {ang:<10.2f} {nx:<10.3f} {ny:.3f}")

arr   = np.array(fold_results)
means = arr.mean(axis=0)
stds  = arr.std(axis=0)
labels = ["MSE", "RMSE_X", "RMSE_Y", "r_X", "r_Y", "Angle", "normX", "normY"]
print()
print("Mean: " + "  ".join(f"{l}={m:.4f}" for l, m in zip(labels, means)))
print("Std:  " + "  ".join(f"{l}={s:.4f}" for l, s in zip(labels, stds)))
