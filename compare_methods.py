"""
Method comparison via Leave-One-Patient-Out CV (13 patients).

Each of the 13 patients in data/ is a distinct individual.
One patient is held out as test per fold; the remaining 12 train.
Within each training fold, 20% is reserved as a validation set for
early stopping so the test fold never influences model selection.

Models compared:  ANN | RNN | LSTM | Transformer | CNN1D
Metrics reported: MSE (m²) | RMSE_X (mm) | RMSE_Y (mm) | r_X | r_Y
Output:           compare_results.csv  +  compare_results.png

Resume behaviour: if compare_results.csv already exists, models that
already have all 13 folds recorded are skipped (no re-training).
"""

import csv
import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

from dataset import GaitDataset
from model import MODEL_REGISTRY

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

DEVICE     = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config["train"]["batch_size"]
LR         = config["train"]["lr"]
EPOCHS     = config["train"]["epochs"]
MODEL_CFG  = config["model"]

os.makedirs("weights", exist_ok=True)

print(f"Device: {DEVICE}  |  Epochs: {EPOCHS}  |  Models: {list(MODEL_REGISTRY)}\n")

# ── Resume: load any already-completed folds from CSV ─────────────────────────
csv_path    = "compare_results.csv"
METRIC_COLS = ["MSE_m2", "RMSE_X_mm", "RMSE_Y_mm", "r_X", "r_Y",
               "Angle_err_deg", "RMSE_X_norm_pct", "RMSE_Y_norm_pct"]

completed = defaultdict(dict)   # completed[model_name][fold_i] = metrics tuple
if os.path.exists(csv_path):
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fold_i  = int(row["Fold"]) - 1
            metrics = tuple(float(row[c]) for c in METRIC_COLS)
            completed[row["Model"]][fold_i] = metrics
    loaded = {m: len(v) for m, v in completed.items()}
    print(f"Resuming from {csv_path}: {loaded}")
    print()

# ── Dataset ───────────────────────────────────────────────────────────────────
full_dataset = GaitDataset(config["train"]["data_path"])

# All 13 patients are distinct individuals — fold directly on patient_id
patient_ids = sorted(set(d["patient_id"] for d in full_dataset.data))
N_FOLDS     = len(patient_ids)

print(f"Total samples: {len(full_dataset)}  |  Patients: {N_FOLDS}")
for pid in patient_ids:
    n = sum(1 for d in full_dataset.data if d["patient_id"] == pid)
    print(f"  {pid}: {n} trials")
print()

# ── Build fold index lists once (shared across all models) ────────────────────
folds = []
for test_pid in patient_ids:
    train_idx = [i for i, d in enumerate(full_dataset.data) if d["patient_id"] != test_pid]
    test_idx  = [i for i, d in enumerate(full_dataset.data) if d["patient_id"] == test_pid]
    folds.append((test_pid, train_idx, test_idx))

# ── Helpers ───────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

def plot_fold(model, loader, model_name, fold_i, test_pid):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            p = model(inputs.to(DEVICE)).cpu().numpy().reshape(-1, 2)
            t = targets.numpy().reshape(-1, 2)
            all_preds.append(p)
            all_trues.append(t)
    
    pred = np.concatenate(all_preds)   # (N*100, 2)
    true = np.concatenate(all_trues)

    p_trial = pred[:100]
    t_trial = true[:100]
    gait_pct = np.linspace(0, 100, 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{model_name} — Fold {fold_i+1} (test={test_pid})", fontsize=13)

    axes[0, 0].plot(gait_pct, t_trial[:, 0], color='gray', label='Ground Truth')
    axes[0, 0].plot(gait_pct, p_trial[:, 0], 'r--', label=f'{model_name} Prediction')
    axes[0, 0].set_title('Direction X (ML)')
    axes[0, 0].set_ylabel('Displacement (m)')
    axes[0, 0].legend()

    axes[1, 0].plot(gait_pct, t_trial[:, 1], color='gray', label='Ground Truth')
    axes[1, 0].plot(gait_pct, p_trial[:, 1], 'b--', label=f'{model_name} Prediction')
    axes[1, 0].set_title('Direction Y (AP)')
    axes[1, 0].set_ylabel('Displacement (m)')
    axes[1, 0].set_xlabel('Gait Cycle (%)')
    axes[1, 0].legend()

    p_mean = pred.reshape(-1, 100, 2).mean(axis=0)
    t_mean = true.reshape(-1, 100, 2).mean(axis=0)

    axes[0, 1].plot(gait_pct, t_mean[:, 0], color='gray', label='Ground Truth')
    axes[0, 1].plot(gait_pct, p_mean[:, 0], 'r--', label=f'{model_name} Prediction')
    axes[0, 1].set_title('COM-COP Relative Motion (Direction X) — Mean')
    axes[0, 1].set_ylabel('Displacement (m)')
    axes[0, 1].legend()

    axes[1, 1].plot(gait_pct, t_mean[:, 1], color='gray', label='Ground Truth')
    axes[1, 1].plot(gait_pct, p_mean[:, 1], 'b--', label=f'{model_name} Prediction')
    axes[1, 1].set_title('COM-COP Relative Motion (Direction Y) — Mean')
    axes[1, 1].set_ylabel('Displacement (m)')
    axes[1, 1].set_xlabel('Gait Cycle (%)')
    axes[1, 1].legend()

    plt.tight_layout()
    save_path = f"results/{model_name}_fold{fold_i+1:02d}_{test_pid}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_fold(model_cls, train_idx, test_idx, val_ratio=0.2, seed=42):
    """Train with an internal val split for early stopping.

    The test fold is never seen during training or model selection — it is
    only used once at the end to compute the reported metrics.
    """
    # ── Carve out a validation set from the training indices ──────────────────
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(len(train_idx))
    n_val = max(1, int(len(train_idx) * val_ratio))
    val_idx = [train_idx[i] for i in perm[:n_val]]
    fit_idx = [train_idx[i] for i in perm[n_val:]]

    fit_loader  = DataLoader(Subset(full_dataset, fit_idx),  batch_size=BATCH_SIZE, shuffle=True)
    val_loader  = DataLoader(Subset(full_dataset, val_idx),  batch_size=BATCH_SIZE)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=BATCH_SIZE)

    model     = model_cls(**MODEL_CFG).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None

    for _ in range(EPOCHS):
        model.train()
        for inputs, targets in fit_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ── Early stopping uses validation loss only ──────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                val_loss += criterion(model(inputs), targets).item()

        if val_loss / len(val_loader) < best_val_loss:
            best_val_loss = val_loss / len(val_loader)
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, test_loader, len(fit_idx), len(val_idx)


def eval_metrics(model, loader, test_idx):
    """
    Returns (mse_m2, rmse_x_mm, rmse_y_mm, r_x, r_y,
             angle_err_deg, rmse_x_norm_pct, rmse_y_norm_pct).

    angle_err_deg:
        Mean absolute angular error (degrees) between predicted and ground-truth
        COM-COP displacement vectors, measured from the AP (Y) axis at 0°.

    rmse_x/y_norm_pct:
        RMSE normalised by estimated body height, expressed as % of height.
        Height is estimated per subject as mean(COM_Z) / 0.57 (Dempster 1955).
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            p = model(inputs.to(DEVICE)).cpu().numpy().reshape(-1, 2)
            t = targets.numpy().reshape(-1, 2)
            preds.append(p)
            trues.append(t)
    pred = np.concatenate(preds)   # (N*T, 2)
    true = np.concatenate(trues)

    mse    = float(np.mean((pred - true) ** 2))
    rmse_x = float(np.sqrt(np.mean((pred[:, 0] - true[:, 0]) ** 2)) * 1000)
    rmse_y = float(np.sqrt(np.mean((pred[:, 1] - true[:, 1]) ** 2)) * 1000)
    r_x    = float(np.corrcoef(pred[:, 0], true[:, 0])[0, 1])
    r_y    = float(np.corrcoef(pred[:, 1], true[:, 1])[0, 1])

    # Angular deviation: angle of 2-D vector from AP (Y) axis
    # atan2(X, Y) = 0° when displacement is purely anterior-posterior
    angle_pred = np.arctan2(pred[:, 0], pred[:, 1])
    angle_true = np.arctan2(true[:, 0], true[:, 1])
    diff = (angle_pred - angle_true + np.pi) % (2 * np.pi) - np.pi  # wrap to (-π, π]
    angle_err = float(np.mean(np.abs(diff)) * 180.0 / np.pi)

    # Height-normalised RMSE: each timestep divided by subject height
    heights = np.array([full_dataset.data[i]["height_m"] for i in test_idx])
    heights_per_step = np.repeat(heights, 100)          # (N*T,)
    rmse_x_norm = float(np.sqrt(np.mean(((pred[:, 0] - true[:, 0]) / heights_per_step) ** 2)) * 100)
    rmse_y_norm = float(np.sqrt(np.mean(((pred[:, 1] - true[:, 1]) / heights_per_step) ** 2)) * 100)

    return mse, rmse_x, rmse_y, r_x, r_y, angle_err, rmse_x_norm, rmse_y_norm


# ── Main LOOCV loop ───────────────────────────────────────────────────────────
results  = {}
n_params = {}

for model_name, model_cls in MODEL_REGISTRY.items():
    n_params[model_name] = count_params(model_cls(**MODEL_CFG))

    if len(completed.get(model_name, {})) == N_FOLDS:
        print(f"[skip] {model_name} — all {N_FOLDS} folds already in CSV")
        results[model_name] = [completed[model_name][i] for i in range(N_FOLDS)]
        continue

    print(f"\n{'='*60}")
    print(f"Model: {model_name}  ({n_params[model_name]:,} params)")
    print(f"{'='*60}")

    fold_metrics = []
    for fold_i, (test_pid, train_idx, test_idx) in enumerate(folds):
        if fold_i in completed.get(model_name, {}):
            metrics = completed[model_name][fold_i]
            mse, rx, ry, r_x, r_y = metrics
            print(f"  Fold {fold_i+1:>2}/{N_FOLDS} (test={test_pid}) [cached]  "
                  f"RMSE_X={rx:.1f}mm  RMSE_Y={ry:.1f}mm  r_X={r_x:.3f}  r_Y={r_y:.3f}")
            fold_metrics.append(metrics)
            continue

        print(f"  Fold {fold_i+1:>2}/{N_FOLDS} (test={test_pid}) ...",
              end=" ", flush=True)
        model, test_loader, n_fit, n_val = train_one_fold(model_cls, train_idx, test_idx)
        print(f"fit={n_fit} val={n_val} test={len(test_idx)}", end="  ", flush=True)
        metrics = eval_metrics(model, test_loader, test_idx)
        plot_fold(model, test_loader, model_name, fold_i, test_pid)
        fold_metrics.append(metrics)
        mse, rx, ry, r_x, r_y, ang, nx, ny = metrics
        print(f"RMSE_X={rx:.1f}mm  RMSE_Y={ry:.1f}mm  "
              f"r_X={r_x:.3f}  r_Y={r_y:.3f}  "
              f"Angle={ang:.2f}°  normX={nx:.3f}%  normY={ny:.3f}%")

        weight_path = f"weights/{model_name.lower()}_fold{fold_i+1:02d}.pth"
        torch.save(model.state_dict(), weight_path)

    results[model_name] = fold_metrics

# ── Summary table ─────────────────────────────────────────────────────────────
METRIC_NAMES = ["MSE(m²)", "RMSE_X(mm)", "RMSE_Y(mm)", "r_X", "r_Y",
                "Angle(°)", "normX(%h)", "normY(%h)"]

print(f"\n\n{'='*100}")
print("LOOCV Summary, mean(std)")
print(f"{'='*100}")
header = f"{'Model':<14} {'Params':>8}  " + "  ".join(f"{m:>16}" for m in METRIC_NAMES)
print(header)
print("-" * len(header))

summary = {}
for model_name in MODEL_REGISTRY:
    if model_name not in results:
        continue
    arr   = np.array(results[model_name])   # (N_FOLDS, 8)
    means = arr.mean(axis=0)
    stds  = arr.std(axis=0)
    summary[model_name] = (means, stds)
    row = f"{model_name:<14} {n_params[model_name]:>8,}  "
    row += "  ".join(f"{m:>8.4f}({s:.4f})" for m, s in zip(means, stds))
    print(row)

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Params", "Fold", "Patient"] + METRIC_COLS)
    for model_name in MODEL_REGISTRY:
        if model_name not in results:
            continue
        for fold_i, (metrics, (test_pid, _, __)) in enumerate(
                zip(results[model_name], folds)):
            writer.writerow([model_name, n_params[model_name],
                             fold_i + 1, test_pid,
                             *[f"{v:.6f}" for v in metrics]])
print(f"\nDetailed results saved → {csv_path}")

# ── 2×3 bar-chart ─────────────────────────────────────────────────────────────
model_names = [m for m in MODEL_REGISTRY if m in summary]
x      = np.arange(len(model_names))
bar_w  = 0.6
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9467BD"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f"Leave-One-Patient-Out CV ({N_FOLDS} patients) — Model Comparison",
             fontsize=13)

# (metric_index_in_tuple, title, y-label, higher_is_better)
panels = [
    (1, "RMSE_X  (ML, mm)",            "mm",  False),
    (2, "RMSE_Y  (AP, mm)",            "mm",  False),
    (5, "Angular Error  (°)",          "deg", False),
    (3, "Pearson r_X  (ML)",           "r",   True),
    (4, "Pearson r_Y  (AP)",           "r",   True),
    (6, "Height-norm RMSE_X  (% Ht)",  "%",   False),
]

for ax, (metric_idx, title, ylabel, higher_is_better) in zip(axes.flat, panels):
    means = [summary[m][0][metric_idx] for m in model_names]
    stds  = [summary[m][1][metric_idx] for m in model_names]
    bars  = ax.bar(x, means, bar_w, yerr=stds, capsize=5,
                   color=colors[: len(model_names)])
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    best_i = int(np.argmin(means) if not higher_is_better else np.argmax(means))
    bars[best_i].set_edgecolor("black")
    bars[best_i].set_linewidth(2)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + max(stds) * 0.02,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("compare_results.png", dpi=150)
print("Comparison chart saved → compare_results.png")
plt.show()
