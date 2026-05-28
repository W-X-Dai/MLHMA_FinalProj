# Predicting COM–COP Relative Motion from Ground Reaction Forces

Predicting the centre-of-mass (COM) motion relative to the centre-of-pressure (COP) during locomotion using ground reaction force (GRF) signals and deep learning.

---

## Task

Given a normalised gait cycle's GRF time series (Fx, Fy, Fz), predict the 2-D COM–COP relative displacement (mediolateral X and anterior-posterior Y) at every timestep.

```
Input  : GRF  (T=100, 3)   — Fx, Fy, Fz normalised to body weight
Output : ΔCOM  (T=100, 2)  — X (ML) and Y (AP) in metres
```

---

## Dataset

| Property | Value |
|---|---|
| Subjects | 12 (patient10 excluded — NaN in one trial drops the whole subject) |
| Trials per subject | 8–9 |
| Total samples | 106 |
| Sampling rate | 240 Hz |
| Sequence length | 100 timesteps (time-normalised to % gait cycle) |

### Preprocessing

1. GRF normalised by body weight: `GRF / (mass × 9.81)`
2. COM–COP relative displacement converted from mm to m: `(COM[:, :2] − COP) / 1000`
3. Each trial resampled to 100 timesteps via linear interpolation
4. Any trial containing NaN causes the entire subject to be excluded at load time
5. Subject height estimated from mean COM vertical position: `height ≈ mean(COM_Z) / 0.57` (Dempster 1955)

### Folder structure

```
data/
├── patient01/
│   ├── trial01.mat
│   └── ...
├── patient02/
│   └── ...
└── ...          # patient10 excluded at load time (NaN)
```

Each `.mat` file contains: `GRF`, `COM`, `COP`, `BodyWeight`, `frame_rate`.

---

## Models

All models share the same hyperparameters (`config.yaml`): `hidden_size=64`, `num_layers=2`, `dropout=0.2`.

| Model | Description | Params |
|---|---|---|
| ANN | Pointwise MLP — no temporal context | 4,546 |
| RNN | Vanilla recurrent network | 12,866 |
| LSTM | Long Short-Term Memory | 51,074 |
| Transformer | Self-attention over full sequence + positional encoding | 67,330 |
| CNN1D | Dilated TCN with residual connections (6 blocks, RF=253) | 75,394 |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| RMSE X / Y (mm) | Root mean squared error in ML and AP directions |
| Pearson r X / Y | Correlation between predicted and ground-truth trajectories |
| Angular error (°) | Mean absolute angular deviation of the predicted displacement vector from ground truth, measured from the AP (Y) axis at 0° |
| Height-norm RMSE (% height) | RMSE divided by estimated body height, expressed as percentage — provides a dimensionless, subject-independent error measure |

---

## Results (Leave-One-Patient-Out CV, 12-fold)

| Model | RMSE X (mm) | RMSE Y (mm) | r X | r Y | Angle (°) | norm X (%) | norm Y (%) |
|---|---|---|---|---|---|---|---|
| **CNN1D** | **12.98 ± 2.52** | **20.16 ± 3.77** | **0.976 ± 0.011** | **0.970 ± 0.015** | **11.45 ± 2.09** | **0.775 ± 0.158** | **1.204 ± 0.251** |
| LSTM | 16.73 ± 2.96 | 26.85 ± 4.37 | 0.956 ± 0.014 | 0.938 ± 0.021 | 14.39 ± 2.47 | 0.999 ± 0.197 | 1.602 ± 0.293 |
| ANN | 16.83 ± 3.61 | 35.26 ± 5.28 | 0.963 ± 0.014 | 0.890 ± 0.042 | 19.91 ± 2.25 | 1.003 ± 0.209 | 2.103 ± 0.345 |
| RNN | 18.43 ± 3.03 | 31.33 ± 7.83 | 0.945 ± 0.018 | 0.914 ± 0.047 | 16.04 ± 3.68 | 1.097 ± 0.171 | 1.865 ± 0.483 |
| Transformer | 44.47 ± 11.72 | 38.66 ± 12.68 | 0.577 ± 0.400 | 0.879 ± 0.081 | 37.90 ± 13.63 | 2.640 ± 0.681 | 2.298 ± 0.747 |

CNN1D (TCN) achieves the lowest error across all metrics. The Transformer underperforms due to insufficient training data for attention-based learning.

---

## Usage

### 1. Train a final model (all data, 80/20 val split for early stopping)

```bash
python train.py                     # default: CNN1D
python train.py --model LSTM
python train.py --model ANN --epochs 200
python train.py --model CNN1D --seed 0
```

Weights saved to `weights/final_{model}.pth`.

### 2. Run Leave-One-Patient-Out CV across all models

```bash
python compare_methods.py
```

Outputs:
- `compare_results.csv` — per-fold metrics for every model
- `compare_results.png` — bar chart comparing all metrics across models
- `results/{Model}_fold{N}_{patient}.png` — per-fold prediction plots

Resume-safe: already-completed folds are skipped if `compare_results.csv` exists.

### 3. Visualise a single prediction

```bash
python eval.py                  # default: CNN1D
python eval.py --model LSTM
python eval.py --model CNN1D --seed 7
```

Prints RMSE, height-normalised RMSE, and angular error for the sampled trial.
Output plot saved to `results/eval_{model}.png`.

---

## Configuration

All hyperparameters are in `config.yaml`:

```yaml
model:
  input_size: 3       # GRF channels (Fx, Fy, Fz)
  hidden_size: 64
  num_layers: 2
  output_size: 2      # COM-COP X and Y
  dropout: 0.2

train:
  batch_size: 16
  lr: 0.001
  epochs: 100
  data_path: "./data"
  device: "cuda"
```
