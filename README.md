# Final Project: Predicting COM–COP Relative Motion During Locomotion from Ground Reaction Forces

---

## 1. Objective

The displacement vector of the centre of mass (COM) relative to the centre of pressure (COP) directly determines the rate of change of whole-body angular momentum and is a key indicator of balance control during gait. Conventionally, obtaining COM position requires a full-body motion capture system, which is costly and operationally demanding.

This project investigates whether **ground reaction force (GRF) signals from a force plate alone** are sufficient to predict COM–COP relative displacement using deep learning, and compares the performance of five model architectures.

```
Input  : GRF  (T=100, 3)  —  Fx, Fy, Fz normalised to body weight
Output : ΔCOM (T=100, 2)  —  ML (X) and AP (Y) relative displacement in metres
```

---

## 2. Dataset

### Source

13 subjects performed level walking on a force plate at 240 Hz. Subject 10 was excluded because one trial contained NaN values — when any trial in a subject is corrupted, all trials from that subject are dropped. The final dataset contains **12 subjects and 106 trials**.

| Property | Value |
|---|---|
| Subjects | 12 |
| Trials per subject | 8–9 |
| Total trials | 106 |
| Sampling rate | 240 Hz |
| Normalised sequence length | 100 timesteps (0–100% gait cycle) |

### Signals

- **Input (GRF)**: Fx (mediolateral), Fy (anterior-posterior), Fz (vertical) — 3 channels
- **Output (COM–COP)**: mediolateral (ML, X) and anterior-posterior (AP, Y) relative displacement — 2 channels

### Preprocessing

1. GRF normalised by body weight (`mass × 9.81 N`) to remove inter-subject scaling
2. COM–COP relative displacement: `(COM[:, :2] − COP) / 1000`, converting mm to metres
3. Each trial resampled to 100 timesteps via linear interpolation to align gait cycle percentage
4. Any trial containing NaN causes the entire subject to be excluded at load time
5. Body height estimated as `height ≈ mean(COM_Z) / 0.57` (Dempster, 1955), used for dimensionless error normalisation

### Folder Structure

```
├── data/                        # processed dataset (patient-trial structure)
│   ├── mapping.csv              # maps new patient IDs to original identifiers
│   ├── patient01/
│   │   ├── trial01.mat          # one gait cycle: GRF, COM, COP, BodyWeight
│   │   └── ...
│   └── ...                      # patient10 excluded at load time (NaN)
│
├── compare_results.csv          # per-fold LOOCV metrics for all models (generated)
├── compare_results.png          # bar chart comparing all models and metrics (generated)
│
├── config.yaml                  # all hyperparameters (model, training)
├── dataset.py                   # GaitDataset: loading, preprocessing, NaN exclusion
├── model.py                     # all model architectures + MODEL_REGISTRY
├── train.py                     # train a single model on full dataset
├── compare_methods.py           # LOOCV across all models; generates CSV and charts
├── cv_train.py                  # LOOCV for LSTM only (single-model reference)
├── eval.py                      # visualise one prediction from a trained model
├── move.py                      # one-time script: reorganises DataSet/ → data/
├── load_mat.py                  # utility: inspect raw .mat file contents
└── README.md
```

Not tracked by git (`.gitignore`): `weights/`, `results/`, `DataSet/`, `__pycache__/`

---

## 3. Methods

### Model Architectures

Five architectures are compared with shared hyperparameters (`hidden_size=64`, `num_layers=2`, `dropout=0.2`):

| Model | Description | Params |
|---|---|---|
| ANN | Pointwise MLP with no temporal context — lower-bound baseline | 4,546 |
| RNN | Vanilla recurrent network with hidden state propagation | 12,866 |
| LSTM | Gated recurrent network with input, forget, and output gates | 51,074 |
| Transformer | Self-attention over the full sequence with positional encoding | 67,330 |
| CNN1D | 6-block dilated 1D CNN with residual connections (TCN), receptive field = 253 | 75,394 |

### Training Setup

- Optimiser: Adam, learning rate 0.001
- Loss: Mean Squared Error (MSE)
- Gradient clipping: max norm 1.0
- Epochs: 100, batch size: 16
- Early stopping: within each fold, 20% of the training set is held out as an internal validation set; the checkpoint with the lowest validation MSE is saved

### Cross-Validation Design

**Leave-One-Patient-Out CV (LOOCV, 12 folds)**: each subject serves as the test set exactly once while the remaining 11 subjects train the model. The test fold is never seen during training or model selection. This strategy maximises training data usage under a small dataset and provides a reliable estimate of generalisation to unseen individuals.

### Evaluation Metrics

| Metric | Description |
|---|---|
| RMSE X / Y (mm) | Root mean squared error in the ML and AP directions |
| Pearson r X / Y | Pearson correlation between predicted and ground-truth trajectories |
| Angular error (°) | Mean absolute angular deviation between predicted and GT displacement vectors, referenced to the AP (Y) axis at 0° |
| Height-norm RMSE (% height) | RMSE divided by estimated body height — a dimensionless, subject-independent error measure |

---

## 4. Results

### LOOCV Summary (mean ± std, 12 folds)

| Model | RMSE X (mm) | RMSE Y (mm) | r X | r Y | Angle (°) | norm X (%) | norm Y (%) |
|---|---|---|---|---|---|---|---|
| **CNN1D** | **12.98 ± 2.52** | **20.16 ± 3.77** | **0.976 ± 0.011** | **0.970 ± 0.015** | **11.45 ± 2.09** | **0.775 ± 0.158** | **1.204 ± 0.251** |
| LSTM | 16.73 ± 2.96 | 26.85 ± 4.37 | 0.956 ± 0.014 | 0.938 ± 0.021 | 14.39 ± 2.47 | 0.999 ± 0.197 | 1.602 ± 0.293 |
| ANN | 16.83 ± 3.61 | 35.26 ± 5.28 | 0.963 ± 0.014 | 0.890 ± 0.042 | 19.91 ± 2.25 | 1.003 ± 0.209 | 2.103 ± 0.345 |
| RNN | 18.43 ± 3.03 | 31.33 ± 7.83 | 0.945 ± 0.018 | 0.914 ± 0.047 | 16.04 ± 3.68 | 1.097 ± 0.171 | 1.865 ± 0.483 |
| Transformer | 44.47 ± 11.72 | 38.66 ± 12.68 | 0.577 ± 0.400 | 0.879 ± 0.081 | 37.90 ± 13.63 | 2.640 ± 0.681 | 2.298 ± 0.747 |

Detailed comparison chart: `compare_results.png`. Per-fold prediction plots: `results/` directory.

---

## 5. Discussion

**CNN1D achieves the best performance across all metrics.** The dilated receptive field (253 timesteps) fully covers the normalised gait cycle, allowing the model to simultaneously capture short-range (single-step impulse) and long-range (propulsion–braking cycle) GRF patterns. Residual connections prevent gradient vanishing across 6 stacked blocks, and LayerNorm mitigates feature-scale mismatch between training and test subjects.

**Temporal models outperform ANN in the AP direction.** ANN achieves competitive ML-direction RMSE (16.83 mm) — lateral sway is relatively simple — but degrades significantly in the AP direction (35.26 mm), where the propulsion–braking structure requires temporal context for accurate prediction.

**Transformer fails to generalise.** With only 78 training samples per fold, the self-attention mechanism cannot converge. The r_X standard deviation reaches 0.40, with some folds producing near-zero or negative correlations — a clear sign of overfitting in the small-data regime.

**Height-normalised RMSE enables cross-subject comparison.** CNN1D achieves 0.775% / 1.204% of body height in ML / AP directions, providing a dimensionless benchmark that is independent of subject stature and comparable across datasets.

---

## 6. Conclusion

GRF signals alone contain sufficient information to predict COM–COP relative displacement throughout the gait cycle. Under limited data conditions, CNN1D (TCN) consistently outperforms recurrent networks and Transformer across all metrics, achieving approximately 13 mm RMSE in the ML direction and 20 mm in the AP direction under subject-independent cross-validation. These results support the use of GRF-only signals as a viable alternative for real-time gait balance assessment.

---

## Usage

### `train.py` — Train final models

Trains one or all models on the full dataset (80/20 random val split for early stopping only).

```bash
python train.py                          # train all 5 models (default)
python train.py --model CNN1D            # train one specific model
python train.py --model ANN --epochs 200 # override epoch count
python train.py --seed 0                 # change random seed for the val split
```

| Argument | Default | Description |
|---|---|---|
| `--model` | *(all)* | One of `ANN`, `RNN`, `LSTM`, `Transformer`, `CNN1D` |
| `--epochs` | from `config.yaml` | Override training epoch count |
| `--seed` | `42` | Random seed for the 80/20 val split |

Weights saved to `weights/final_{model}.pth`.

---

### `compare_methods.py` — Leave-One-Patient-Out CV

Runs 12-fold LOOCV across all models and generates a comparison chart.

```bash
python compare_methods.py            # run CV and save summary chart
python compare_methods.py --plot     # also save per-fold prediction plots
```

| Argument | Default | Description |
|---|---|---|
| `--plot` | off | Save per-fold prediction plots to `results/` (60 files total) |

Outputs:
- `compare_results.csv` — per-fold metrics for every model
- `compare_results.png` — 2×3 bar chart (RMSE, r, angular error, height-norm RMSE)
- `results/{Model}_fold{N}_{patient}.png` — per-fold prediction plots (`--plot` only)

**Resume-safe**: if `compare_results.csv` already exists, completed folds are skipped. Pass `--plot` at any time to generate plots from saved fold weights without retraining.

---

### `eval.py` — Visualise a single prediction

```bash
python eval.py                   # default: CNN1D
python eval.py --model LSTM
python eval.py --model CNN1D --seed 7
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `CNN1D` | Model to load from `weights/final_{model}.pth` |
| `--seed` | `42` | Selects which val-set sample to visualise |

Prints RMSE X/Y, height-normalised RMSE, and angular error for the sampled trial.
Plot saved to `results/eval_{model}.png`.

---

### Configuration

All hyperparameters in `config.yaml`:

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
