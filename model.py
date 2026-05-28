import math
import torch
import torch.nn as nn


class ANNModel(nn.Module):
    """Pointwise MLP — no temporal context, applied independently at each timestep.
    Serves as the baseline that ignores sequential structure."""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, **kwargs):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B, T, input_size) → (B, T, output_size)


class RNNModel(nn.Module):
    """Vanilla RNN — simple recurrent connection, no gating."""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, **kwargs):
        super().__init__()
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)


class LSTMModel(nn.Module):
    """LSTM — Long Short-Term Memory with learnable gating."""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


class _TCNBlock(nn.Module):
    """Pre-LN TCN block: LayerNorm → Conv → ReLU → Dropout + residual.

    Without normalisation, ReLU residual blocks accumulate positive feature
    magnitudes across layers (residual drift). The output projection is
    calibrated to the training subjects' feature scale; a different test
    subject drives features to a different magnitude, causing correct-shape
    but wrong-scale predictions (high r, catastrophic RMSE).

    LayerNorm normalises over the channel dimension independently per
    timestep and per sample — no running statistics, no train/test
    distribution mismatch regardless of which subject is held out."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T) — normalise over C at each timestep, then conv + residual
        normed = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x + self.drop(self.relu(self.conv(normed)))


class CNN1DModel(nn.Module):
    """TCN (Temporal Convolutional Network): dilated 1D CNN with residual connections.

    Dilation doubles each block (2^0 → 2^1 → ... → 2^5), giving a total
    receptive field of 253 timesteps with kernel_size=3 — sufficient to cover
    the full 100-step gait cycle with margin.

    Why the naive CNN (kernel=5, 2 layers, RF=9) failed:
      Each output timestep could only see ±4 neighbours, so the model had no
      access to the GRF dynamics across the rest of the stride. Dilated
      convolutions solve this without adding parameters proportional to the
      sequence length."""

    # RF with kernel=3, 6 blocks, dilations 1,2,4,8,16,32:
    #   RF = 1 + 4*(1+2+4+8+16+32) = 1 + 4*63 = 253
    _N_BLOCKS = 6

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout,
                 kernel_size=3, **kwargs):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.blocks = nn.ModuleList([
            _TCNBlock(hidden_size, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(self._N_BLOCKS)
        ])
        self.norm_out = nn.LayerNorm(hidden_size)   # guards output projection from drift
        self.fc = nn.Conv1d(hidden_size, output_size, kernel_size=1)

    def forward(self, x):
        # x: (B, T, C_in) → (B, C_in, T) → blocks → norm → (B, T, C_out)
        x = self.input_proj(x.permute(0, 2, 1))
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.fc(x).permute(0, 2, 1)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerModel(nn.Module):
    """Transformer encoder — self-attention over the full GRF sequence.
    nhead must divide hidden_size evenly (default nhead=4, hidden_size=64 → OK)."""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout,
                 nhead=4, **kwargs):
        super().__init__()
        assert hidden_size % nhead == 0, f"hidden_size ({hidden_size}) must be divisible by nhead ({nhead})"
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_enc = _PositionalEncoding(hidden_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        return self.fc(self.encoder(x))


# Registry: name → class (used by compare_methods.py and cv_train.py)
MODEL_REGISTRY = {
    "ANN":         ANNModel,
    "RNN":         RNNModel,
    "LSTM":        LSTMModel,
    "Transformer": TransformerModel,
    "CNN1D":       CNN1DModel,
}

# Backward-compatible alias so existing eval.py / train.py still work
GRFtoCOMModel = LSTMModel
