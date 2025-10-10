import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_erbb1_mlp import Erbb1MlpConfig


# ─── building blocks ────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_out * 2)
        self.fc2 = nn.Linear(d_out, d_out)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        return self.fc2(F.silu(x1) * x2)


class FeedForwardLayer(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1, layer_norm_eps=1e-12):
        super().__init__()
        self.ff = FeedForward(d_in, d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out, eps=layer_norm_eps) if layer_norm_eps else nn.Identity() 

    def forward(self, x):
        y = self.ff(self.dropout(x)) + self.skip(x)
        return self.norm(y)


# ─── HF Model wrapper ───────────────────────────────────────────
@dataclass
class Erbb1MlpOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    prediction: torch.FloatTensor = None         # denormalised
    prediction_norm: torch.FloatTensor = None    # normalised


class Erbb1MlpModel(PreTrainedModel):
    config_class = Erbb1MlpConfig

    def __init__(self, config: Erbb1MlpConfig):
        super().__init__(config)

        layers = [FeedForwardLayer(config.d_in, config.d_hidden, 0.0, config.layer_norm_eps)]
        layers += [
            FeedForwardLayer(config.d_hidden, config.d_hidden, config.dropout, config.layer_norm_eps)
            for _ in range(config.n_layers - 1)
        ]
        self.body = nn.Sequential(*layers)
        self.out_proj = nn.Linear(config.d_hidden, 1)

        # stats for de-normalising (stored in state dict)
        mean = torch.tensor(config.dataset_mean or 0.0, dtype=torch.float32)
        std = torch.tensor(config.dataset_std or 1.0, dtype=torch.float32)
        self.register_buffer("target_mean", mean, persistent=True)
        self.register_buffer("target_std", std, persistent=True)

        self.post_init()

    def forward(self, embedding, labels=None, return_dict=True):
        x = self.body(embedding)
        pred_norm = self.out_proj(x).squeeze(-1)
        pred = pred_norm * self.target_std + self.target_mean

        loss = None
        if labels is not None:
            loss = F.mse_loss(pred_norm, labels)

        if not return_dict:
            return (loss, pred, pred_norm)

        return Erbb1MlpOutput(
            loss=loss,
            prediction=pred,
            prediction_norm=pred_norm,
        )
