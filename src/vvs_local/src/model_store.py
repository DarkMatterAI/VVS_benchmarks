from __future__ import annotations
from typing import Sequence
import torch, torch.nn.functional as F
from transformers import (AutoTokenizer,
                          AutoModel,
                          DataCollatorWithPadding)

from .constants import DEVICE, EMB_MODEL_NM, DECOMP_MODEL_NM


class ModelStore:
    """Singleton-ish wrapper that owns the encoder & decomposer once."""

    def __init__(self, comp_size: int, *, device: str = DEVICE) -> None:
        self.comp_size = comp_size
        self.device    = device

        self.tok   = AutoTokenizer.from_pretrained(EMB_MODEL_NM, use_fast=False)
        self.coll  = DataCollatorWithPadding(self.tok, return_tensors="pt")
        self.enc   = (AutoModel
                      .from_pretrained(EMB_MODEL_NM,
                                       add_pooling_layer=False)
                      .to(device).eval())
        self.dec   = (AutoModel
                      .from_pretrained(DECOMP_MODEL_NM,
                                       trust_remote_code=True)
                      .to(device).eval())

    # ── public helper ──────────────────────────────────────────────
    @torch.no_grad()
    def encode_compress(self, smiles: Sequence[str]) -> torch.Tensor:
        """
        SMILES → compressed embedding  (shape [B, comp_size])  on **device**.
        """
        outs = []
        for chunk in [smiles[i:i+1024] for i in range(0, len(smiles), 1024)]:
            toks = self.coll(self.tok(chunk, truncation=True, max_length=256))
            toks = {k: v.to(self.device) for k, v in toks.items()}

            last = self.enc(**toks, output_hidden_states=True).hidden_states[-1]
            mask = toks["attention_mask"]
            z768 = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)

            zcmp = self.dec.compress(z768, [self.comp_size])[self.comp_size]
            outs.append(zcmp)

        return torch.cat(outs)                           # on device
