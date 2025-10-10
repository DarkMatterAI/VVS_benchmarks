import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_compression import CompressionConfig

# pairwise cosine  ----------------------------------------------------------------
def pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    return x @ x.t()                              # [B, B]

# def cross_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     """Cosine similarity for all pairs between two sets: [m,d] x [n,d] -> [m,n]."""
#     a_n = F.normalize(a, p=2, dim=-1)
#     b_n = F.normalize(b, p=2, dim=-1)
#     return a_n @ b_n.T

# remove diagonal -----------------------------------------------------------------
def drop_diag(M: torch.Tensor) -> torch.Tensor:
    n = M.size(0)
    return M.masked_select(~torch.eye(n, dtype=torch.bool, device=M.device)).view(n, n - 1)

# # rank-weighted mse ---------------------------------------------------------------
# def rank_weighted_mse(ref: torch.Tensor, comp: torch.Tensor) -> torch.Tensor:
#     # weights = 1 / rank  (largest sim = rank 1)
#     ranks = ref.argsort(dim=1, descending=True).argsort(dim=1) + 1
#     weights = 1.0 / ranks.float()
#     return ((ref - comp).pow(2) * weights).mean()

# pearson row-wise ----------------------------------------------------------------
def rowwise_pearson(ref: torch.Tensor, comp: torch.Tensor, rm_diag: bool=True) -> torch.Tensor:
    if rm_diag:
        ref   = drop_diag(ref)
        comp  = drop_diag(comp)
    ref_z = F.normalize(ref  - ref.mean(dim=1, keepdim=True), p=2, dim=1)
    cmp_z = F.normalize(comp - comp.mean(dim=1, keepdim=True), p=2, dim=1)
    return 1 - (ref_z * cmp_z).sum(dim=1).mean()   #  0 = perfect corr

# aggregate loss ------------------------------------------------------------------
def compute_losses(
    embedding: torch.Tensor, # (batch_size, d)
    compressed: Dict[int, torch.Tensor], # Dict[size, (batch_size, size)]
    recon_stack: torch.Tensor | None, # (batch_size, n_heads, d)
    cfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (total_loss, terms_dict)"""
    device = embedding.device
    loss_total = torch.zeros((), device=device)
    terms: dict[str, float] = {}

    # ---- base similarities (detach to save mem) ---------------------------
    with torch.no_grad():
        base_sims = pairwise_cosine(embedding)
        ranks     = base_sims.argsort(-1, descending=True)

    # ======================================================================
    # 1) encoder / compressed losses
    # ======================================================================
    for size, z in compressed.items():
        tag = f"cmp{size}"
        comp_sims = pairwise_cosine(z)

        # plain MSE --------------------------------------------------------
        if cfg.mse_loss_weight:
            mse = F.mse_loss(drop_diag(base_sims), drop_diag(comp_sims))
            loss_total += cfg.mse_loss_weight * mse
            terms[f"{tag}_mse"] = mse.detach()

        # top-k MSE --------------------------------------------------------
        if cfg.topk_mse_loss_weight and cfg.topk_values:
            tk_vals = []
            for k in cfg.topk_values:
                idx = ranks[:, 1 : k + 1]
                ref_k = torch.gather(base_sims, 1, idx)
                cmp_k = torch.gather(comp_sims, 1, idx)
                tk_mse = F.mse_loss(ref_k, cmp_k)
                tk_vals.append(tk_mse)
                terms[f"{tag}_top{k}"] = tk_mse.detach()
            tk_agg = torch.stack(tk_vals).mean()
            loss_total += cfg.topk_mse_loss_weight * tk_agg
            terms[f"{tag}_topk_mean"] = tk_agg.detach()

        # # rank-weighted -----------------------------------------------------
        # if cfg.rank_mse_weight:
        #     rw = rank_weighted_mse(base_sims, comp_sims)
        #     loss_total += cfg.rank_mse_weight * rw
        #     terms[f"{tag}_rank"] = rw.detach()

        # Pearson ----------------------------------------------------------
        if cfg.pearson_loss_weight:
            pr = rowwise_pearson(base_sims, comp_sims)
            loss_total += cfg.pearson_loss_weight * pr
            terms[f"{tag}_pearson"] = pr.detach()

        if cfg.pearson_loss_weight and cfg.topk_values:
            pr_vals = []
            for k in cfg.topk_values:
                idx = ranks[:, 1 : k + 1]
                ref_k = torch.gather(base_sims, 1, idx)
                cmp_k = torch.gather(comp_sims, 1, idx)
                pr = rowwise_pearson(ref_k, cmp_k, rm_diag=False)
                pr_vals.append(pr)
                terms[f"{tag}_pearson_top{k}"] = pr.detach()
            pr_agg = torch.stack(pr_vals).sum()
            loss_total += cfg.pearson_loss_weight * pr_agg

        # # margin ranking ---------------------------------------------------
        # if cfg.margin_ranking_weight:
        #     strategy = cfg.margin_strategy
        #     k        = cfg.margin_k
        #     mrl      = nn.MarginRankingLoss(margin=cfg.margin, reduction="mean")

        #     if strategy == "top1-vs-median":
        #         pos = torch.gather(comp_sims, 1, ranks[:, 1:2])                     # [B,1]
        #         neg = torch.gather(comp_sims, 1,
        #                            ranks[:, comp_sims.size(1) // 2 : comp_sims.size(1) // 2 + 1])
        #         mrg = mrl(pos, neg, torch.ones_like(pos))

        #     elif strategy == "topK-avg":
        #         pos_idx = ranks[:, 1 : k + 1]                           # top-k indices (skip self)
        #         neg_idx = ranks.flip(-1)[:, :k]                         # bottom-k
        #         pos = torch.gather(comp_sims, 1, pos_idx).mean(1, keepdim=True)
        #         neg = torch.gather(comp_sims, 1, neg_idx).mean(1, keepdim=True)
        #         mrg = mrl(pos, neg, torch.ones_like(pos))

        #     elif strategy == "hard":
        #         hard_mask = ranks[:, 1 : k + 1]                         # base top-k
        #         hard_neg  = torch.gather(comp_sims, 1, hard_mask).min(1, keepdim=True).values
        #         pos       = torch.gather(comp_sims, 1, ranks[:, 1:2])
        #         mrg = mrl(pos, hard_neg, torch.ones_like(pos))

        #     elif strategy == "semi-hard":
        #         pos = torch.gather(comp_sims, 1, ranks[:, 1:2])         # [B,1]
        #         sims_sorted = torch.gather(comp_sims, 1, ranks)         # [B,B] sorted desc
        #         mask = (sims_sorted < pos) & (sims_sorted > pos - cfg.margin)
        #         cand = torch.where(mask, sims_sorted, sims_sorted.new_full((), -float("inf")))
        #         neg  = cand.max(dim=1, keepdim=True).values
        #         none = neg == -float("inf")
        #         neg  = torch.where(none, sims_sorted[:, -1:], neg)      # fallback lowest
        #         mrg = mrl(pos, neg, torch.ones_like(pos))

        #     else:
        #         raise ValueError(f"Unknown margin_strategy '{strategy}'")

        #     loss_total += cfg.margin_ranking_weight * mrg
        #     terms[f"{tag}_margin_{strategy}"] = mrg.detach()

    # ======================================================================
    # 2) decoder losses
    # ======================================================================
    if recon_stack is not None:
        # cosine -----------------------------------------------------------
        if cfg.decoder_cosine_weight:
            cos_loss = 1 - F.cosine_similarity(
                recon_stack,
                embedding.unsqueeze(1).expand_as(recon_stack),
                dim=-1,
            ).mean()
            loss_total += cfg.decoder_cosine_weight * cos_loss
            terms["dec_cosine"] = cos_loss.detach()

        # # pairwise sim MSE -------------------------------------------------
        # if cfg.decoder_pairwise_weight:
        #     recon_norm = F.normalize(recon_stack, p=2, dim=-1)
        #     recon_sim  = torch.bmm(
        #         recon_norm.permute(1, 0, 2), recon_norm.permute(1, 2, 0)
        #     )  # [n_sizes,B,B]
        #     pw = F.mse_loss(
        #         recon_sim,
        #         base_sims.unsqueeze(0).expand_as(recon_sim),
        #     )
        #     loss_total += cfg.decoder_pairwise_weight * pw
        #     terms["dec_pair"] = pw.detach()

    return loss_total, terms


# ─── basic blocks ───────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_out * 2)
        self.fc2 = nn.Linear(d_out, d_out)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        return self.fc2(F.silu(x1) * x2)

class FeedForwardLayer(nn.Module):
    def __init__(
        self, d_in: int, d_out: int, dropout: float = 0.1, layer_norm_eps: Optional[float] = 1e-12
    ):
        super().__init__()
        self.ff = FeedForward(d_in, d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = (
            nn.LayerNorm(d_out, eps=layer_norm_eps) 
            if layer_norm_eps is not None else nn.Identity()
        )

    def forward(self, x):
        y = self.ff(self.dropout(x)) + self.skip(x)
        return self.norm(y)


# ─── pure PyTorch compressor ────────────────────────────────────
class CompressionModel(nn.Module):
    """
    Encoder → (optional) Decoder.
    """

    def __init__(
        self,
        d_in: int,
        d_comp: int,
        encoder_layers: int,
        decoder_layers: int,
        dropout: float,
        layer_norm_eps: Optional[float],
    ):
        super().__init__()

        enc_layers: List[nn.Module] = []
        for i in range(encoder_layers):
            last = i == encoder_layers - 1
            enc_layers.append(
                FeedForwardLayer(
                    d_in,
                    d_comp if last else d_in,
                    dropout if not last else 0.0,
                    None if last else layer_norm_eps,
                )
            )
        self.encoder = nn.Sequential(*enc_layers)

        # optional decoder
        dec_layers: List[nn.Module] = []
        for i in range(decoder_layers):
            last = i == decoder_layers - 1
            d_prev = d_comp if i==0 else d_in
            dec_layers.append(
                FeedForwardLayer(
                    d_prev,
                    d_in,
                    dropout if not last else 0.0,
                    None if last else layer_norm_eps,
                )
            )
        self.decoder = nn.Sequential(*dec_layers) if dec_layers else None

    def forward(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = self.encoder(x)
        x_recon = self.decoder(z) if self.decoder is not None else None
        return z, x_recon


# ─── HF wrapper ─────────────────────────────────────────────────
@dataclass
class CompressionOutput(ModelOutput):
    loss: torch.FloatTensor
    loss_terms: Dict[str, torch.Tensor] | None = None
    compressed: Dict[int, torch.FloatTensor] | None = None
    reconstructed: torch.FloatTensor | None = None

class CompressionHFModel(PreTrainedModel):
    config_class = CompressionConfig

    def __init__(self, config: CompressionConfig):
        super().__init__(config)

        self.compressors = nn.ModuleDict(
            {
                str(size): CompressionModel(
                    d_in=config.input_size,
                    d_comp=size,
                    encoder_layers=config.encoder_layers,
                    decoder_layers=config.decoder_layers,
                    dropout=config.dropout,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for size in config.compression_sizes
            }
        )

        self.post_init()

    def get_encoders(self, unpack_single=False):
        encoders = {}
        for k,v in self.compressors.items():
            v = v.encoder
            if len(v)==1 and unpack_single:
                # unpack from nn.Sequential if only a single layer
                v = v[0]
            encoders[k] = v
        encoders = nn.ModuleDict(encoders)
        return encoders 
    
    def save_encoders(self, path, unpack_single=False):
        encoders = self.get_encoders(unpack_single)
        torch.save(encoders.state_dict(), path)
        
    def forward(self, embedding, return_dict=True, compute_loss=True):
        # ---------- forward passes ------------------------------------------------
        compressed, recons = {}, []
        for size, module in self.compressors.items():
            z, rec = module(embedding)
            compressed[int(size)] = z
            if rec is not None:
                recons.append(rec)
        recon_stack = torch.stack(recons, dim=1) if recons else None

        # ---------- losses --------------------------------------------------------
        if compute_loss:
            loss_total, terms = compute_losses(embedding, compressed, recon_stack, self.config)
        else:
            loss_total, terms = torch.zeros((), device=embedding.device), {}

        if not return_dict:
            return compressed, recon_stack, loss_total, terms

        return CompressionOutput(
            loss=loss_total,
            loss_terms=terms,
            compressed=compressed,
            reconstructed=recon_stack,
        )