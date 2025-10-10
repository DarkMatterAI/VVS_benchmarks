from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_decomposer import DecomposerConfig

def pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    """
    x : [B,d]  or  [N,B,d]
    returns a square similarity matrix:
      [B,B]  or  [N,B,B]
    """
    x = F.normalize(x, p=2, dim=-1)
    return torch.matmul(x, x.transpose(-1, -2))
    
def cross_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a : [M,d] or [N,M,d]
    b : [K,d]               (reference set - no extra axis)
    returns:
      [M,K]  or  [N,M,K]
    """
    a_n = F.normalize(a, 2, -1)
    b_n = F.normalize(b, 2, -1)

    if a.ndim == 2:          # [M,d]
        return a_n @ b_n.T                # [M,K]

    if a.ndim == 3:          # [N,M,d]
        return torch.einsum("n m d , k d -> n m k", a_n, b_n)  # [N,M,K]

    raise ValueError("cross_cosine: unexpected tensor rank.")
    
def _drop_diag(M: torch.Tensor) -> torch.Tensor:
    """
    Remove the main diagonal per similarity matrix.
    works for 2-D [B,B] or 3-D [N,B,B] tensors.
    """
    if M.ndim == 2:
        n = M.size(0)
        return M.masked_select(~torch.eye(n, dtype=torch.bool, device=M.device)
                              ).view(n, n - 1)

    if M.ndim == 3:
        n = M.size(1)
        mask = torch.eye(n, dtype=torch.bool, device=M.device).unsqueeze(0)  # [1,B,B]
        return M.masked_select(~mask).view(M.size(0), n, n - 1)
    
    raise ValueError("_drop_diag expects 2- or 3-D tensor")
    
    
def rowwise_pearson(ref: torch.Tensor,
                    pred: torch.Tensor,
                    *,
                    rm_diag: bool = True) -> torch.Tensor:
    """
    Pearson row-by-row; supports 2-D or 3-D inputs with identical shape.
    returns mean correlation error  (0 → perfect).
    """
    if rm_diag:
        ref  = _drop_diag(ref)
        pred = _drop_diag(pred)

    ref_z  = F.normalize(ref  - ref.mean(-1, keepdim=True), p=2, dim=-1)
    pred_z = F.normalize(pred - pred.mean(-1, keepdim=True), p=2, dim=-1)
    loss = 1 - (ref_z * pred_z).sum(-1).mean(-1)
    if loss.ndim==0:
        loss = loss.unsqueeze(0)
    return loss 

def similarity_mse(ref: torch.Tensor,
                   pred: torch.Tensor,
                   *,
                   rm_diag: bool = True) -> torch.Tensor:
    if rm_diag:
        ref, pred = _drop_diag(ref), _drop_diag(pred)
    
    if pred.ndim==2:
        loss = F.mse_loss(pred, ref).mean().unsqueeze(0)
    elif pred.ndim==3:
        loss = F.mse_loss(pred, 
                          ref.expand_as(pred), 
                          reduction="none"
                         ).reshape(pred.size(0), -1).mean(-1)
        
    return loss


def sim_loss(pred:  torch.Tensor,       # [N,B,d]   or [B,d]
             targ:  torch.Tensor,       # [B,d]     (ground truth)
             ref:   Optional[torch.Tensor],
             k_vals: Optional[List[int]],
             loss_type: str = "pearson") -> torch.Tensor:
    """
    Returns stacked tensor of losses:
        len = 1 + len(k_vals)
    If `ref` is given we compute cross-similarities pred↔ref / targ↔ref,
    otherwise self-similarities pred↔pred / targ↔targ.
    """

    loss_fn = rowwise_pearson if loss_type == "pearson" else similarity_mse

    if ref is None:                         # self-sim
        p_sim, t_sim = pairwise_cosine(pred), pairwise_cosine(targ)
        rm_diag      = True
    else:                                   # cross-sim vs fixed reference
        p_sim, t_sim = cross_cosine(pred, ref), cross_cosine(targ, ref)
        rm_diag      = False

    losses = [loss_fn(t_sim, p_sim, rm_diag=rm_diag)]

    if k_vals:
        # ranks based on target sims (works for 2- or 3-D)
        ranks = t_sim.argsort(-1, descending=True)
        start = 1 if rm_diag else 0
        for k in k_vals:
            idx = ranks[..., start:start + k]
            t_k = torch.gather(t_sim, -1, idx)
            if p_sim.ndim==2:
                p_k = torch.gather(p_sim, -1, idx)
            elif p_sim.ndim==3:
                p_k = torch.gather(p_sim, -1, idx.repeat(p_sim.size(0), 1, 1))
            losses.append(loss_fn(t_k, p_k, rm_diag=False))

    return torch.stack(losses, 1)              # shape [n_losses]


# ─────────────────────────────── building blocks ──────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_out * 2)
        self.fc2 = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.fc1(x).chunk(2, -1)
        return self.fc2(F.silu(x1) * x2)


class FeedForwardLayer(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 *,
                 dropout: float = .1,
                 ln_eps: Optional[float] = 1e-12):
        super().__init__()
        self.ff   = FeedForward(d_in, d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out, eps=ln_eps) if ln_eps else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.ff(self.drop(x)) + self.skip(x))
    
class OutputLinear(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 n_head_layers: int,
                 n_output: int, 
                 output_sizes: List[int], 
                 dropout: float=0.1,
                 ln_eps: Optional[float] = 1e-12):
        super().__init__()
        self.n_output = n_output
        ff_layers = [FeedForwardLayer(input_size, input_size, dropout=dropout, 
                                      ln_eps=None if i==n_head_layers-1 else ln_eps)
                     for i in range(n_head_layers)]
        self.ff = nn.Sequential(*ff_layers)
        self.layers = nn.ModuleDict({str(d): nn.Linear(input_size, d*n_output) 
                                     for d in output_sizes})
        
    def forward(self, inputs: torch.Tensor, sizes: List[int]):
        inputs = self.ff(inputs)
        weights = torch.cat([self.layers[str(i)].weight for i in sizes])
        biases = torch.cat([self.layers[str(i)].bias for i in sizes])
        outputs = F.linear(inputs, weights, biases)
        output_dict = {}
        current = 0
        
        slice_sizes = [d*self.n_output for d in sizes]
        for size in slice_sizes:
            p = outputs[:, :, current:current+size]
            p = p.view(p.size(0), p.size(1), self.n_output, size//self.n_output)
            output_dict[size//self.n_output] = p
            current += size
        return output_dict

def get_compression_heads(d_in, comp_sizes, n_layers, add_input_identity=False):
    compression_heads = nn.ModuleDict({})
    for d in comp_sizes:
        enc_layers = []
        for i in range(n_layers):
            last = i == n_layers - 1
            enc_layers.append(
                FeedForwardLayer(
                    d_in,
                    d if last else d_in,
                    dropout=0.0,
                    ln_eps=None if last else 1e-12,
                )
            )
        compression_heads[str(d)] = nn.Sequential(*enc_layers)
    if add_input_identity:
        compression_heads[str(d_in)] = nn.Identity()

    return compression_heads 

# ───────────────────────────── output dataclass ───────────────────────────────
@dataclass
class DecomposerOutput(ModelOutput):
    loss:        torch.FloatTensor
    loss_terms:  Optional[Dict[str, torch.Tensor]] = None
    decomp:      Optional[Dict[int, torch.FloatTensor]] = None  # {size:[B,2,size]}
    ref_idxs:    Optional[torch.LongTensor] = None


# ──────────────────────────────── main model ──────────────────────────────────
class DecomposerModel(PreTrainedModel):
    """Maps an embedding to *n_output* building-block embeddings for every
    requested `output_size`. All loops are left intact for clarity."""
    config_class = DecomposerConfig

    # ---------------------------------------------------------------- init
    def __init__(self, config: DecomposerConfig):
        super().__init__(config)
        
        # compression heads to avoid needing to save all embedding sizes for training
        self.compression_heads = get_compression_heads(config.input_size, 
                                                       config.comp_sizes, 
                                                       config.n_comp_layers,
                                                       add_input_identity=True)
        # input → shared_dim
        self.in_proj = nn.ModuleDict({
            str(d): FeedForwardLayer(d, config.shared_dim, 
                                     dropout=config.dropout, 
                                     ln_eps=config.layer_norm_eps)
            for d in config.comp_sizes
        })

        # shared trunk
        blk = lambda: FeedForwardLayer(config.shared_dim,
                                       config.shared_dim,
                                       dropout=config.dropout,
                                       ln_eps=config.layer_norm_eps)
        self.trunk = nn.Sequential(*[blk() for _ in range(config.n_shared_layers)])

        # shared_dim → each output size x n_output
        self.out_proj = OutputLinear(self.config.shared_dim, 
                                     self.config.n_head_layers,
                                     config.n_output, 
                                     config.output_sizes,
                                     config.dropout,
                                     config.layer_norm_eps)

        # reference embeddings (optional corr-loss)
        self.ref_emb = nn.ModuleDict({
            str(d): nn.Embedding(config.n_refs_total, d)
            for d in config.output_sizes if config.n_refs_total
        })

        self.post_init()

    # ---------------------------------------------------------------- forward
    def compress(self,
                 inputs: torch.Tensor,                   # {size: [B,size]}
                 comp_sizes: List[int]):
        compressed = {d: self.compression_heads[str(d)](inputs) for d in comp_sizes}
        return compressed
    
    # def decompose(self, 
    #               inputs:  Dict[int, torch.Tensor],           # {size: [B,size]}
    #               output_sizes: List[int]):
    #     hiddens = []
    #     for input_size in self.config.comp_sizes:
    #         if input_size not in inputs:
    #             continue 

    #         h = self.in_proj[str(input_size)](x)  # [B,shared_dim]
    #         hiddens.append(h)

    #     # for in_size, x in inputs.items():
    #     #     h = self.in_proj[str(in_size)](x)  # [B,shared_dim]
    #     #     hiddens.append(h)
            
    #     hiddens = torch.stack(hiddens, dim=0) # [n_sizes, B, shared_dim]
    #     hiddens = self.trunk(hiddens)
        
    #     preds = self.out_proj(hiddens, output_sizes) # {size: [n_sizes, B, n_output, size]}
    #     return preds

    def decompose(self, 
                  inputs:  Dict[int, torch.Tensor],           # {size: [B,size]}
                  output_sizes: List[int]):
        hiddens = []
        for input_size in self.config.comp_sizes:
            if input_size not in inputs:
                continue 

            h = self.in_proj[str(input_size)](inputs[input_size])  # [B,shared_dim]
            hiddens.append(h)
            
        hiddens = torch.stack(hiddens, dim=0) # [n_sizes, B, shared_dim]
        hiddens = self.trunk(hiddens)
        
        preds = self.out_proj(hiddens, output_sizes) # {size: [n_sizes, B, n_output, size]}
        return preds
    
    def load_targets(self,
                     bb1_ids: torch.LongTensor,                  # [B,]
                     bb2_ids: torch.LongTensor):                 # [B,]
        targets = {}
        for size in self.config.output_sizes:
            embedding = self.ref_emb[str(size)]
            targets[size] = torch.stack([embedding(bb1_ids), embedding(bb2_ids)], dim=1)
        return targets
    
    def compute_loss(self,
                     inputs: Dict[int, torch.Tensor],
                     preds: Dict[int, torch.Tensor],
                     targets: Dict[int, torch.Tensor],
                     ref_idxs: Optional[torch.LongTensor]=None,):
        device = next(iter(preds.values())).device
        loss_terms: Dict[str, torch.Tensor]  = {}
        loss_total  = torch.zeros((), device=device)
        cfg = self.config
        for out_size in cfg.output_sizes:
            p = preds[out_size]
            t = targets[out_size]                                    # [B, n_out, d]

            # 1) cosine to target ------------------------------------
            if cfg.cosine_weight>0:
                cos = 1 - F.cosine_similarity(p, t, dim=-1).view(p.size(0), -1).mean(-1)
                loss_total += cfg.cosine_weight * cos.sum()
                for i, in_size in enumerate(cfg.comp_sizes):
                    loss_terms[f"{in_size}->{out_size}_cos"] = cos[i]
                    
            # 2) mse to target ---------------------------------------
            if cfg.mse_weight>0:
                mse = F.mse_loss(p, t.expand_as(p), reduction="none").view(p.size(0), -1).mean(-1)
                loss_total += cfg.mse_weight * mse.sum()
                for i, in_size in enumerate(cfg.comp_sizes):
                    loss_terms[f"{in_size}->{out_size}_mse"] = mse[i]
                    
            # 3) correlation losses ----------------------------------
            if cfg.corr_weight:
                flat_p = p.flatten(1, 2)
                flat_t = t.flatten(0, 1)

                if cfg.ref_corr:
                    with torch.no_grad():
                        ref = self.ref_emb[str(out_size)](ref_idxs)
                        
                    ref_corr = sim_loss(flat_p, flat_t, ref,
                                            cfg.corr_k_vals, cfg.corr_loss_type).mean(-1)
                    loss_total += cfg.corr_weight * ref_corr.sum()
                    for i, in_size in enumerate(cfg.comp_sizes):
                        loss_terms[f"{in_size}->{out_size}_corr_ref"] = ref_corr[i]

                # if cfg.input_corr and out_size in inputs:
                #     ref = inputs[out_size]
                #     ref_corr = sim_loss(flat_p, flat_t, ref,
                #                             cfg.corr_k_vals, cfg.corr_loss_type).mean(-1)
                #     loss_total += cfg.corr_weight * ref_corr.sum()
                #     for i, in_size in enumerate(cfg.comp_sizes):
                #         loss_terms[f"{in_size}->{out_size}_corr_input"] = ref_corr[i]
        return loss_total, loss_terms 

    def forward(self,
                embedding:  torch.Tensor,                      # [B,size]
                bb1_id: torch.LongTensor,                  # [B,]
                bb2_id: torch.LongTensor,                  # [B,]
                *,
                ref_idxs: Optional[torch.LongTensor]=None,
                return_preds: bool = False,
                compute_loss: bool = True,
                return_dict: bool  = True) -> DecomposerOutput: # | tuple:
        
        cfg        = self.config
        device     = embedding.device
        targets    = self.load_targets(bb1_id, bb2_id)

        if cfg.corr_weight and cfg.n_refs_total and ref_idxs is None:
            ref_idxs = torch.randint(cfg.n_refs_total,
                                     (cfg.n_refs_batch,),
                                     device=device)

        loss_terms: Dict[str, torch.Tensor]  = {}
        loss_total  = torch.zeros((), device=device) if compute_loss else None
        
        with torch.no_grad():
            compressed_inputs = self.compress(embedding, cfg.comp_sizes)
        
        if cfg.input_size in cfg.comp_sizes:
            compressed_inputs[cfg.input_size] = embedding
        
        preds = self.decompose(compressed_inputs, cfg.output_sizes)
        
        loss_total = None 
        loss_terms = {}
        if compute_loss:
            loss_total, loss_terms = self.compute_loss(compressed_inputs, preds, targets, ref_idxs)
            
        decomp = {k:v.permute(1,0,2,3) for k,v in preds.items()}

        return DecomposerOutput(loss        = loss_total,
                                loss_terms  = loss_terms,
                                decomp      = decomp,
                                ref_idxs    = ref_idxs)
            
            