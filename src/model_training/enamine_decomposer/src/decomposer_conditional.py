import torch
import torch.nn as nn 
import random 
from typing import Optional, Dict 
from transformers import default_data_collator

from .configuration_decomposer import DecomposerConfig
from .modeling_decomposer import DecomposerModel, FeedForwardLayer, DecomposerOutput

REACTION_IDS = [11, 22, 27, 40, 527, 1458, 2230, 2430, 2708, 2718, 240690, 271948]
REACTION_ID_TO_IDX = {rid: i for i, rid in enumerate(REACTION_IDS)}
IDX_TO_REACTION_ID = {i: rid for rid, i in REACTION_ID_TO_IDX.items()}
N_REACTIONS = len(REACTION_IDS)

class ConditionalDecomposerCollator:
    def __init__(self, reaction_id_map: dict):
        self.reaction_id_map = reaction_id_map

    def __call__(self, batch: list[dict]) -> dict:
        # batch is a list of dicts, one per example
        reaction_ids = []
        for ex in batch:
            rids = ex.pop("reaction_ids")          # variable-length list
            if isinstance(rids, torch.Tensor):
                rids = rids.tolist()
            chosen = random.choice(rids)
            reaction_ids.append(self.reaction_id_map[int(chosen)])
        
        # default-stack the remaining fixed-size fields
        collated = default_data_collator(batch)
        collated["reaction_ids"] = torch.tensor(reaction_ids, dtype=torch.long)
        return collated

class ConditionalDecomposerConfig(DecomposerConfig):
    model_type = "conditional_embedding_decomposer"

    def __init__(self, *, n_reactions=12, reaction_dim=64,
                 reaction_id_map=None, **kwargs):
        self.n_reactions = n_reactions
        self.reaction_dim = reaction_dim
        self.reaction_id_map = reaction_id_map or REACTION_ID_TO_IDX
        super().__init__(**kwargs)

class ConditionalDecomposerModel(DecomposerModel):
    config_class = ConditionalDecomposerConfig

    def __init__(self, config):
        super().__init__(config)
        self.reaction_emb  = nn.Embedding(config.n_reactions, config.reaction_dim)
        # self.reaction_proj = FeedForwardLayer(config.reaction_dim, 
        #                                       config.shared_dim,
        #                                       dropout=config.dropout,
        #                                       ln_eps=config.layer_norm_eps)
        self.reaction_proj = FeedForwardLayer(config.reaction_dim + config.shared_dim, 
                                              config.shared_dim,
                                              dropout=config.dropout,
                                              ln_eps=config.layer_norm_eps)
        self.inference_id = None 
        self.post_init()

    def decompose(self, inputs, output_sizes, reaction_ids=None):
        hiddens = []
        for input_size in self.config.comp_sizes:
            if input_size not in inputs:
                continue
            h = self.in_proj[str(input_size)](inputs[input_size])   # [B, shared_dim]
            hiddens.append(h)

        hiddens = torch.stack(hiddens, dim=0)  # [n_sizes, B, shared_dim]

        if (reaction_ids is not None) and (self.inference_id is not None):
            reaction_ids = torch.zeros(hiddens.size(1), 
                                      device=hiddens.device, 
                                      dtype=torch.long) + self.inference_id

        # ── conditioning ──
        # if reaction_ids is not None:
        #     r = self.reaction_emb(reaction_ids)       # [B, reaction_dim]
        #     r = self.reaction_proj(r)                 # [B, shared_dim]
        #     hiddens = hiddens + r.unsqueeze(0)        # broadcast over n_sizes dim
        if reaction_ids is not None:
            r = self.reaction_emb(reaction_ids)       # [B, reaction_dim]
            r = r.repeat(hiddens.size(0), 1, 1)       # [n_sizes, B, reaction_dim]
            hiddens = torch.cat((hiddens, r), -1)     # [n_sizes, B, shared_dim+reaction_dim]
            hiddens = self.reaction_proj(hiddens)     # [n_sizes, B, shared_dim]

        hiddens = self.trunk(hiddens)
        preds = self.out_proj(hiddens, output_sizes)
        return preds


    def forward(self,
                embedding:  torch.Tensor,                      # [B,size]
                reaction_ids: torch.LongTensor,                  # [B,]
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
        
        preds = self.decompose(compressed_inputs, cfg.output_sizes, reaction_ids)
        
        loss_total = None 
        loss_terms = {}
        if compute_loss:
            loss_total, loss_terms = self.compute_loss(compressed_inputs, preds, targets, ref_idxs)
            
        decomp = {k:v.permute(1,0,2,3) for k,v in preds.items()}

        return DecomposerOutput(loss        = loss_total,
                                loss_terms  = loss_terms,
                                decomp      = decomp,
                                ref_idxs    = ref_idxs)
            
            