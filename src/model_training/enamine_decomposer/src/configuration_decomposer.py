from typing import List, Optional
from transformers import PretrainedConfig


class DecomposerConfig(PretrainedConfig):
    """
    Config for the embedding-decomposition model.

    Args:
        input_size (int):          input embedding size
        comp_sizes (List[int]):    compressed embedding sizes
        output_sizes (List[int]):  desired output dims (for the two blocks).
        shared_dim (int):          common hidden size after input projection.
        n_shared_layers (int):     how many FeedForwardLayers in shared trunk.
        dropout (float):           dropout prob in *every* non-final layer.
        layer_norm_eps (float|None): epsilon for LayerNorm (None → no LN).
        n_output (int):            number of output embeddings.
        n_refs_batch (int):        number of reference embeddings to sample per batch
        n_refs_total (int):        number of reference embeddings total
        cosine_weight (float):     weight of 1-1 cosine similarity loss
        mse_weight (float):        weight of 1-1 mse loss
        corr_weight (float):       pairwise correlation loss weight
        self_corr (bool):          compute self-correlation loss
        ref_corr (bool):           compute self-to-reference loss
        input_corr (bool):         compute self-to-input loss 
        corr_loss_type (str):      correlation loss type - "pearson" or "mse"
        corr_k_vals (List[int]):   k-vals for weighting correlation loss
    """
    model_type = "embedding_decomposer"

    def __init__(
        self,
        input_size:      int = 768,
        comp_sizes:      List[int] = (768, 512, 256, 128, 64, 32),
        output_sizes:    List[int] = (768, 512, 256, 128, 64, 32),
        n_comp_layers:   int       = 4,
        shared_dim:      int       = 1024,
        n_shared_layers: int       = 8,
        n_head_layers:   int       = 1,
        dropout:         float     = 0.1,
        layer_norm_eps:  Optional[float] = 1e-12,
        n_output:        int       = 2,
        n_refs_batch:    int       = 128,
        n_refs_total:    int       = 2000,
        cosine_weight:   float     = 1.0,
        mse_weight:      float     = 1.0, 
        self_corr:       bool      = True,
        ref_corr:        bool      = True,
        input_corr:      bool      = True,
        corr_weight:     float     = 1.0,
        corr_loss_type:  str       = "pearson", # "pearson" or "mse"
        corr_k_vals:     List[int] = [10, 100],
        **kwargs,
    ):
        self.input_size      = input_size
        self.comp_sizes      = list(comp_sizes)
        self.output_sizes    = list(output_sizes)
        self.n_comp_layers   = n_comp_layers
        self.shared_dim      = shared_dim
        self.n_shared_layers = n_shared_layers
        self.n_head_layers   = n_head_layers
        self.dropout         = dropout
        self.layer_norm_eps  = layer_norm_eps
        self.n_output        = n_output
        self.n_refs_batch    = n_refs_batch
        self.n_refs_total    = n_refs_total
        self.cosine_weight   = cosine_weight
        self.mse_weight      = mse_weight
        self.self_corr       = self_corr
        self.ref_corr        = ref_corr
        self.input_corr      = input_corr
        self.corr_weight     = corr_weight
        self.corr_loss_type  = corr_loss_type
        self.corr_k_vals     = corr_k_vals
        super().__init__(**kwargs)
