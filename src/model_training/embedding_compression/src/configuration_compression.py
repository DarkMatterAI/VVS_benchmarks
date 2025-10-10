from typing import List, Optional
from transformers import PretrainedConfig


class CompressionConfig(PretrainedConfig):
    """
    Configuration for the embedding-compression models.

    Args:
        input_size (int):  Dimension of the input embedding.
        compression_sizes (List[int]):  One or more output dimensions.
        encoder_layers (int):  Number of FeedForwardLayers in the encoder path.
        decoder_layers (int):  Number of FeedForwardLayers in the optional decoder.
        dropout (float):  Drop-out prob in every layer except the final ones.
        layer_norm_eps (float | None):  Epsilon for LayerNorm.
    """
    model_type = "embedding_compression"

    def __init__(
        self,
        # ── model params ─────────────────────────────────────────────
        input_size: int = 768,
        compression_sizes: List[int] = (128, 64),
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        dropout: float = 0.1,
        layer_norm_eps: Optional[float] = 1e-12,
        # ── loss knobs ───────────────────────────────────────────────
        mse_loss_weight:          float = 0.0,
        topk_mse_loss_weight:     float = 0.0,
        topk_values:              list[int] = (10, 100),
        rank_mse_weight:          float = 0.0,              # smooth rank-weighted MSE
        pearson_loss_weight:      float = 0.0,
        margin_ranking_weight:    float = 0.0,
        margin:                   float = 0.2,              # for MarginRankingLoss
        margin_strategy:          str = "top1-vs-median",   # ["top1-vs-median", "topK-avg", "hard", "semi-hard"]
        margin_k:                 int = 5,                  # used by topK-avg / hard / semi-hard
        decoder_cosine_weight:    float = 0.0,
        decoder_pairwise_weight:  float = 0.0,
        **kwargs,
    ):
        self.input_size = input_size
        self.compression_sizes = list(compression_sizes)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.mse_loss_weight = mse_loss_weight
        self.topk_mse_loss_weight = topk_mse_loss_weight
        self.topk_values = topk_values
        self.rank_mse_weight = rank_mse_weight
        self.pearson_loss_weight = pearson_loss_weight
        self.margin_ranking_weight = margin_ranking_weight
        self.margin = margin
        self.margin_k = margin_k
        self.margin_strategy = margin_strategy
        self.decoder_cosine_weight = decoder_cosine_weight
        self.decoder_pairwise_weight = decoder_pairwise_weight
        super().__init__(**kwargs)