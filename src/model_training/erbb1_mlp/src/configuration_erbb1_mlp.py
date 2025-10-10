from transformers import PretrainedConfig


class Erbb1MlpConfig(PretrainedConfig):
    model_type = "erbb1_mlp"

    def __init__(
        self,
        d_in: int = 768,
        d_hidden: int = 1024,
        n_layers: int = 6,
        dropout: float = 0.1,
        layer_norm_eps: float | None = 1e-12,
        dataset_mean: float | None = None,
        dataset_std: float | None = None,
        **kwargs,
    ):
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        super().__init__(**kwargs)
