# 2. Train Models

**Prerequisites:** [Download Datasets and Create Artifacts](01_datasets.md)

Three classes of models are trained in this repo. Pretrained weights for each are
available for download if you wish to skip training.

## EGFR MLP

An MLP predicting EGFR binding affinity from ChEMBL IC50 data. This model is used
as a target score function in downstream benchmarks.

```bash
cd src/model_training/erbb1_mlp
./process_dataset.sh
./train_erbb1_mlp.sh
```

> **Pretrained weights:** [HuggingFace](https://huggingface.co/entropy/erbb1_mlp)

## Embedding Compression Models

Embedding Compression models compress chemical language model (CLM) embeddings to
lower dimensions.

```bash
cd src/model_training/embedding_compression
./process_dataset.sh
./train_model1.sh
```

> **Note:** `process_dataset.sh` pre-computes CLM embeddings, which is slow and
> disk-intensive. Processing both datasets requires ~1000 GB of disk space and
> yields ~500 GB of files.

Hyperparameters can be configured in
`src/model_training/embedding_compression/sweep_space1.yaml`. See
`src/model_training/embedding_compression/src/train.py` for the full list of
supported training arguments.

> **Pretrained weights:** [HuggingFace](https://huggingface.co/entropy/roberta_zinc_compression_encoder)

## Embedding Decomposer Models

Embedding Decomposer models predict building block embeddings from product molecule
embeddings.

```bash
cd src/model_training/enamine_decomposer
./process_dataset.sh
./train_model.sh
```

> **Note:** `process_dataset.sh` pre-computes CLM embeddings, which is slow and
> disk-intensive. Processing both datasets requires ~1000 GB of disk space and
> yields ~500 GB of files.

> **Note:** The training code assumes the Embedding Compression models have already
> been trained and saved to the relevant `blob_store` location (done automatically
> by the Embedding Compression training code). To run without training the
> compression models first, download the pretrained weights to the correct
> location:
>
> ```python
> from transformers import AutoModel
>
> compression_encoder = AutoModel.from_pretrained(
>     "entropy/roberta_zinc_compression_encoder",
>     trust_remote_code=True,
> )
>
> SAVE_DIR = "./src/blob_store/internal/model_weights/compression_heads.pt"
> compression_encoder.save_encoders(SAVE_DIR)
> ```

Hyperparameters can be configured in
`src/model_training/enamine_decomposer/sweep_space.yaml`. See
`src/model_training/enamine_decomposer/src/train.py` for the full list of supported
training arguments.

> **Pretrained weights:** The final decomposer model is available on
> [HuggingFace](https://huggingface.co/entropy/roberta_zinc_compression_encoder).
> Weights of the decomposer models trained for the loss ablation experiments are
> available in the [Zenodo data archive](https://zenodo.org/records/18615633).