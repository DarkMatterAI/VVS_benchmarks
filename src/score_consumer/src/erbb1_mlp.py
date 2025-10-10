import torch
from transformers import AutoModel 
from sentence_transformers import models, SentenceTransformer
from .utils import log

EMBEDDING_NAME = "entropy/roberta_zinc_480m"
MLP_NAME = "entropy/erbb1_mlp"

class ErbB1MLP():
    def __init__(self):
        self.roberta_zinc = None 
        self.erbb1_mlp = None 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_models(self):
        self._load_roberta_zinc()
        self._load_erbb1_mlp()

    def _load_roberta_zinc(self):
        if self.roberta_zinc is not None:
            return 
        
        transformer = models.Transformer(EMBEDDING_NAME, 256,
                                         model_args={"add_pooling_layer": False})
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), 
                                 pooling_mode="mean")
        self.roberta_zinc = SentenceTransformer(modules=[transformer, pooling],
                                                device=self.device)
        self.roberta_zinc.eval()

    def _load_erbb1_mlp(self):
        if self.erbb1_mlp is not None:
            return 
        self.erbb1_mlp = AutoModel.from_pretrained(MLP_NAME, trust_remote_code=True)
        self.erbb1_mlp.eval()
        self.erbb1_mlp.to(self.device)

    def __call__(self, request_batch):
        """
        Parameters
        ----------
        request_batch : List[dict]  - each dict has at least key 'item' = SMILES

        Returns
        -------
        List[dict]  - same length, each with keys {valid, score, data}
        """
        self.load_models()
        inputs = [d["item_data"]["item"] for d in request_batch]

        with torch.inference_mode(), torch.autocast(device_type=self.device):
            embeddings = self.roberta_zinc.encode(inputs, 
                                                  batch_size=1024,
                                                  show_progress_bar=False,
                                                  convert_to_tensor=True)
            predictions = self.erbb1_mlp(embeddings).prediction.detach().cpu().numpy()

        out = [
            {
                "valid": True,
                "score": float(predictions[i]),
            }
            for i in range(len(request_batch))
        ]
        return out

ERBB1 = ErbB1MLP()
