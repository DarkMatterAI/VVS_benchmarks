import torch 

def cosine_gradient(query_embedding: torch.Tensor, # shape (1, d)
                    embeddings: torch.Tensor, # shape (n, d)
                    scores: torch.Tensor, # shape (n, )
                   ) -> torch.Tensor:
    """Estimates the gradient at `query_embedding` using 
    `embeddings` and `advantages` using cosine distance.
    
    The gradient is computed directly using numpy. This 
    is the equivalent gradient to computing the following in pytorch
    
    ```
    # assume `query_embedding` has `requires_grad=True`
    advantages = (scores - scores.mean()) / (scores.std() + 1e-8)
    distance = 1 - torch.cosine_similarity(query_embedding, embeddings, dim=1)
    loss = (advantages * distance).mean()
    loss.backward()
    ```
    """
    advantages = (scores - scores.mean()) / (scores.std() + 1e-8)    
    query_norm = torch.norm(query_embedding, p=2, dim=-1)
    embedding_norms = torch.norm(embeddings, p=2, dim=-1)
    denom = query_norm * embedding_norms
    dot_product = (embeddings * query_embedding).sum(-1)
    adv_denom = advantages / denom
    
    term1 = (embeddings * adv_denom[:,None]).mean(0)
    term2 = (dot_product * adv_denom).mean() * (query_embedding[0] / query_norm.pow(2))
    grad = term2 - term1
    return grad
