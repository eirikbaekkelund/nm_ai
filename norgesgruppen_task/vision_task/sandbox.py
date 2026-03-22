import torch
from typing import List, Union


@torch.no_grad()
def identify_product(
    crop_tensor: torch.Tensor, model: torch.nn.Module, ref_embeddings: torch.Tensor, ref_labels: List[Union[int, str]]
) -> Union[int, str]:
    model.eval()
    query_feat = model(crop_tensor)  # [1, 768]

    # Normalize for cosine similarity
    query_feat = torch.nn.functional.normalize(query_feat, p=2, dim=1)
    ref_embeddings = torch.nn.functional.normalize(ref_embeddings, p=2, dim=1)

    # Simple dot product on normalized vectors = Cosine Similarity
    scores = torch.mm(query_feat, ref_embeddings.t())
    best_idx = torch.argmax(scores)

    return ref_labels[best_idx]


if __name__ == "__main__":
    raise NotImplementedError(
        "This is a sandbox file for testing code snippets. Please run the main training and evaluation scripts instead."
    )
