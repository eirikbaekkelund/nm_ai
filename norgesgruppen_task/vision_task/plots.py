import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_embeddings(embeddings, labels, class_names, save_path="plots/tsne_latest.png"):
    """
    Visualizes high-dim DINOv2 embeddings in 2D.
    embeddings: Tensor [N, 768]
    labels: Tensor [N]
    """
    # Move to CPU and numpy
    features = embeddings.detach().cpu().numpy()
    target_labels = labels.detach().cpu().numpy()

    # Run T-SNE (Set perplexity based on sample size, usually 30-50)
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    embeds_2d = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    # Filter for top 10-20 classes to avoid a "hairball" plot
    unique_labels = np.unique(target_labels)[:15]

    for label in unique_labels:
        mask = target_labels == label
        plt.scatter(embeds_2d[mask, 0], embeds_2d[mask, 1], label=class_names[label], alpha=0.7)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("DINOv2 + ArcFace: Product Embedding Clusters")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
