import numpy as np
from sklearn.metrics import pairwise_distances

def knn_uniqueness(embeddings: np.ndarray, num_neighbors: int) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on knn.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group
    num_neighbors: int :
        number of neighbors to estimate uniqueness

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    # Calculate pairwise Euclidean distances between all embeddings
    distances = pairwise_distances(embeddings, embeddings, metric='euclidean')

    # Sort the distances to find the nearest neighbors for each embedding
    sorted_distances = np.sort(distances, axis=1)

    # Calculate the uniqueness score for each item
    uniqueness_scores = np.mean(sorted_distances[:, 1:num_neighbors + 1], axis=1)

    return uniqueness_scores
