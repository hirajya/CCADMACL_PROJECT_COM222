from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def train_kmeans(X_scaled, n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-4):
    """
    Trains a KMeans model with the given parameters.

    Args:
        X_scaled: Scaled data.
        n_clusters: Number of clusters.
        init: Initialization method for KMeans.
        n_init: Number of times the K-means algorithm will be run with different centroid seeds.
        max_iter: Maximum number of iterations of the K-means algorithm for a single run.
        tol: Tolerance for the convergence of the algorithm.

    Returns:
        A trained KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, random_state=42)
    kmeans.fit(X_scaled)
    return kmeans

def evaluate_clustering(X, labels):
    """
    Evaluates clustering performance using various metrics.

    Args:
        X: Scaled data.
        labels: Cluster labels.

    Returns:
        A tuple containing silhouette score, Calinski-Harabasz score, and Davies-Bouldin score.
    """
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    return silhouette, calinski, davies_bouldin