from _1_data_preprocessing import load_data, preprocess_data
from _2_model import train_kmeans, evaluate_clustering

def train_and_evaluate():
    """
    Loads data, preprocesses it, trains the KMeans model, and evaluates the model.

    Returns:
        A tuple containing:
            - Trained KMeans model
            - MinMaxScaler (used for preprocessing input during evaluation)
    """
    df = load_data()
    X, scalers, scaled_data = preprocess_data(df)

    # Use MinMaxScaler data for training
    X_MMscaled_data = scaled_data['MinMaxScaler']

    # Train KMeans with the specified hyperparameters (from Optuna)
    best_kmeans_params = {
        'n_clusters': 3,
        'init': 'k-means++',
        'n_init': 7,
        'max_iter': 500,
        'tol': 0.0007009625455530479
    }
    kmeans_model = train_kmeans(X_MMscaled_data, **best_kmeans_params)

    # Get cluster labels
    labels = kmeans_model.labels_

    # Evaluate the model
    silhouette, calinski, davies_bouldin = evaluate_clustering(X_MMscaled_data, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Calinski-Harabasz Score: {calinski:.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")

    return kmeans_model, scalers["MinMaxScaler"]
