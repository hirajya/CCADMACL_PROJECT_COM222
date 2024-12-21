from _1_data_preprocessing import load_data, preprocess_data
from _2_model import train_kmeans, evaluate_clustering
from _3_train_model import train_and_evaluate
from _4_evaluate_model import evaluate_model_on_inputs


def run_clustering_pipeline(data_path="../../data/raw/Country-data.csv"):
    """
    Executes the entire clustering pipeline.

    Args:
        data_path: Path to the CSV file containing the dataset.
    """

    # Train and evaluate the KMeans model
    print("Training and evaluating KMeans model...")
    trained_model, scaler = train_and_evaluate()  # Unpack the tuple
    print("KMeans model trained and evaluated successfully.")

    # Evaluate the model on example inputs
    print("\nEvaluating the model on example input values...")
    evaluate_model_on_inputs(trained_model, scaler)  # Pass the model and scaler
    print("Model evaluation completed.")

    print("\nClustering pipeline completed.")



if __name__ == "__main__":
    print("Running clustering pipeline...")
    run_clustering_pipeline()
