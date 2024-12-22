from _1_data_preprocessing import load_data, preprocess_data
from _2_model import train_kmeans, evaluate_clustering
from _3_train_model import train_and_evaluate
from _4_evaluate_model import evaluate_model_on_inputs
import joblib  # For saving the model and scaler


def run_clustering_pipeline(data_path="../../data/raw/Country-data.csv"):
    """
    Executes the entire clustering pipeline.

    Args:
        data_path: Path to the CSV file containing the dataset.
    """

    # Train and evaluate the KMeans model
    print("Training and evaluating KMeans model...")
    trained_model, scaler = train_and_evaluate()  # Unpack the tuple
    print("KMeans model trained and evaluated successfully.\n")

    # Export the trained model and scaler
    export_model_and_scaler(trained_model, scaler)
    print("Model and scaler exported successfully.\n")

    # Evaluate the model on example inputs
    print("\nEvaluating the model on example input values...")
    evaluate_model_on_inputs(trained_model, scaler)  # Pass the model and scaler
    print("Model evaluation completed.\n")

    print("\nClustering pipeline completed.")


def export_model_and_scaler(model, scaler, model_path="../model/kmeans_model.joblib", scaler_path="../scaler/scaler.joblib"):
    """
    Exports the trained model and scaler to disk.

    Args:
        model: Trained clustering model (e.g., KMeans).
        scaler: Fitted scaler (e.g., MinMaxScaler).
        model_path: File path to save the model.
        scaler_path: File path to save the scaler.
    """
    try:
        # Save the model
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Save the scaler
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
    except Exception as e:
        print(f"Error saving model or scaler: {e}")


if __name__ == "__main__":
    print("Running clustering pipeline...")
    run_clustering_pipeline()
