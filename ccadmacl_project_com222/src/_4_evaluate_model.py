from _1_data_preprocessing import load_data, preprocess_data
from _2_model import train_kmeans
import numpy as np


def predict_cluster(input_values, scaler, model):
    """
    Predicts the cluster for given input values.

    Args:
        input_values: A list of input values.
        scaler: The scaler used for data preprocessing.
        model: The trained KMeans model.

    Returns:
        The predicted cluster label.
    """
    input_values = np.array(input_values).reshape(1, -1)
    scaled_input = scaler.transform(input_values)
    cluster = model.predict(scaled_input)
    return cluster[0]


def evaluate_model_on_inputs(model, scaler, inputs=None):
    """
    Evaluates the model on example or user-provided input values.

    Args:
        model: The trained KMeans model.
        scaler: The scaler used for preprocessing.
        inputs: A list of input values or None for default examples.

    Returns:
        Predictions for input values.
    """
    if inputs is None:
        # Default example inputs
        inputs = [
            [90.2, 10.0, 7.58, 44.9, 1610, 9.44, 56.2, 5.82, 553],
            [16.6, 28.0, 6.55, 48.6, 9930, 4.49, 76.3, 1.65, 4090],
        ]

    predictions = []
    for input_values in inputs:
        cluster = predict_cluster(input_values, scaler, model)
        predictions.append(cluster)
        print(f"Predicted cluster for input {input_values}: {cluster}")

    return predictions
