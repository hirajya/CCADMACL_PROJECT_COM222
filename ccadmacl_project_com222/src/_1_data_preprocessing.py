import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(data_path="../../data/raw/Country-data.csv"):
    """Loads the dataset from the CSV file."""
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """
    Extracts numerical columns and preprocesses the data.

    Args:
        df: The input DataFrame.

    Returns:
        A tuple containing:
            - X: DataFrame containing only numerical columns.
            - scalers: A dictionary of scalers (StandardScaler and MinMaxScaler).
            - scaled_data: A dictionary containing scaled data for each scaler.
    """
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[numerical_columns]
    X = X.dropna()

    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler()
    }

    scaled_data = {name: scaler.fit_transform(X) for name, scaler in scalers.items()}

    return X, scalers, scaled_data