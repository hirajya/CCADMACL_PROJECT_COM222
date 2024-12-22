import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Set up absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Base directory of the current script
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../model/kmeans_model.joblib"))  # Model path
SCALER_PATH = os.path.abspath(os.path.join(BASE_DIR, "../scaler/scaler.joblib"))  # Scaler path
DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../data/raw/Country-data.csv"))  # Dataset path

# Data dictionary for the features
data_dict = {
    "child_mort": "Child Mortality Rate (per 1,000 live births)",
    "exports": "Exports as a percentage of GDP",
    "health": "Health expenditure as a percentage of GDP",
    "imports": "Imports as a percentage of GDP",
    "income": "Income per capita (in USD)",
    "inflation": "Inflation rate (annual %)",
    "life_expec": "Life expectancy at birth (years)",
    "total_fer": "Total fertility rate (children per woman)",
    "gdpp": "GDP per capita (USD)"
}

# Problem statement and objective
problem_statement = """
### Problem Statement:
This project aims to cluster countries based on various socio-economic features using the KMeans algorithm. The dataset contains information about countries and their characteristics, such as health, income, inflation, life expectancy, and other factors.

### Objective:
The goal is to segment countries into clusters to identify patterns or similarities in their socio-economic conditions. These clusters can help in identifying countries that may require aid, have a medium need for aid, or do not need aid at all.
"""
st.sidebar.header("Country Cluster Prediction")
sidebar_selection = st.sidebar.selectbox("Choose Page", ["Home", "Data Exploration", "Cluster Prediction", "Analysis"])

# Function to load model and scaler
def load_model_and_scaler():
    """
    Load the pre-trained model and scaler.
    """
    try:
        model = joblib.load(MODEL_PATH)  # Load model
        scaler = joblib.load(SCALER_PATH)  # Load scaler
        st.success("Model and scaler loaded successfully!")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None

# Function to predict cluster for input data
def predict_cluster(model, scaler, input_data):
    """
    Predict the cluster for the given input data.
    
    Args:
        model: The trained clustering model.
        scaler: The fitted scaler for preprocessing.
        input_data: A NumPy array of input values.
    
    Returns:
        Predicted cluster label.
    """
    scaled_data = scaler.transform([input_data])
    cluster = model.predict(scaled_data)
    return cluster[0]

# Load dataset
def load_dataset():
    """
    Load the dataset for exploration.
    """
    try:
        data = pd.read_csv(DATASET_PATH)
        st.success("Dataset loaded successfully!")
        return data
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None

# Home Page
if sidebar_selection == "Home":
    st.title("Clustering Countries with KMeans")
    
    # Add Problem Statement and Objective
    st.subheader("Problem Statement:")
    st.markdown("""
    HELP International has been able to raise around $ 10 million. The CEO of the NGO needs to decide how to use this money strategically and effectively. Therefore, the CEO must choose the countries that are in the direst need of aid. 
    Your job as a Data Scientist is to categorize countries using socio-economic and health factors that determine the overall development of a country. Based on this categorization, you will suggest which countries the CEO should focus on the most.
    """)

    st.subheader("Objective:")
    st.markdown("""
    The goal of this project is to cluster countries using various socio-economic and health factors. By analyzing the data, we can segment countries into different groups, which can then be analyzed to decide where aid is needed the most.
    """)

    # Add About the Organization section
    st.subheader("About HELP International:")
    st.markdown("""
    HELP International is an international humanitarian NGO committed to fighting poverty and providing the people of underdeveloped countries with basic amenities and relief during natural disasters and calamities.
    """)

    # Display Data Dictionary
    st.subheader("Data Dictionary")
    for feature, description in data_dict.items():
        st.write(f"**{feature}:** {description}")
    
    # Add the link to the dataset
    st.subheader("Dataset")
    st.write(
        "The dataset used in this project is available on Kaggle. You can access it here: "
        "[Country Data Dataset on Kaggle](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data?select=Country-data.csv)"
    )

# Data Exploration Page
elif sidebar_selection == "Data Exploration":
    st.title("Data Exploration")
    st.write("You can search for a country and explore its data.")

    # Load the dataset
    data = load_dataset()
    if data is not None:
        country_name = st.text_input("Search for a country:")
        if country_name:
            country_data = data[data['country'].str.contains(country_name, case=False)]
            if not country_data.empty:
                st.write(country_data)
            else:
                st.write("Country not found in the dataset.")
        else:
            st.write("Please enter a country name to search.")

# Cluster Prediction Page
elif sidebar_selection == "Cluster Prediction":
    st.title("Cluster Prediction")
    st.write("Enter the feature values for prediction:")

    # Load the dataset
    data = load_dataset()
    if data is not None:
        st.subheader("Dataset Head")
        st.write(data.head())  # Displaying the first 5 rows of the dataset

    # Input fields for prediction
    input_features = [
        "child_mort", "exports", "health", "imports",
        "income", "inflation", "life_expec", "total_fer", "gdpp"
    ]
    input_values = []
    for feature in input_features:
        value = st.number_input(f"{feature}:", step=0.01)
        input_values.append(value)

    # Load model and scaler
    model, scaler = load_model_and_scaler()

    # Predict cluster
    if st.button("Predict Cluster"):
        if model and scaler:
            try:
                cluster = predict_cluster(model, scaler, np.array(input_values))
                # Interpreting the cluster
                if cluster == 0:
                    cluster_message = "No need for aid"
                elif cluster == 1:
                    cluster_message = "Medium need for aid"
                elif cluster == 2:
                    cluster_message = "In need of aid"
                else:
                    cluster_message = "Unknown cluster"
                
                st.success(f"The predicted cluster is: {cluster_message}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Model or scaler is not loaded. Please check the file paths.")

# Analysis Page
elif sidebar_selection == "Analysis":
    st.title("Cluster Analysis")

    # Load dataset
    data = load_dataset()
    if data is not None:
        # Load the model and scaler
        model, scaler = load_model_and_scaler()
        
        if model and scaler:
            # Prepare input data (features) from the dataset
            features = data[["child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"]]
            
            # Scale the data using the same scaler
            scaled_data = scaler.transform(features)
            
            # Predict the clusters for all countries in the dataset
            data['cluster'] = model.predict(scaled_data)
            
            # Group countries by their cluster predictions
            cluster_0 = data[data['cluster'] == 0]  # No need for aid
            cluster_1 = data[data['cluster'] == 1]  # Medium need for aid
            cluster_2 = data[data['cluster'] == 2]  # In need of aid

            # Display the results
            st.subheader("Countries with No Need for Aid (Cluster 0)")
            st.write(cluster_0[['country', 'cluster']])
            
            st.subheader("Countries with Medium Need for Aid (Cluster 1)")
            st.write(cluster_1[['country', 'cluster']])
            
            st.subheader("Countries in Need of Aid (Cluster 2)")
            st.write(cluster_2[['country', 'cluster']])
            
            # Show a summary of the clusters
            st.subheader("Cluster Summary")
            st.write(f"Total countries in need of aid (Cluster 2): {len(cluster_2)}")
            st.write(f"Total countries with medium need for aid (Cluster 1): {len(cluster_1)}")
            st.write(f"Total countries with no need for aid (Cluster 0): {len(cluster_0)}")

        else:
            st.error("Model or scaler is not loaded. Please check the file paths.")

# Footer
st.write("---")
st.write("**Advance Machine Learning Project (COM222)**")
st.write("Christian Joshua Alberto | Mark Rhey Anthony De Luna | Rodney Lei Estrada")
st.write("*Built with Streamlit*")
