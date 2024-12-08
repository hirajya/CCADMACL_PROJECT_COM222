# Unsupervised ML Project: Country Data

## Overview

This project compares various clustering techniques to analyze the provided dataset. The techniques used include **K-Means**, **Hierarchical Clustering**, and **DBSCAN**. The goal is to evaluate and compare how each model clusters the data based on different metrics.

The dataset used is related to various indicators of countries, such as child mortality, exports, health spending, inflation, GDP, and more. The aim is to gain insights into how different countries are grouped based on these factors.

You can find the dataset [here on Kaggle](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data?select=Country-data.csv).

## Features

- **K-Means Clustering**: A partition-based clustering algorithm that divides the dataset into k clusters.
- **Hierarchical Clustering**: A method that builds a hierarchy of clusters.
- **DBSCAN**: A density-based clustering algorithm that groups data points based on density.

## Prerequisites

Before running the project, ensure you have the following tools installed:

- **Python** (version 3.8 or higher)
- **Poetry** (for managing dependencies)

## How to Clone and Set Up the Project

Follow these steps to clone the repository and set up the environment:

1. **Clone the Repository**
   Clone this repository to your local machine by running the following command:
   ```bash
   git clone https://github.com/yourusername/ml-clustering-comparison.git```

2. **Navigate to the Project Directory**
   ```bash
   cd ml-clustering-comparison```


3. **Install Dependencies using Poetry**
   Install the dependencies listed in the pyproject.toml file by running:
   ```bash
   poetry shell```
