# experiments.py
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from NDRindex import NDRindex


def generate_data(n_samples=100, n_features=10, n_clusters=3):
    # Generate a synthetic dataset with n_samples samples, n_features features, and n_clusters clusters
    data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters)
    return data


def run_experiment():
    # Generate the data
    data = generate_data()

    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    # Define the normalization and dimensionality reduction methods
    normalization_methods = [lambda x: x]  # no normalization
    pca = PCA(n_components=2)
    dimension_reduction_methods = [pca.fit_transform]  # PCA for dimensionality reduction

    # Create an NDRindex object
    ndr = NDRindex(normalization_methods, dimension_reduction_methods)

    # Evaluate the data quality
    best_methods, best_score = ndr.evaluate_data_quality(normalized_data)
    print(f"Best methods: {best_methods}")
    print(f"Best score (NDRindex): {best_score}")


# Run the experiment
run_experiment()
