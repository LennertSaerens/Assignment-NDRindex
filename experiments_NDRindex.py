# experiments.py
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from NDRindex import NDRindex


def generate_data(n_samples=1000, n_features=2, n_clusters=3, cluster_std=1.0, center_box=(-10.0, 10.0)):
    # Generate a synthetic dataset with n_samples samples, n_features features, and n_clusters clusters
    data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=cluster_std, center_box=center_box)
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

    # Plot the data
    plt.scatter(data[:, 0], data[:, 1])
    plt.title('Generated Data')
    plt.show()

# Run the experiment
run_experiment()

