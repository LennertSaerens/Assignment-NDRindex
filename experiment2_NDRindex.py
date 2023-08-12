import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans

from NDRindex import NDRindex
from experimentsSetup_NDRindex import *

# Initialize NDRindex algorithm
ndr = NDRindex(normalization_methods, dimension_reduction_methods, verbose=True)

# Find the best preprocessing path for the Yan dataset according to the NDRindex algorithm
# best_methods, best_score = ndr.evaluate_data_quality(yan_dataset, num_runs=1)
# print(f"Best score: {best_score}; Best methods: {best_methods}")

# Output of running the NDRindex algorithm:
# Best score: 0.8828749538618043; Best methods: (<function scale_normalization at 0x139f98ae0>, <function pca_reduction at 0x139f98cc0>)
# Results show that the best combination of normalization and dimensionality reduction methods for the "yan" dataset
# is the scale normalization method along with PCA for dimensionality reduction.

# BENCHMARKING THE RESULT WITH ARI

# Apply scale normalization
normalized_data = linnorm_normalization(yan_dataset)
# Apply PCA for dimensionality reduction
reduced_data = pca_reduction(normalized_data)

# Apply k-means clustering (set number of clusters based on unique cell types)
kmeans = KMeans(n_clusters=len(np.unique(true_labels)), n_init=10)
kmeans_labels = kmeans.fit_predict(reduced_data)

print(true_labels.shape)
print(kmeans_labels.shape)

# Compute ARI for k-means
kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)
print(f"ARI for k-means clustering: {kmeans_ari}")


# def run_experiment(dataset, ground_truth):
#     for normalization_method in normalization_methods:
#         print("poep")
#         normalized_data = normalization_method(dataset)
#         print("hier")
#         for dimension_reduction_method in dimension_reduction_methods:
#             reduced_data = dimension_reduction_method(normalized_data)
#             print("daar")
#             clustering_labels = kmeans.fit_predict(reduced_data)
#             ari = adjusted_rand_score(ground_truth, clustering_labels)
#             print(f"ARI for {normalization_method.__name__} and {dimension_reduction_method.__name__}: {ari}")
#
#
# run_experiment(yan_dataset, true_labels)
