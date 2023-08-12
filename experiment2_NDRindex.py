import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from NDRindex import NDRindex
from experimentsSetup_NDRindex import yan_dataset, ann_dataset, normalization_methods, dimension_reduction_methods, \
    scale_normalization, pca_reduction

# Initialize NDRindex algorithm
ndr = NDRindex(normalization_methods, dimension_reduction_methods, verbose=True)

# Find the best preprocessing path for the Yan dataset according to the NDRindex algorithm
# best_methods, best_score = ndr.evaluate_data_quality(yan_dataset, num_runs=10)
# print(f"Best score: {best_score}; Best methods: {best_methods}")

# Output of running the NDRindex algorithm:
# Best score: 0.8828749538618043; Best methods: (<function scale_normalization at 0x139f98ae0>, <function pca_reduction at 0x139f98cc0>)
# Results show that the best combination of normalization and dimensionality reduction methods for the "yan" dataset
# is the scale normalization method along with PCA for dimensionality reduction.

# BENCHMARKING THE RESULT WITH ARI

# Convert to appropriate data structures
expression_matrix = np.array(yan_dataset)  # Gene expression matrix
true_labels = np.array(ann_dataset)  # Cell type labels

# Flatten true_labels if it's a 2D array
if true_labels.ndim == 2:
    true_labels = true_labels.flatten()

# Apply scale normalization
normalized_data = scale_normalization(expression_matrix)

# Apply PCA for dimensionality reduction
reduced_data = pca_reduction(normalized_data)

# Apply k-means clustering (set number of clusters based on unique cell types)
kmeans = KMeans(n_clusters=len(np.unique(true_labels)), n_init=10)
kmeans_labels = kmeans.fit_predict(reduced_data)

# Compute ARI for k-means
kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)
print(f"ARI for k-means clustering: {kmeans_ari}")
