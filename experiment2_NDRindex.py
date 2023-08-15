from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

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

# # Apply scale normalization
# normalized_data = linnorm_normalization(yan_dataset)
# # Apply PCA for dimensionality reduction
# reduced_data = pca_reduction(normalized_data)
#
# # Apply k-means clustering (set number of clusters based on unique cell types)
kmeans = KMeans(n_clusters=len(np.unique(true_labels)), n_init=10)
# kmeans_labels = kmeans.fit_predict(reduced_data)
#
# # Compute ARI for k-means
# kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)
# print(f"ARI for k-means clustering: {kmeans_ari}")


def run_experiment(dataset, ground_truth, clustering_method):
    ari_scores = []
    method_combinations = []

    for normalization_method in normalization_methods:
        normalized_data = normalization_method(dataset)
        for dimension_reduction_method in dimension_reduction_methods:
            reduced_data = dimension_reduction_method(normalized_data)
            clustering_labels = clustering_method(reduced_data)
            ari = adjusted_rand_score(ground_truth, clustering_labels)
            ari_scores.append(ari)
            method_combinations.append((normalization_method.__name__, dimension_reduction_method.__name__))
            print(f"ARI for {normalization_method.__name__} and {dimension_reduction_method.__name__}: {ari}")

    return ari_scores, method_combinations


ari_scores, method_combinations = run_experiment(yan_dataset, true_labels, kmeans.fit_predict)
chosen_ari = ari_scores[method_combinations.index(('scale_normalization', 'pca_reduction'))]
avg_ari = np.mean(ari_scores)
median_ari = np.median(ari_scores)
upper_quartile_ari = np.percentile(ari_scores, 75)
max_ari = np.max(ari_scores)

# Metrics to plot
metrics = ['Chosen ARI', 'Average ARI', 'Median ARI', 'Upper Quartile ARI', 'Max ARI']
values = [chosen_ari, avg_ari, median_ari, upper_quartile_ari, max_ari]

plt.bar(metrics, values, color=['red', 'green', 'blue', 'orange', 'purple'])
plt.ylabel('ARI Score')
plt.title('Comparison of ARI Metrics')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

