from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from experimentsSetup_NDRindex import *

# This map is available in the third experiment because the optimal preprocessing methods according
# to the NDRindex algorithm are already known from the previous experiment.
best_methods_map = {
    "yan": ('scale_normalization', 'pca_reduction'),
    "biase": ('linnorm_normalization', 'pca_reduction'),
    "deng": ('linnorm_normalization', 'pca_reduction'),
    "usoskin": ('linnorm_normalization', 'tsne_reduction')
}

# Mapping the method names to the actual functions. Only those in the best_methods_map are needed.
method_mapping = {
    'scale_normalization': scale_normalization,
    'linnorm_normalization': linnorm_normalization,
    'pca_reduction': pca_reduction,
    'tsne_reduction': tsne_reduction
}


def compute_ARI_with_best_methods(data, true_labels, best_methods, num_runs=100):
    normalization_method = method_mapping[best_methods[0]]
    dimension_reduction_method = method_mapping[best_methods[1]]
    normalized_data = normalization_method(data)
    reduced_data = dimension_reduction_method(normalized_data)

    aris = []
    for _ in range(num_runs):
        clustering = AgglomerativeClustering(n_clusters=len(np.unique(true_labels)))
        inferred_labels = clustering.fit_predict(reduced_data)
        ari = adjusted_rand_score(true_labels, inferred_labels)
        aris.append(ari)

    avg_ari = sum(aris) / len(aris)
    return avg_ari, aris


# Compute ARI for each dataset using NDRindex with the best methods
datasets = {
    "yan": (yan_expression_matrix, yan_true_labels),
    "biase": (biase_expression_matrix, biase_true_labels),
    "deng": (deng_expression_matrix, deng_true_labels),
    "usoskin": (usoskin_expression_matrix, usoskin_true_labels)
}

ari_results = {}
for dataset_name, (data, labels) in datasets.items():
    avg_ari, aris = compute_ARI_with_best_methods(data, labels, best_methods_map[dataset_name])
    ari_results[dataset_name] = (avg_ari, aris)
