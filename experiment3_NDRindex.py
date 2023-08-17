from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

r_results_mapping = {
    'yan': {
        'SC3': 0.6584306,
        'pcaReduce': 0.8258125,
        'SNN-Cliq': 0.5777778,
        'Seurat': 0.6911318
    },
    'biase': {
        'SC3': 0.9516773,
        'pcaReduce': 0.9475349,
        'SNN-Cliq': 0.6034298,
        'Seurat': 0.8775349
    },
    'deng': {
        'SC3': 0.4964559,
        'pcaReduce': 0.4467143,
        'SNN-Cliq': 0.3378124,
        'Seurat': 0.3685968
    },
    'usoskin': {
        'SC3': 0.8804224,
        'pcaReduce': 0.5681257,
        'SNN-Cliq': 0.2874987,
        'Seurat': 0.5653912
    }
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

# VISUALIZATION

# Convert ARI results and r_results_mapping into long form DataFrame
dataset_names = []
ari_scores = []
score_types = []

# Append NDRindex scores
for dataset, (avg_ari, aris) in ari_results.items():
    dataset_names.extend([dataset] * len(aris))
    ari_scores.extend(aris)
    score_types.extend(['NDRindex'] * len(aris))

# Append sc3_ARI scores
for dataset, scores in r_results_mapping.items():
    dataset_names.append(dataset)
    ari_scores.append(scores['SC3'])
    score_types.append('SC3')

    # Append pcaReduce_ARI scores
    dataset_names.append(dataset)
    ari_scores.append(scores['pcaReduce'])
    score_types.append('pcaReduce')

    # Append SNN-Cliq_ARI scores
    dataset_names.append(dataset)
    ari_scores.append(scores['SNN-Cliq'])
    score_types.append('SNN-Cliq')

    # Append Seurat_ARI scores
    dataset_names.append(dataset)
    ari_scores.append(scores['Seurat'])
    score_types.append('Seurat')

ari_df = pd.DataFrame({
    'Dataset': dataset_names,
    'ARI': ari_scores,
    'Score Type': score_types
})

# Plot using Seaborn
plt.figure(figsize=(12, 8))

# Use barplot to display the average ARI scores as rectangles
sns.barplot(data=ari_df, x='Dataset', y='ARI', hue='Score Type', errorbar=None, estimator=np.mean, palette="pastel")

plt.title('Distribution of ARI scores for different datasets')
plt.ylabel('Adjusted Rand Index (ARI)')
plt.xlabel('Dataset')
plt.tight_layout()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title='Score Type')
plt.show()
