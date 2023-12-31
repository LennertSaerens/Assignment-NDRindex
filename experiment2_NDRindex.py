from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import hdbscan

from NDRindex import NDRindex
from experimentsSetup_NDRindex import *

# Initialize NDRindex algorithm
ndr = NDRindex(normalization_methods, dimension_reduction_methods, verbose=True)

# ----------------------------------------------------------------------------------------------

# Find the best preprocessing path for the Yan dataset according to the NDRindex algorithm
# best_methods, best_score = ndr.evaluate_data_quality(yan_dataset, num_runs=100)
# print(f"Best score: {best_score}; Best methods: {best_methods}")

# Output of running the NDRindex algorithm:
# Best score: 0.8828749538618043; Best methods: (<function scale_normalization at 0x139f98ae0>, <function pca_reduction at 0x139f98cc0>)
# Results show that the best combination of normalization and dimensionality reduction methods for the "Yan" dataset
# is the scale normalization method along with PCA for dimensionality reduction.

# ----------------------------------------------------------------------------------------------

# Find the best preprocessing path for the Biase dataset according to the NDRindex algorithm
# best_methods, best_score = ndr.evaluate_data_quality(biase_expression_matrix, num_runs=100)
# print(f"Best score: {best_score}; Best methods: {best_methods}")

# Output of running the NDRindex algorithm:
# Best score: 0.7462070825358466; Best methods: (<function linnorm_normalization at 0x152bcad40>, <function pca_reduction at 0x152bcae80>)
# Results show that the best combination of normalization and dimensionality reduction methods for the "Biase" dataset
# is the linnorm normalization method along with PCA for dimensionality reduction.

# ----------------------------------------------------------------------------------------------

# Find the best preprocessing path for the Deng dataset according to the NDRindex algorithm
# best_methods, best_score = ndr.evaluate_data_quality(deng_expression_matrix, num_runs=100)
# print(f"Best score: {best_score}; Best methods: {best_methods}")

# Output of running the NDRindex algorithm:
# Best score: 0.7462070825358466; Best methods: (<function linnorm_normalization at 0x152bcad40>, <function pca_reduction at 0x152bcae80>)
# Results show that the best combination of normalization and dimensionality reduction methods for the "Deng" dataset
# is the linnorm normalization method along with PCA for dimensionality reduction.

# ----------------------------------------------------------------------------------------------

# Find the best preprocessing path for the Usoskin dataset according to the NDRindex algorithm
# best_methods, best_score = ndr.evaluate_data_quality(usoskin_expression_matrix, num_runs=1000)
# print(f"Best score: {best_score}; Best methods: {best_methods}")

# Output of running the NDRindex algorithm:
# Best score: 0.7050653804288648; Best methods: (<function linnorm_normalization at 0x14d8f0b80>, <function tsne_reduction at 0x14d8f0c20>)
# Results show that the best combination of normalization and dimensionality reduction methods for the "Usoskin" dataset
# is the linnorm normalization method along with TSNE for dimensionality reduction.

# ----------------------------------------------------------------------------------------------

# BENCHMARKING THE RESULT WITH ARI

# Initialize all clustering algorithms
clustering_algorithms = {
    "kmeans": {
        "yan": KMeans(n_clusters=len(np.unique(yan_true_labels)), n_init=10),
        "biase": KMeans(n_clusters=len(np.unique(biase_true_labels)), n_init=10),
        "deng": KMeans(n_clusters=len(np.unique(deng_true_labels)), n_init=10),
        "usoskin": KMeans(n_clusters=len(np.unique(usoskin_true_labels)), n_init=10)
    },
    "hclust": {
        "yan": AgglomerativeClustering(n_clusters=len(np.unique(yan_true_labels))),
        "biase": AgglomerativeClustering(n_clusters=len(np.unique(biase_true_labels))),
        "deng": AgglomerativeClustering(n_clusters=len(np.unique(deng_true_labels))),
        "usoskin": AgglomerativeClustering(n_clusters=len(np.unique(usoskin_true_labels)))
    },
    "ap_clust": {
        "yan": AffinityPropagation(),
        "biase": AffinityPropagation(),
        "deng": AffinityPropagation(),
        "usoskin": AffinityPropagation()
    },
    "hdbscan": {
        "yan": hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True),
        "biase": hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True),
        "deng": hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True),
        "usoskin": hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    }
}

results = {}  # Store results for each dataset and clustering algorithm
datasets = {
    "yan": yan_expression_matrix,
    "biase": biase_expression_matrix,
    "deng": deng_expression_matrix,
    "usoskin": usoskin_expression_matrix
}
ground_truths = {
    "yan": yan_true_labels,
    "biase": biase_true_labels,
    "deng": deng_true_labels,
    "usoskin": usoskin_true_labels
}


def run_experiment(dataset, ground_truth, clustering_method):
    ari_scores = []
    method_combinations = []

    for normalization_method in normalization_methods:
        normalized_data = normalization_method(dataset)
        for dimension_reduction_method in dimension_reduction_methods:
            reduced_data = dimension_reduction_method(normalized_data)
            # print(f"Shape of reduced data: {reduced_data.shape}")  # Add this line
            clustering_labels = clustering_method(reduced_data)
            ari = adjusted_rand_score(ground_truth, clustering_labels)
            ari_scores.append(ari)
            method_combinations.append((normalization_method.__name__, dimension_reduction_method.__name__))
            print(f"ARI for {normalization_method.__name__} and {dimension_reduction_method.__name__}: {ari}")

    return ari_scores, method_combinations


for clustering_name, clustering_methods in clustering_algorithms.items():
    for dataset_name, clustering_method in clustering_methods.items():
        print(f"Running {clustering_name} clustering on {dataset_name} dataset...")
        ari_scores, method_combinations = run_experiment(datasets[dataset_name], ground_truths[dataset_name],
                                                         clustering_method.fit_predict)
        results[f"{clustering_name}_{dataset_name}"] = {
            "ari_scores": ari_scores,
            "method_combinations": method_combinations
        }


def calculate_metrics(ari_scores, method_combinations, best_normalization_method, best_dimension_reduction_method):
    chosen_ari = ari_scores[method_combinations.index((best_normalization_method, best_dimension_reduction_method))]
    avg_ari = np.mean(ari_scores)
    median_ari = np.median(ari_scores)
    upper_quartile_ari = np.percentile(ari_scores, 75)
    max_ari = np.max(ari_scores)
    return [chosen_ari, avg_ari, median_ari, upper_quartile_ari, max_ari]


# Calculate metrics for each dataset and clustering method
values = {}
best_methods_map = {
    "yan": ('scale_normalization', 'pca_reduction'),
    "biase": ('linnorm_normalization', 'pca_reduction'),
    "deng": ('linnorm_normalization', 'pca_reduction'),
    "usoskin": ('linnorm_normalization', 'tsne_reduction')
}
for key, result in results.items():
    dataset_name = key.split("_")[-1]
    best_methods = best_methods_map[dataset_name]
    values[key] = calculate_metrics(result["ari_scores"], result["method_combinations"], *best_methods)

# VISUALIZATION

# Define the layout for the subplots
num_clustering_algorithms = len(clustering_algorithms)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))  # Assuming you have 4 clustering algorithms

# Flatten the axes for easy iteration
axes = axes.ravel()

colors = ['red', 'green', 'blue', 'purple', 'cyan']  # List of colors for bars
labels = ['Chosen ARI', 'Average ARI', 'Median ARI', 'Upper Quartile ARI', 'Max ARI']
num_metrics = len(labels)
x = np.arange(num_metrics)  # the label locations
width = 0.7  # width of a bar

# Define positions for all datasets
positions_yan = np.linspace(0, num_metrics - 1, num_metrics)
positions_biase = positions_yan + num_metrics + 1
positions_deng = positions_biase + num_metrics + 1
positions_usoskin = positions_deng + num_metrics + 1

dataset_name_y_position = -0.1  # Position below the x-axis for clarity


# Autolabel function to display the label on top of bars
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# Iterate over each clustering algorithm and plot in a respective subplot
for idx, (clustering_name, _) in enumerate(clustering_algorithms.items()):
    ax = axes[idx]
    ax.set_title(clustering_name)

    # Fetch results for this clustering algorithm
    values_yan = values[f"{clustering_name}_yan"]
    values_biase = values[f"{clustering_name}_biase"]
    values_deng = values[f"{clustering_name}_deng"]
    values_usoskin = values[f"{clustering_name}_usoskin"]

    # Plot bars for each dataset
    rects1 = ax.bar(positions_yan, values_yan, width, label='Yan', color=colors)
    rects2 = ax.bar(positions_biase, values_biase, width, label='Biase', color=colors)
    rects3 = ax.bar(positions_deng, values_deng, width, label='Deng', color=colors)
    rects4 = ax.bar(positions_usoskin, values_usoskin, width, label='Usoskin', color=colors)

    # Adding dataset names below the sets of bars
    ax.text(np.mean(positions_yan), dataset_name_y_position, 'Yan', ha='center', va='center')
    ax.text(np.mean(positions_biase), dataset_name_y_position, 'Biase', ha='center', va='center')
    ax.text(np.mean(positions_deng), dataset_name_y_position, 'Deng', ha='center', va='center')
    ax.text(np.mean(positions_usoskin), dataset_name_y_position, 'Usoskin', ha='center', va='center')

    # Add legend for colors
    if idx == 0:  # Add legend only to the first subplot for clarity
        legend_elements = [Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(colors, labels)]
        ax.legend(handles=legend_elements, loc='upper left')

    # Label bars
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    autolabel(rects4, ax)

    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])

    # Add x- and y-axis labels only to the border subplots for clarity
    if idx in [0, 1]:
        ax.set_xticks([])
    if idx % 2 == 0:
        ax.set_ylabel('ARI Value')
    else:
        ax.set_yticklabels([])

# Adjust layout for clarity
fig.tight_layout()
plt.show()
