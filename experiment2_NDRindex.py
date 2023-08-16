from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    }
}


def run_experiment(dataset, ground_truth, clustering_method):
    print(f"Running experiment with clustering method: {clustering_method.__name__} ...")
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


ari_scores_yan, method_combinations_yan = run_experiment(yan_expression_matrix, yan_true_labels, kmeans_yan.fit_predict)
ari_scores_biase, method_combinations_biase = run_experiment(biase_expression_matrix, biase_true_labels,
                                                             kmeans_biase.fit_predict)
ari_scores_deng, method_combinations_deng = run_experiment(deng_expression_matrix, deng_true_labels,
                                                           kmeans_deng.fit_predict)
ari_scores_usoskin, method_combinations_usoskin = run_experiment(usoskin_expression_matrix, usoskin_true_labels, kmeans_usoskin.fit_predict)


def calculate_metrics(ari_scores, method_combinations, best_normalization_method, best_dimension_reduction_method):
    chosen_ari = ari_scores[method_combinations.index((best_normalization_method, best_dimension_reduction_method))]
    avg_ari = np.mean(ari_scores)
    median_ari = np.median(ari_scores)
    upper_quartile_ari = np.percentile(ari_scores, 75)
    max_ari = np.max(ari_scores)
    return [chosen_ari, avg_ari, median_ari, upper_quartile_ari, max_ari]


values_yan = calculate_metrics(ari_scores_yan, method_combinations_yan, 'scale_normalization', 'pca_reduction')
values_biase = calculate_metrics(ari_scores_biase, method_combinations_biase, 'linnorm_normalization', 'pca_reduction')
values_deng = calculate_metrics(ari_scores_deng, method_combinations_deng, 'linnorm_normalization', 'pca_reduction')
values_usoskin = calculate_metrics(ari_scores_usoskin, method_combinations_usoskin, 'linnorm_normalization', 'tsne_reduction')

# VISUALIZATION

colors = ['red', 'green', 'blue', 'purple', 'cyan']  # List of colors for bars
labels = ['Chosen ARI', 'Average ARI', 'Median ARI', 'Upper Quartile ARI', 'Max ARI']
num_metrics = len(labels)
x = np.arange(num_metrics)  # the label locations
width = 0.7  # width of a bar

fig, ax = plt.subplots()

# Adjust positions for yan bars
start_yan = 0
end_yan = num_metrics
positions_yan = np.linspace(start_yan, end_yan - 1, num_metrics)
rects1 = ax.bar(positions_yan, values_yan, width, label='Yan', color=colors)

# Adjust positions for biase bars
start_biase = end_yan + 1
end_biase = start_biase + num_metrics
positions_biase = np.linspace(start_biase, end_biase - 1, num_metrics)
rects2 = ax.bar(positions_biase, values_biase, width, label='Biase', color=colors)

# Adjust positions for deng bars
start_deng = end_biase + 1
end_deng = start_deng + num_metrics
positions_deng = np.linspace(start_deng, end_deng - 1, num_metrics)
rects3 = ax.bar(positions_deng, values_deng, width, label='Deng', color=colors)

# Adjust positions for usoskin bars
start_usoskin = end_deng + 1
end_usoskin = start_usoskin + num_metrics
positions_usoskin = np.linspace(start_usoskin, end_usoskin - 1, num_metrics)
rects4 = ax.bar(positions_usoskin, values_usoskin, width, label='Usoskin', color=colors)

# Define position for dataset names below the sets of bars
dataset_name_y_position = -max(values_yan + values_biase + values_deng) * 0.05  # 5% below the x-axis for clarity

# Adding dataset names below the sets of bars
ax.text(np.mean(positions_yan), dataset_name_y_position, 'Yan', ha='center', va='center')
ax.text(np.mean(positions_biase), dataset_name_y_position, 'Biase', ha='center', va='center')
ax.text(np.mean(positions_deng), dataset_name_y_position, 'Deng', ha='center', va='center')
ax.text(np.mean(positions_usoskin), dataset_name_y_position, 'Usoskin', ha='center', va='center')

# Add legend for colors
legend_elements = [Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(colors, labels)]
ax.legend(handles=legend_elements, loc='upper left')


# Autolabel function to display the label on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()
plt.show()
