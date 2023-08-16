from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score

from experimentsSetup_NDRindex import *

# Now pass the standardized data to the sc3_clustering function
sc3_labels = sc3_clustering(yan_expression_matrix, yan_true_labels)

# 2. Compute the ARI score
ari_score = adjusted_rand_score(yan_true_labels, sc3_labels)
print(f"ARI score for SC3 clustering on yan dataset: {ari_score:.4f}")

# 3. Visualize the clustering results (if the data is 2D)
# Reduce the data to 2D using PCA for visualization
reduced_data = pca_reduction(yan_expression_matrix)

plt.figure(figsize=(10, 8))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=sc3_labels, cmap='viridis', marker='o', edgecolor='k')
plt.title("SC3 clustering results on yan dataset (visualized using PCA)")
plt.colorbar()
plt.show()
