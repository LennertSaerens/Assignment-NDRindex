import numpy as np
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from NDRindex import NDRindex
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Import necessary R packages
rcsl = importr('RCSL')
edgeR = importr('edgeR')
Linnorm = importr('Linnorm')

# Access the 'yan' dataset
yan_dataset = robjects.r['yan']

# Convert the R data frame to a pandas DataFrame
pandas2ri.activate()
yan_df = pandas2ri.rpy2py(yan_dataset)

# Convert the pandas DataFrame to a NumPy array
yan_array = yan_df.values


# TMM (Trimmed Mean of M-values) Normalization
def tmm_normalization(data):
    dge_object = edgeR.DGEList(counts=data)
    dge_tmm = edgeR.calcNormFactors(dge_object, method="TMM")
    return np.array(edgeR.cpm(dge_tmm, log=True))


# Linnorm Normalization
def linnorm_normalization(data):
    return np.array(Linnorm.Linnorm(data))


# Scale Normalization
def scale_normalization(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


# Define PCA function
def pca_reduction(data, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


# Define t-SNE function
def tsne_reduction(data, n_components=2):
    tsne = TSNE(n_components=n_components)
    return tsne.fit_transform(data)


# Define normalization and dimension reduction methods
normalization_methods = [linnorm_normalization, tmm_normalization, scale_normalization]
dimension_reduction_methods = [pca_reduction, tsne_reduction]

# Initialize NDRindex
ndr = NDRindex(normalization_methods, dimension_reduction_methods, verbose=True)

# Evaluate the data quality using the yan_array
# best_methods, best_score = ndr.evaluate_data_quality(yan_array, num_runs=10)
# print(f"Best score: {best_score}; Best methods: {best_methods}")

# Output of running the NDRindex algorithm:
# Best score: 0.8828749538618043; Best methods: (<function scale_normalization at 0x139f98ae0>, <function pca_reduction at 0x139f98cc0>)
# Results show that the best combination of normalization and dimensionality reduction methods for the "yan" dataset
# is the scale normalization method along with PCA for dimensionality reduction.

# BENCHMARKING THE RESULT WITH ARI

# Load the 'yan' and 'ann' datasets from R
yan_dataset = robjects.r['yan']
ann_dataset = robjects.r['ann']

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
