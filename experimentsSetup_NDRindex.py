import numpy as np
from rpy2 import robjects
from rpy2.robjects.packages import importr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Import necessary R packages
rcsl = importr('RCSL')
edgeR = importr('edgeR')
Linnorm = importr('Linnorm')

# Load the 'yan' and 'ann' datasets from R
yan_dataset = robjects.r['yan']
ann_dataset = robjects.r['ann']

# Convert to appropriate data structures
expression_matrix = np.array(yan_dataset)  # Gene expression matrix
true_labels = np.array(ann_dataset)  # Cell type labels

# Flatten true_labels if it's a 2D array
if true_labels.ndim == 2:
    true_labels = true_labels.flatten()


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
