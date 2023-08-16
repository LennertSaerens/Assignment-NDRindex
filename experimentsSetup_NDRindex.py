import numpy as np
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
from rpy2.robjects import pandas2ri


numpy2ri.activate()  # Activate the NumPy to R conversion
pandas2ri.activate()  # Activate the Pandas to R conversion

# Import necessary R packages
rcsl = importr('RCSL')
edgeR = importr('edgeR')
Linnorm = importr('Linnorm')
SparseMDC = importr('SparseMDC')

# Load the 'yan' and 'ann' datasets from R
yan_dataset = robjects.r['yan']
ann_dataset = robjects.r['ann']

# Load the Biase datasets from R
data_biase = robjects.r['data_biase']
cell_type_biase = robjects.r['cell_type_biase']

# Convert to appropriate data structures
yan_expression_matrix = np.array(yan_dataset)  # Gene expression matrix
yan_true_labels = np.array(ann_dataset).flatten()  # Cell type labels
biase_expression_matrix = np.array(data_biase).T  # Gene expression matrix, transpose to be same format as Yan
biase_true_labels = np.array(cell_type_biase).flatten()  # Cell type labels

print(type(yan_dataset))
print(type(data_biase))
print(type(yan_expression_matrix))
print(type(biase_expression_matrix))

print(f"Yan dataset shape: {yan_expression_matrix.shape}")
print(f"Biase dataset shape: {biase_expression_matrix.shape}")
print(f"Yan true labels shape: {yan_true_labels.shape}")
print(f"Biase true labels shape: {biase_true_labels.shape}")


# TMM (Trimmed Mean of M-values) Normalization
def tmm_normalization(data):
    data = pd.DataFrame(data)
    data = pandas2ri.py2rpy(data)
    dge_object = edgeR.DGEList(counts=data)
    dge_tmm = edgeR.calcNormFactors(dge_object, method="TMM")
    result = np.array(edgeR.cpm(dge_tmm, log=True)).T
    return result


# Linnorm Normalization
def linnorm_normalization(data):
    transposed_data = data.T  # Transpose the NumPy array
    result = np.array(Linnorm.Linnorm(transposed_data)).T
    return result


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
