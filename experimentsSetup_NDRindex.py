import numpy as np
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from rpy2.robjects.vectors import ListVector
from rpy2.robjects import DataFrame

numpy2ri.activate()  # Activate the NumPy to R conversion
pandas2ri.activate()  # Activate the Pandas to R conversion

# Import necessary R packages
rcsl = importr('RCSL')
edgeR = importr('edgeR')
Linnorm = importr('Linnorm')
SparseMDC = importr('SparseMDC')
SingleCellExperiment = importr('SingleCellExperiment')
scDatasets = importr('scDatasets')
SC3 = importr('SC3')

# Load the 'yan' and 'ann' datasets from R
yan_dataset = robjects.r['yan']
ann_dataset = robjects.r['ann']

# Load the Biase datasets from R
data_biase = robjects.r['data_biase']
cell_type_biase = robjects.r['cell_type_biase']

# Load the Deng datasets from R
deng_dataset = robjects.r['deng']

# Load the Usoskin datasets from R
usoskin_dataset = robjects.r['usoskin']

# Convert to appropriate data structures
# YAN
yan_expression_matrix = np.array(yan_dataset)  # Gene expression matrix
yan_true_labels = np.array(ann_dataset).flatten()  # Cell type labels

# BIASE
biase_expression_matrix = np.array(data_biase).T  # Gene expression matrix, transpose to be same format as Yan
biase_true_labels = np.array(cell_type_biase).flatten()  # Cell type labels

# DENG
deng_expression_matrix_R = robjects.r['assay'](deng_dataset)
deng_expression_matrix = np.array(deng_expression_matrix_R).T
# Access the colData slot of the Deng dataset
deng_metadata = robjects.r['colData'](deng_dataset)
# Extract 'group' labels directly using R functions
deng_labels_R = robjects.r['$'](deng_metadata, 'group')
deng_true_labels = np.array(deng_labels_R).flatten()

# USOSKIN
usoskin_expression_matrix_R = robjects.r['assay'](usoskin_dataset)
usoskin_expression_matrix = np.array(usoskin_expression_matrix_R).T
# Access the colData slot of the Usoskin dataset
usoskin_metadata = robjects.r['colData'](usoskin_dataset)
# Extract 'Level.1' labels directly using R functions
usoskin_labels_R = robjects.r['$'](usoskin_metadata, 'Level.1')
usoskin_true_labels = np.array(usoskin_labels_R).flatten()


# print(f"Yan dataset shape: {yan_expression_matrix.shape}")
# print(f"Biase dataset shape: {biase_expression_matrix.shape}")
# print(f"Deng dataset shape: {deng_expression_matrix.shape}")
# print(f"Usoskin dataset shape: {usoskin_expression_matrix.shape}")
# print(f"Yan true labels shape: {yan_true_labels.shape}")
# print(f"Biase true labels shape: {biase_true_labels.shape}")
# print(f"Deng true labels shape: {deng_true_labels.shape}")
# print(f"Usoskin true labels shape: {usoskin_true_labels.shape}")


def tmm_normalization(data_np):
    # Remove rows with all zeros
    data_np = data_np[~np.all(data_np == 0, axis=1)]
    # Remove columns (samples) with sum of zero
    data_np = data_np[:, np.sum(data_np, axis=0) != 0]
    # Convert numpy array to R matrix
    data_r = numpy2ri.py2rpy(data_np)
    # Create a DGEList object
    dge_object = edgeR.DGEList(counts=data_r)
    # Perform TMM normalization
    dge_tmm = edgeR.calcNormFactors(dge_object, method="TMM")
    # Convert normalized data to logCPM
    result = edgeR.cpm(dge_tmm, log=True)
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


def sc3_clustering(data, true_labels):
    # Convert data to R matrix
    data_r = numpy2ri.py2rpy(data)

    # Create a SingleCellExperiment object in R
    sce = SingleCellExperiment.SingleCellExperiment(assays=ListVector({"counts": data_r}))

    # Add dummy gene names to rowData
    num_genes = data.shape[0]
    dummy_gene_names = robjects.StrVector(["Gene" + str(i + 1) for i in range(num_genes)])

    # Assign data and dummy gene names to R environment
    robjects.r.assign("sce", sce)
    robjects.r.assign("dummy_gene_names", dummy_gene_names)

    # Calculate logcounts (using a simple log transformation as an example)
    # Set the rownames, feature_symbol, and logcounts within the R environment
    true_k = len(np.unique(true_labels))
    robjects.r('''
        rownames(sce) <- dummy_gene_names
        rowData(sce)$feature_symbol <- rownames(sce)
        logcounts(sce) <- log2(counts(sce) + 1)
        sce <- SC3::sc3(sce, ks={})
    '''.format(true_k))

    # Extract SC3 clustering results
    sc3_results = robjects.r('sce@colData$sc3_{}clusters'.format(true_k))

    return np.array(sc3_results)


# Define normalization and dimension reduction methods
normalization_methods = [tmm_normalization, linnorm_normalization, scale_normalization]
dimension_reduction_methods = [pca_reduction, tsne_reduction]

# def test(data):
#     for normalization_method in normalization_methods:
#         normalized_data = normalization_method(data)
#         for dimension_reduction_method in dimension_reduction_methods:
#             print(f"Testing combination of {normalization_method.__name__} and {dimension_reduction_method.__name__}")
#             reduced_data = dimension_reduction_method(normalized_data)
#             print(f"Shape of reduced data: {reduced_data.shape}")
#
#
# test(usoskin_expression_matrix)
