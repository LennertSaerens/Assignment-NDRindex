from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from NDRindex import NDRindex
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

# Import the RCSL package
rcsl = importr('RCSL')

# Access the 'yan' dataset
yan_dataset = robjects.r['yan']

# Convert the R data frame to a pandas DataFrame
pandas2ri.activate()
yan_df = pandas2ri.rpy2py(yan_dataset)

# Convert the pandas DataFrame to a NumPy array
yan_array = yan_df.values


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


# Define the Sammon Function
def sammon_reduction(data, n_components=2):
    # Approximate Sammon's mapping by setting the dissimilarity parameter to 'euclidean'
    mds = MDS(n_components=n_components, dissimilarity='euclidean', random_state=42)
    return mds.fit_transform(data)


# Define normalization and dimension reduction methods
normalization_methods = [scale_normalization]
dimension_reduction_methods = [pca_reduction, tsne_reduction, sammon_reduction]

# Initialize NDRindex
ndr = NDRindex(normalization_methods, dimension_reduction_methods, verbose=True)

# Evaluate the data quality using the yan_array
best_methods, best_score = ndr.evaluate_data_quality(yan_array, num_runs=3)
print(f"Best score: {best_score}; Best methods: {best_methods}")
