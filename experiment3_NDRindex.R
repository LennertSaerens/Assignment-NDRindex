library(SC3)
library(SingleCellExperiment)
library(mclust)
library(RCSL)

data(yan)

# Convert the dataset to the appropriate format
# Transpose the matrix since SC3 expects cells as rows and genes as columns
yan_matrix <- as.matrix(t(yan))

# Create a SingleCellExperiment object
sce <- SingleCellExperiment(assays = list(counts = yan_matrix))

# Normalize the data and perform clustering
sce <- sc3_estimate_k(sce)
sce <- sc3_prepare(sce)
sce <- sc3_calc_dists(sce)
sce <- sc3_calc_transfs(sce)
sce <- sc3_kmeans(sce)
sce <- sc3_calc_consens(sce)

# Get SC3 clusters
clusters <- colData(sce)$sc3_clus_groups_k

# Assuming you have the true labels for the yan dataset
# true_labels <- ...  # Load or define the true labels here

# Compute ARI
ari_sc3 <- adjustedRandIndex(clusters, true_labels)

