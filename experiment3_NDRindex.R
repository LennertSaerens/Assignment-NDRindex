# Load the necessary libraries
library(SC3)
library(SingleCellExperiment)
library(mclust)
library(RCSL)

# Load the 'yan' dataset
data(yan)

# Load the associated true labels
ann <- RCSL::ann

# Filter out rows (genes) with very low counts or zero across all cells
# Keeping only genes with a count greater than a threshold (e.g., 1) in at least one cell
filter_threshold <- 1
yan_matrix <- yan_matrix[rowSums(yan_matrix) > filter_threshold, ]

# Check dimensions after filtering
cat("Dimensions of yan_matrix after filtering: ", dim(yan_matrix), "\n")

# Transpose the yan_matrix so that genes are rows and cells are columns
yan_matrix_transposed <- t(yan_matrix)

# Create a SingleCellExperiment object
sce <- SingleCellExperiment(assays = list(counts = yan_matrix_transposed), 
                            rowData = DataFrame(feature_symbol = colnames(yan_matrix)))


# Log transform the counts and store in logcounts assay
logcounts(sce) <- log2(counts(sce) + 1)

# Check assay names
cat("Assay names in SingleCellExperiment: ", assayNames(sce), "\n")

# Directly set the number of clusters based on unique labels in 'ann'
k <- 6
cat("Number of clusters (k) based on unique labels in 'ann':", k, "\n")

# If yan dataset has row names, use them as gene names
if (!is.null(rownames(yan))) {
  rowData(sce)$feature_symbol <- rownames(yan)
} else {
  # Otherwise, use placeholder gene names
  rowData(sce)$feature_symbol <- paste0("Gene_", seq_len(nrow(sce)))
}

# Set k in the SC3 metadata and proceed with the clustering
metadata(sce)$sc3$k <- k
sce <- sc3_prepare(sce)
sce <- sc3_calc_dists(sce)
sce <- sc3_calc_transfs(sce)
sce <- sc3_kmeans(sce, ks = k)
sce <- sc3_calc_consens(sce)

# Get SC3 clusters
clusters <- colData(sce)$sc3_6_clusters

# Convert factor levels of ann to numeric format
true_labels <- as.numeric(as.factor(ann$cell_type1))

# Now, compute the ARI
ari_sc3 <- adjustedRandIndex(clusters, true_labels)
print(ari_sc3)

