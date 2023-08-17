# Load the necessary libraries
library(SC3)
library(SingleCellExperiment)
library(mclust)
library(RCSL)
library(SparseMDC)
library(scDatasets)

filter_threshold <- 1

# # Load the 'yan' dataset
# data(yan)
# ann <- RCSL::ann
# yan_matrix <- as.matrix(t(yan))
# yan_matrix <- yan_matrix[rowSums(yan_matrix) > filter_threshold, ]
# yan_matrix_transposed <- t(yan_matrix)
# yan_sce <- SingleCellExperiment(assays = list(counts = yan_matrix_transposed),
#                                 rowData = DataFrame(feature_symbol = colnames(yan_matrix)))
# logcounts(yan_sce) <- log2(counts(yan_sce) + 1)
# yan_true_labels <- as.numeric(as.factor(ann$cell_type1))

# # Load the 'biase' dataset
# data(data_biase)
# biase_matrix <- as.matrix(data_biase)
# biase_matrix <- biase_matrix[rowSums(biase_matrix) > filter_threshold, ]
# biase_matrix_transposed <- t(biase_matrix)
# biase_sce <- SingleCellExperiment(assays = list(counts = biase_matrix_transposed),
#                                 rowData = DataFrame(feature_symbol = colnames(biase_matrix)))
# logcounts(biase_sce) <- log2(counts(biase_sce) + 1)
# biase_true_labels_raw <- cell_type_biase
# biase_true_labels <- as.numeric(as.factor(biase_true_labels_raw))

# # Load the 'deng' dataset
# data(deng)
# # Extract count matrix directly
# deng_matrix <- assays(deng)$count
# # Filter genes (rows) that meet the threshold
# filter_rows <- which(rowSums(deng_matrix) > filter_threshold)
# # Subset the matrix
# deng_matrix <- deng_matrix[filter_rows, ]
# # Create SingleCellExperiment object without transposing
# deng_sce <- SingleCellExperiment(assays = list(counts = deng_matrix), 
#                                  rowData = DataFrame(feature_symbol = rownames(deng_matrix)))
# logcounts(deng_sce) <- log2(counts(deng_sce) + 1)
# # Extract true labels for the 'deng' dataset (directly from the original deng object)
# deng_true_labels <- as.numeric(as.factor(deng$group))

# Load the 'usoskin' dataset
data(usoskin)
# Extract count matrix directly
usoskin_matrix <- assays(usoskin)$count
# Filter genes (rows) that meet the threshold
filter_rows <- which(rowSums(usoskin_matrix) > filter_threshold)
# Subset the matrix
usoskin_matrix <- usoskin_matrix[filter_rows, ]
# Create SingleCellExperiment object without transposing
usoskin_sce <- SingleCellExperiment(assays = list(counts = usoskin_matrix),
                                 rowData = DataFrame(feature_symbol = rownames(usoskin_matrix)))
logcounts(usoskin_sce) <- log2(counts(usoskin_sce) + 1)
# Extract true labels for the 'usoskin' dataset (directly from the original usoskin object)
usoskin_true_labels <- as.numeric(as.factor(usoskin$Level.1))


# The function to run SC3 clustering and calculate ARI
run_sc3_ari <- function(sce, k, true_labels) {

  # Prepare SCE object for SC3 calculations
  sce <- sc3_prepare(sce)
  sce <- sc3_calc_dists(sce)
  sce <- sc3_calc_transfs(sce)
  sce <- sc3_kmeans(sce, ks = k)
  sce <- sc3_calc_consens(sce)

  # Get SC3 clusters
  clusters <- colData(sce)[[paste0("sc3_", k, "_clusters")]]

  # Compute the ARI
  ari_value <- adjustedRandIndex(clusters, true_labels)

  return(ari_value)
}

# yan_sc3_ari = run_sc3_ari(yan_sce, 6, yan_true_labels)
# cat("YAN_SC3_ARI:", yan_sc3_ari, "\n")
# biase_sc3_ari = run_sc3_ari(biase_sce, 3, biase_true_labels)
# cat("BIASE_SC3_ARI:", biase_sc3_ari, "\n")
# usoskin_sc3_ari = run_sc3_ari(usoskin_sce, 10, usoskin_true_labels)
# cat("DENG_SC3_ARI:", usoskin_sc3_ari, "\n")
usoskin_sc3_ari = run_sc3_ari(usoskin_sce, 4, usoskin_true_labels)
cat("USOSKIN_SC3_ARI:", usoskin_sc3_ari, "\n")
