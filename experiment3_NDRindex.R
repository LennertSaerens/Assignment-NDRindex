# Load the necessary libraries
library(SC3)
library(SingleCellExperiment)
library(mclust)
library(RCSL)
library(SparseMDC)
library(foreach)
library(rngtools)

# Load the 'yan' dataset
# data(yan)
# Load the 'biase' dataset
data(data_biase)

# Load the associated true labels
# ann <- RCSL::ann

# Filter out rows (genes) with very low counts or zero across all cells
# Keeping only genes with a count greater than a threshold (e.g., 1) in at least one cell
filter_threshold <- 1
# yan_matrix <- as.matrix(t(yan))
# yan_matrix <- yan_matrix[rowSums(yan_matrix) > filter_threshold, ]
biase_matrix <- as.matrix(data_biase)
biase_matrix <- biase_matrix[rowSums(biase_matrix) > filter_threshold, ]

# Transpose the yan_matrix so that genes are rows and cells are columns
# yan_matrix_transposed <- t(yan_matrix)
biase_matrix_transposed <- t(biase_matrix)

# Create a SingleCellExperiment object
# yan_sce <- SingleCellExperiment(assays = list(counts = yan_matrix_transposed), 
#                             rowData = DataFrame(feature_symbol = colnames(yan_matrix)))
biase_sce <- SingleCellExperiment(assays = list(counts = biase_matrix_transposed), 
                                rowData = DataFrame(feature_symbol = colnames(biase_matrix)))

# Log transform the counts and store in logcounts assay
# logcounts(yan_sce) <- log2(counts(yan_sce) + 1)
logcounts(biase_sce) <- log2(counts(biase_sce) + 1)

# Convert factor levels of ann to numeric format for ARI computation later
# yan_true_labels <- as.numeric(as.factor(ann$cell_type1))

# Load true labels for Biase dataset
biase_true_labels_raw <- cell_type_biase
biase_true_labels <- as.numeric(as.factor(biase_true_labels_raw))

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

metadata(biase_sce)$sc3$params$cores <- 1

# yan_sc3_ari = run_sc3_ari(yan_sce, 6, yan_true_labels)
# cat("YAN_SC3_ARI:", yan_sc3_ari, "\n")
biase_sc3_ari = run_sc3_ari(biase_sce, 3, biase_true_labels)
cat("BIASE_SC3_ARI:", biase_sc3_ari, "\n")
