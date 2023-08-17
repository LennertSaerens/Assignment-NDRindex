# Load the necessary libraries
library(SC3)
library(SingleCellExperiment)
library(mclust)
library(RCSL)
library(SparseMDC)

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

# Convert factor levels of ann to numeric format for ARI computation later
true_labels <- as.numeric(as.factor(ann$cell_type1))

# Define the number of clusters
k <- 6

# Vector to store ARI values
ari_values <- numeric(100)

# Loop to calculate ARI 100 times
for (i in 1:100) {
  
  # Reset SCE object for SC3 calculations
  sce <- sc3_prepare(sce)
  sce <- sc3_calc_dists(sce)
  sce <- sc3_calc_transfs(sce)
  sce <- sc3_kmeans(sce, ks = k)
  sce <- sc3_calc_consens(sce)
  
  # Get SC3 clusters
  clusters <- colData(sce)$sc3_6_clusters
  
  # Compute the ARI and store in the vector
  ari_values[i] <- adjustedRandIndex(clusters, true_labels)
}

# Compute and print the average ARI
average_ari <- mean(ari_values)
cat("Average ARI over 100 iterations:", average_ari, "\n")

# Store the ARI values for further use
write.csv(ari_values, file="ari_values.csv", row.names=FALSE)
