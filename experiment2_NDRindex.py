from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from NDRindex import NDRindex

# Import the RCSL package
rcsl = importr('RCSL')

# Access the 'yan' dataset
yan_dataset = robjects.r['yan']

# Convert the R data frame to a pandas DataFrame
pandas2ri.activate()
yan_df = pandas2ri.rpy2py(yan_dataset)

# Convert the pandas DataFrame to a NumPy array
yan_array = yan_df.values

# Define normalization and dimension reduction methods
normalization_methods = [lambda x: x]  # No normalization
dimension_reduction_methods = [lambda x: x]  # No dimension reduction

# Initialize NDRindex
ndr = NDRindex(normalization_methods, dimension_reduction_methods, verbose=True)

# Evaluate the data quality using the yan_array
best_methods, best_score = ndr.evaluate_data_quality(yan_array, num_runs=10)
print(f"Best score: {best_score}; Best methods: {best_methods}")
