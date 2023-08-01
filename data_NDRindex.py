import numpy as np
import matplotlib.pyplot as plt

# Define the centers of the four clusters
centers = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])

# Define the standard deviations for the three variations
std_devs = [0.5, 0.25, 0.1]

# Define the number of points per cluster
num_points = 100

# Initialize a dictionary to store the datasets
datasets = {}

# For each shape
for shape in ['normal', 'square', 'hexagram', 'random']:
    datasets[shape] = []
    # For each standard deviation
    for std_dev in std_devs:
        # Initialize an array to store the data
        data = np.empty((0, 2))
        # For each center
        for center in centers:
            if shape == 'normal':
                # Generate normally distributed data
                cluster_data = np.random.normal(loc=center, scale=std_dev, size=(num_points, 2))
            elif shape == 'square':
                # Generate uniformly distributed data in a square
                cluster_data = np.random.uniform(low=center-std_dev, high=center+std_dev, size=(num_points, 2))
            elif shape == 'hexagram':
                # Generate data in a hexagonal distribution
                angles = np.random.uniform(0, 2*np.pi, num_points)
                radii = std_dev * np.sqrt(np.random.uniform(0, 1, num_points))
                x = radii * np.cos(angles) + center[0]
                y = radii * np.sin(angles) + center[1]
                cluster_data = np.stack((x, y), axis=-1)
            elif shape == 'random':
                # Generate random data
                cluster_data = np.random.random((num_points, 2)) * 2 * std_dev + center - std_dev
            # Append the cluster data to the data array
            data = np.append(data, cluster_data, axis=0)
        # Append the data array to the list of datasets for the current shape
        datasets[shape].append(data)

# Now you can access the datasets like this:
normal_datasets = datasets['normal']
square_datasets = datasets['square']
hexagram_datasets = datasets['hexagram']
random_datasets = datasets['random']

# Define the kind of datasets you want to visualize
kind = 'random'

# Get the datasets for the specified kind
datasets_for_kind = datasets[kind]

# Create a figure and axes
fig, axs = plt.subplots(1, len(datasets_for_kind), figsize=(15, 5))

# For each dataset
for i, data in enumerate(datasets_for_kind):
    # Plot the dataset
    axs[i].scatter(data[:, 0], data[:, 1])
    axs[i].set_title(f'Standard Deviation: {std_devs[i]}')

# Display the figure
plt.show()

