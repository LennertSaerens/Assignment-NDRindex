import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(0)

# Number of points in each cluster
num_points = 500

# Center points for each quadrant
centers = [(5, 5), (-5, 5), (-5, -5), (5, -5)]

# Parameters for each variant
normal_stdevs = [2, 1, 0.5]
square_sizes = [4, 2, 1]
hexagram_sizes = [4, 2, 1]
galaxy_radii = [4, 2, 1]

def generate_normal_clusters(stdev):
    datasets = []
    for center in centers:
        x = np.random.normal(center[0], stdev, num_points)
        y = np.random.normal(center[1], stdev, num_points)
        datasets.append(np.column_stack((x, y)))
    return np.concatenate(datasets)

def generate_square_clusters(side_length):
    datasets = []
    for center in centers:
        x = np.random.uniform(center[0] - side_length / 2, center[0] + side_length / 2, num_points)
        y = np.random.uniform(center[1] - side_length / 2, center[1] + side_length / 2, num_points)
        datasets.append(np.column_stack((x, y)))
    return np.concatenate(datasets)

def generate_hexagram_clusters(size):
    datasets = []
    for center in centers:
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        points = []
        for angle in angles:
            x = center[0] + size * np.cos(angle) + np.random.normal(0, size * 0.1, num_points // 6)
            y = center[1] + size * np.sin(angle) + np.random.normal(0, size * 0.1, num_points // 6)
            points.append(np.column_stack((x, y)))
        datasets.append(np.concatenate(points))
    return np.concatenate(datasets)

def generate_galaxy_clusters(radius):
    datasets = []
    for center in centers:
        angles = np.random.uniform(0, 2 * np.pi, num_points)
        distances = radius * np.sqrt(np.random.uniform(0, 1, num_points))
        x = center[0] + distances * np.cos(angles)
        y = center[1] + distances * np.sin(angles)
        datasets.append(np.column_stack((x, y)))
    return np.concatenate(datasets)

# Generate the datasets
normal_datasets = [generate_normal_clusters(stdev) for stdev in normal_stdevs]
square_datasets = [generate_square_clusters(size) for size in square_sizes]
hexagram_datasets = [generate_hexagram_clusters(size) for size in hexagram_sizes]
galaxy_datasets = [generate_galaxy_clusters(radius) for radius in galaxy_radii]

# Visualize the datasets
fig, axes = plt.subplots(4, 3, figsize=(15, 20))

for i, datasets in enumerate([normal_datasets, square_datasets, hexagram_datasets, galaxy_datasets]):
    for j, data in enumerate(datasets):
        axes[i, j].scatter(data[:, 0], data[:, 1], s=10)
        axes[i, j].set_xlim(-10, 10)
        axes[i, j].set_ylim(-10, 10)

axes[0, 0].set_title("Normal, large spread")
axes[0, 1].set_title("Normal, medium spread")
axes[0, 2].set_title("Normal, small spread")

axes[1, 0].set_title("Square, large size")
axes[1, 1].set_title("Square, medium size")
axes[1, 2].set_title("Square, small size")

axes[2, 0].set_title("Hexagram, large size")
axes[2, 1].set_title("Hexagram, medium size")
axes[2, 2].set_title("Hexagram, small size")

axes[3, 0].set_title("Galaxy, large radius")
axes[3, 1].set_title("Galaxy, medium radius")
axes[3, 2].set_title("Galaxy, small radius")

plt.tight_layout()
plt.show()

