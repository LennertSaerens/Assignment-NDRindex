import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform


class NDRindex:
    def __init__(self, normalization_methods, dimension_reduction_methods):
        self.normalization_methods = normalization_methods
        self.dimension_reduction_methods = dimension_reduction_methods

    def calculate_distance_matrix(self, data):
        # Calculate the distance matrix
        distance_matrix = squareform(pdist(data))
        return distance_matrix

    def calculate_average_scale(self, data):
        # Calculate the 'average scale' of data
        distance_matrix = self.calculate_distance_matrix(data)
        M = np.percentile(distance_matrix, 25)  # lower quartile distance
        n = data.shape[0]  # number of samples
        average_scale = M * np.log10(n)
        return average_scale

    def clustering(self, data):
        # Perform clustering and find the point gathering areas
        average_scale = self.calculate_average_scale(data)
        clusters = []
        points = list(range(data.shape[0]))  # list of point indices
        np.random.shuffle(points)  # randomize the order of points

        while points:
            if not clusters:  # if there are no clusters, create the first one
                clusters.append([points.pop()])  # pop a point and create a cluster with it
            else:
                for cluster in clusters:
                    cluster_center = np.mean(data[cluster], axis=0)  # geometric center of the cluster
                    distances = np.linalg.norm(data[points] - cluster_center, axis=1)  # distances from the center to all remaining points
                    closest_point_index = np.argmin(distances)  # index of the closest point
                    if distances[closest_point_index] < average_scale:  # if the closest point is close enough, add it to the cluster
                        cluster.append(points.pop(closest_point_index))
                    else:  # if the closest point is not close enough, create a new cluster with it
                        clusters.append([points.pop(closest_point_index)])
        return clusters

    def calculate_final_index(self, clusters):
        # Calculate the final index
        final_index = 0
        # TODO: Calculate the final index as described in the paper
        return final_index

    def evaluate_data_quality(self, data):
        # Evaluate the data qualities
        best_score = -np.inf
        best_methods = None
        for normalization_method in self.normalization_methods:
            for dimension_reduction_method in self.dimension_reduction_methods:
                pass
                # TODO: Apply the normalization and dimensionality reduction methods to the data
                # TODO: Calculate the final index for the preprocessed data
                # TODO: If the final index is higher than the current best score, update the best score and best methods
        return best_methods
