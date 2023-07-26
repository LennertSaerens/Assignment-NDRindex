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
                    if not points:  # if there are no points left, break the loop
                        break
                    cluster_center = np.mean(data[cluster], axis=0)  # geometric center of the cluster
                    distances = np.linalg.norm(data[points] - cluster_center,
                                               axis=1)  # distances from the center to all remaining points
                    closest_point_index = np.argmin(distances)  # index of the closest point
                    if distances[
                        closest_point_index] < average_scale:  # if the closest point is close enough, add it to the cluster
                        cluster.append(points.pop(closest_point_index))
                    else:  # if the closest point is not close enough, create a new cluster with it
                        clusters.append([points.pop(closest_point_index)])
        return clusters

    def calculate_final_index(self, data, clusters):
        # Calculate the final index
        final_index = 0
        average_scale = self.calculate_average_scale(data)
        for cluster in clusters:
            cluster_center = np.mean(data[cluster], axis=0)  # geometric center of the cluster
            distances = np.linalg.norm(data[cluster] - cluster_center, axis=1)  # distances from the center to all points in the cluster
            cluster_radius = np.mean(distances)  # average distance, defined as the cluster radius
            final_index += 1 - (cluster_radius / average_scale)  # add 1 - (cluster_radius / average_scale) to the final index
        return final_index / (len(clusters) * 100)  # divide by the number of clusters and the number of runs to get the average

    def evaluate_data_quality(self, data):
        # Evaluate the data qualities
        best_score = -np.inf
        best_methods = None
        for normalization_method in self.normalization_methods:
            normalized_data = normalization_method(data)  # apply normalization
            for dimension_reduction_method in self.dimension_reduction_methods:
                reduced_data = dimension_reduction_method(normalized_data)  # apply dimensionality reduction
                final_index_sum = 0
                for _ in range(100):  # run the algorithm 100 times with different starting points
                    clusters = self.clustering(reduced_data)  # perform clustering
                    final_index = self.calculate_final_index(reduced_data, clusters)  # calculate final index
                    final_index_sum += final_index
                final_index_avg = final_index_sum / 100  # average final index over the 100 runs
                if final_index_avg > best_score:  # if the average final index is higher than the current best score, update the best score and best methods
                    best_score = final_index_avg
                    best_methods = (normalization_method, dimension_reduction_method)
        return best_methods, best_score

