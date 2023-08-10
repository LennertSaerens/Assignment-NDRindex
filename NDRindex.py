import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


class NDRindex:
    def __init__(self, normalization_methods, dimension_reduction_methods, verbose=False):
        self.normalization_methods = normalization_methods
        self.dimension_reduction_methods = dimension_reduction_methods
        self.verbose = verbose

    def calculate_distance_matrix(self, data):
        # Calculate the distance matrix
        distance_matrix = squareform(pdist(data))
        return distance_matrix

    def calculate_average_scale(self, data):
        # Calculate the 'average scale' of data
        distance_matrix = self.calculate_distance_matrix(data)
        M = np.percentile(distance_matrix, 25)  # lower quartile distance
        n = data.shape[0]  # number of samples
        average_scale = M / np.log10(n)
        return average_scale

    def clustering(self, data, average_scale):
        # Perform clustering and find the point gathering areas
        clusters = []
        cluster_centers = []  # cache for cluster centers
        points = list(range(data.shape[0]))  # list of point indices
        np.random.shuffle(points)  # randomize the order of points

        while points:
            if not clusters:  # if there are no clusters, create the first one
                point = points.pop()
                clusters.append([point])  # pop a point and create a cluster with it
                cluster_centers.append(data[point])  # add the point itself as the center
            else:
                cluster = clusters[-1]  # the cluster we are trying to expand by one point
                cluster_center = cluster_centers[-1]  # get the last cluster center
                if not points:  # if there are no points left, break the loop
                    break
                distances = np.linalg.norm(data[points] - cluster_center,
                                           axis=1)  # distances from the center to all remaining points
                closest_point_index = np.argmin(distances)  # index of the closest point
                if distances[
                    closest_point_index] < average_scale:  # if the closest point is close enough, add it to the cluster
                    closest_point = points.pop(closest_point_index)
                    cluster.append(closest_point)
                    # Update the cluster center incrementally
                    cluster_center = (cluster_center * len(cluster) + data[closest_point]) / (len(cluster) + 1)
                    cluster_centers[-1] = cluster_center
                else:  # if the closest point is not close enough, create a new cluster with it
                    closest_point = points.pop(closest_point_index)
                    clusters.append([closest_point])
                    cluster_centers.append(data[closest_point])  # add the new point as the center of the new cluster
        return clusters

    def calculate_NDRindex(self, data, clusters):
        R = 0
        average_scale = self.calculate_average_scale(data)
        for cluster in clusters:
            cluster_center = np.mean(data[cluster], axis=0)  # geometric center of the cluster
            distances = np.linalg.norm(data[cluster] - cluster_center, axis=1)  # distances from the center to all points in the cluster
            cluster_radius = np.mean(distances)  # average distance, defined as the cluster radius
            R += cluster_radius
        R /= len(clusters)  # divide by the number of clusters K
        return 1 - (R / average_scale)

    def evaluate_data_quality(self, data, num_runs):
        # Evaluate the data qualities
        best_score = -np.inf
        best_methods = None
        total_iterations = len(self.normalization_methods) * len(self.dimension_reduction_methods) * num_runs
        pbar = tqdm(total=total_iterations, disable=not self.verbose, desc="Processing", ncols=100)  # Initialize tqdm progress bar

        for normalization_method in self.normalization_methods:
            normalized_data = normalization_method(data)  # apply normalization
            for dimension_reduction_method in self.dimension_reduction_methods:
                reduced_data = dimension_reduction_method(normalized_data)  # apply dimensionality reduction
                average_scale = self.calculate_average_scale(reduced_data)
                final_index_sum = 0
                for i in range(num_runs):  # run the algorithm num_runs times with different starting points
                    clusters = self.clustering(reduced_data, average_scale)  # perform clustering
                    final_index = self.calculate_NDRindex(reduced_data, clusters)  # calculate final index
                    final_index_sum += final_index
                    pbar.update(1)  # Update the progress bar
                final_index_avg = final_index_sum / num_runs  # average final index over the total number of runs
                if final_index_avg > best_score:  # if the average final index is higher than the current best score, update the best score and best methods
                    best_score = final_index_avg
                    best_methods = (normalization_method, dimension_reduction_method)

        pbar.close()  # Close the tqdm progress bar
        return best_methods, best_score
