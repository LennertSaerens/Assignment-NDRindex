import numpy as np
from NDRindex import NDRindex

data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
    [19, 20, 21],
    [22, 23, 24],
    [25, 26, 27],
    [28, 29, 30]
])


def test_calculate_distance_matrix():
    ndr = NDRindex([], [])
    distance_matrix = ndr.calculate_distance_matrix(data)
    assert distance_matrix.shape == (10, 10)  # the distance matrix should be a 10x10 matrix
    assert np.allclose(distance_matrix, distance_matrix.T)  # the distance matrix should be symmetric


def test_calculate_average_scale():
    ndr = NDRindex([], [])
    average_scale = ndr.calculate_average_scale(data)
    assert average_scale > 0  # the average scale should be a positive number


def test_clustering():
    ndr = NDRindex([], [])
    distance_matrix = ndr.calculate_distance_matrix(data)
    avg_scale = ndr.calculate_average_scale(distance_matrix)
    clusters = ndr.clustering(data, avg_scale)
    assert len(clusters) > 0  # there should be at least one cluster
    assert sum(len(cluster) for cluster in clusters) == 10  # all points should belong to a cluster


def test_calculate_final_index():
    ndr = NDRindex([], [])
    distance_matrix = ndr.calculate_distance_matrix(data)
    avg_scale = ndr.calculate_average_scale(distance_matrix)
    clusters = ndr.clustering(data, avg_scale)
    final_index = ndr.calculate_NDRindex(data, clusters)
    assert final_index >= 0  # the final index should be a positive number


test_calculate_distance_matrix()
test_calculate_average_scale()
test_clustering()
test_calculate_final_index()
