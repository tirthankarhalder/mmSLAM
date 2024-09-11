import numpy as np
from scipy.spatial import cKDTree
from pyemd import emd


def chamfer_distance(point_cloud1, point_cloud2):
    tree1 = cKDTree(point_cloud1)
    tree2 = cKDTree(point_cloud2)
    dist1, _ = tree1.query(point_cloud2, k=1)
    dist2, _ = tree2.query(point_cloud1, k=1)
    chamfer_dist = np.mean(dist1) + np.mean(dist2)
    return chamfer_dist

def earth_movers_distance(point_cloud1, point_cloud2):
    if len(point_cloud1) != len(point_cloud2):
        raise ValueError("Point clouds must have the same number of points for EMD computation.")
    dists = np.linalg.norm(point_cloud1[:, np.newaxis] - point_cloud2[np.newaxis, :], axis=2)
    weights = np.ones(len(point_cloud1)) / len(point_cloud1)
    emd_distance = emd(weights, weights, dists)
    return emd_distance

def combined_loss(point_cloud1, point_cloud2, alpha=0.5):
    cd = chamfer_distance(point_cloud1, point_cloud2)
    emd_dist = earth_movers_distance(point_cloud1, point_cloud2)
    loss = alpha * cd + (1 - alpha) * emd_dist
    return loss

def mse_loss(point_cloud1, point_cloud2):
    assert point_cloud1.shape == point_cloud2.shape, "Point clouds must have the same shape."
    squared_diffs = np.square(point_cloud1 - point_cloud2)
    mse = np.mean(squared_diffs)
    return mse


if __name__== "__main__":
    upsampled_point_cloud = np.random.rand(1000, 3)  # Replace with your data
    ground_truth_point_cloud = np.random.rand(1000, 3)  # Replace with your data

    distance = chamfer_distance(upsampled_point_cloud, ground_truth_point_cloud)
    print(f'Chamfer Distance: {distance}')
    loss = combined_loss(upsampled_point_cloud, ground_truth_point_cloud)
    print(f'Combined Loss: {loss}')
    loss = mse_loss(upsampled_point_cloud, ground_truth_point_cloud)
    print(f'MSE Loss: {loss}')