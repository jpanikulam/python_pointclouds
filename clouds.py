"""Point Cloud Manipulation

TODO: Use FLANN for nn (Big bottleneck!)
sudo apt-get install libflann-dev libflann1.8

TODO: Cython implementation of normal accumulation + covariance
TODO: voxel downsampling
"""

import time
import numpy as np
# holy shit always use cKDtree...
from scipy.spatial import cKDTree
import scipy.linalg as scl

import visualize


def uniform_voxelgrid_sample(points, voxel_size=1.0):
    _min = np.min(points, axis=0)
    _max = np.max(points, axis=0)

    # Is there a one-liner for doing this?
    x_range = np.arange(_min[0], _max[0], voxel_size)
    y_range = np.arange(_min[1], _max[1], voxel_size)
    z_range = np.arange(_min[2], _max[2], voxel_size)
    (X, Y, Z) = np.meshgrid(x_range, y_range, z_range)

    # Get voxel vorners
    voxel_grid_corners = np.vstack((
        X.ravel(),
        Y.ravel(),
        Z.ravel()
    )).transpose()

    group_means = []
    for n in range(voxel_grid_corners.shape[0]):
        in_box = np.logical_and(points <= voxel_grid_corners[n] + voxel_size, points > voxel_grid_corners[n])
        pts_in = np.all(in_box, axis=1)

        if np.sum(pts_in) == 0:
            continue

        points_in_box = points[pts_in]
        group_mean = np.mean(points_in_box, axis=0)
        group_means.append(group_mean)

    downsampled_points = np.vstack(group_means)
    return downsampled_points


def robust_normals(points, leaf_size=0.05, outlier_angle=0.1):
    """TODO"""
    tic = time.time()
    # need copy?
    kdtree = cKDTree(np.copy(points))
    toc = time.time() - tic
    print "KDtree build took", toc

    tic = time.time()
    # Note: Each query includes itself!
    distances, point_clusters = kdtree.query(points, 3)

    toc = time.time() - tic
    print "KDtree query took", toc

    tic = time.time()
    results = {
        'good_points': [],
        'normals': [],
        'degenerate': []
    }

    for n, group in enumerate(point_clusters):
        distance = distances[n]
        if np.std(distance) > 0.07:
            # To dispersed
            continue

        group_pts = points[group]
        cov = np.cov(group_pts.transpose())

        # Get the smallest eigen vector (the local plane normal)
        _, eigen_vectors = scl.eigh(cov, eigvals=(0, 0))
        results['normals'].append(eigen_vectors.transpose())
        results['good_points'].append(points[n])

    normals = np.vstack(results['normals'])
    good_points = np.vstack(results['good_points'])

    toc = time.time() - tic
    print "Normal build took", toc
    return good_points, normals


def compute_normals(points, leaf_size=0.05):
    """Estimate point cloud normals

    TODO:
        - Robust/Adaptive normal estimation
            --> adaptive leaf-size
            --> rejection by geometric consistency

        --> Accumulate points at different cloud samplings
        --> Accumulate points at different query ball sizes (different fractions of leaf size)

        x --> Uniform downsampling >><<
        --> For points that could not get a normal, inherit normals from other points in the scene

    """
    tic = time.time()
    # need copy?
    kdtree = cKDTree(np.copy(points))
    toc = time.time() - tic
    print "KDtree build took", toc

    tic = time.time()
    point_clusters = kdtree.query_ball_point(points, leaf_size)
    toc = time.time() - tic
    print "KDtree query took", toc

    # TODO: cython
    tic = time.time()

    results = {
        'normals': [],
        'good_points': [],
    }
    for n, group in enumerate(point_clusters):
        group_pts = points[group]
        # if group_pts.shape[0] > 3 and group_pts.shape[0] < 10:
        if group_pts.shape[0] > 3 and group_pts.shape[0] < 10:

            cov = np.cov(group_pts.transpose())

            # Get only the smallest eigenvector
            # _, eigen_vectors = scl.eigh(cov, eigvals=(0, 0))
            # normals.append(eigen_vectors.transpose())

            vals, eigen_vectors = scl.eigh(cov, eigvals=(0, 1))

            # Discard erratic normals
            confidence = (vals[1] / vals[0])
            if (100 > np.fabs(confidence) > 1):
                continue

            results['normals'].append(eigen_vectors.transpose()[0])
            results['good_points'].append(points[n, :])

    normals = np.vstack(results['normals'])
    good_points = np.vstack(results['good_points'])

    toc = time.time() - tic
    print "Assembling normals took", toc
    visualize.quiver3d(good_points, normals)
    return good_points, normals
