import visualize
import time
import numpy as np

import load
import geometry
import clouds


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
    visualize.points3d(downsampled_points, scale_factor=voxel_size * 0.1, color=(0.0, 1.0, 0.0))
    return downsampled_points


if __name__ == '__main__':
    mesh = load.load_mesh('Y3043_Finch.obj')

    vertices = mesh[0]
    faces = mesh[1]

    # pts = np.random.random((10000, 3)) * 5
    intf = geometry.interpolate_faces(vertices, faces, amt=5)
    tic = time.time()

    downsampled = uniform_voxelgrid_sample(intf, voxel_size=0.05)
    toc = time.time() - tic
    print toc
    clouds.compute_normals(downsampled, leaf_size=0.05)

    # visualize.points3d(vertices, scale_factor=0.1)
    visualize.show()
