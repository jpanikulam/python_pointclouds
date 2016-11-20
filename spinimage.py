import numpy as np
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa
import cv2
import load
import time

import visualize
import geometry


def upsample_mesh(mesh):
    (vertices, faces, normals, _) = mesh

    # face_centroids = geometry.vcentroids_from_faces(vertices, faces)
    intf = geometry.interpolate_faces(vertices, faces)

    return intf


def analyze_spins(vertices, faces, intf):
    # cv2.imshow('z', cv2.resize(spin_image / np.max(spin_image), (500, 500)))
    # cv2.waitKey(500)
    spins = []
    vn = []
    # for n, face in enumerate(faces):
    #     vertex = vertices[face[0]]
    #     normal = face_normals[n]
    #     spin_image = spin(intf, vertex, normal)
    #     vn.append(vertex)
    #     spins.append(spin_image)

    first_spin = spin(intf, vertex, normal)

    # visualize.quiver3d(face_centroids, -face_normals)
    for n, _ in enumerate(spins):
        # for k, _ in enumerate(spins):
        k = 0
        corr = rpq(spins[n], spins[k])
        print "{}: {}".format(n, k), corr
        if corr > 0.8:
            visualize.points3d(intf, scale_factor=0.6, color=(1.0, 0.0, 0.0))
            visualize.triangular_mesh(vertices, faces)

            visualize.points3d(np.reshape(vn[k], (1, -1)), scale_factor=2.0, color=(0.0, 1.0, 1.0))
            visualize.points3d(np.reshape(vn[n], (1, -1)), scale_factor=2.0, color=(0.0, 0.0, np.clip(corr, 0.0, 1.0)))


def render_points(points, scale, width):
    """Assuming 2d points.

    NOTE:
        - The list loop version permits only 160 Hz
        - The numpy version (which cannot accumulate) can run at 5,000 Hz
        - The add.at version allows 1,400 Hz

    :param points: Array of points
    :param units: The scale of the image (in units of, for example, meters)
    :param width: The width of the image in pixels
    """
    image = np.zeros((width, width * 2), dtype=np.float64)

    # _min = np.min(points, axis=0)
    # _max = np.max(points, axis=0)
    # sw = scale / float(width)
    # normalized_points = ((points - _min) / (_max - _min))
    # scaled_points = normalized_points * (width - 1)
    # indices = np.int0(scaled_points)

    normalized_points = (points / scale)

    settable_indices = np.all(np.fabs(normalized_points) < 1.0, axis=1)
    settable_points = normalized_points[settable_indices] * (width - 1)
    settable_points[:, 1] += width
    indices = np.int0(settable_points)

    # 1.4 kHz max
    np.add.at(image, (indices[:, 0], indices[:, 1]), 1)
    return image


def spin(vertices, point, direction, scale=25):
    """Cylindrical deprojection about a point and normal.

    TODO: occlusion
    TODO: maximum support angle
    """
    difference = (vertices - point).transpose()
    xmp = np.linalg.norm(vertices - point, axis=1) ** 2

    projection = (direction.dot(difference)) ** 2

    xmp_m_proj = xmp - projection
    if np.any(xmp_m_proj < 0):
        print 'Encountered invalid vertex in spinimage.'
        xmp_m_proj[xmp_m_proj < 0] = 0

    alpha = np.sqrt(xmp_m_proj)
    beta = direction.dot(difference)

    spin_points = np.vstack((alpha, beta)).transpose()
    spin_image = render_points(spin_points, scale, 10)
    return spin_image


def rpq(im1, im2):
    """Linear correlation of two images."""
    p = np.squeeze(np.reshape(im1, (-1, 1)))
    q = np.squeeze(np.reshape(im2, (-1, 1)))

    N = p.size

    qsum = np.sum(q)
    psum = np.sum(p)

    correlation = (N * np.dot(p, q)) - (psum * qsum)

    p_normalizer = (N * np.sum(np.square(p))) - np.square(psum)
    q_normalizer = (N * np.sum(np.square(q))) - np.square(qsum)

    normalization = np.sqrt(p_normalizer * q_normalizer)
    return correlation / normalization


def build_spinimages(vertices, normal_vertices, normals, scale=25):
    images = []
    for n in range(normal_vertices.shape[0]):
        image = spin(vertices, normal_vertices[n], normals[n], scale=scale)
        images.append(image)
    return images


def show_spinimage(image):
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    import clouds
    mesh = load.load_mesh('Y3043_Finch.obj')

    vertices = mesh[0]
    faces = mesh[1]

    # Generate trainable data from a model
    # intf = geometry.interpolate_faces(vertices, faces, amt=5)
    # leaf = 0.05
    # downsampled_intf = clouds.uniform_voxelgrid_sample(intf, voxel_size=leaf * 0.5)
    # normal_pts, normals = clouds.robust_normals(downsampled_intf, leaf_size=leaf)
    normal_pts = vertices
    normals = mesh[2]

    # Build spinimages
    tic = time.time()
    spin_images = build_spinimages(vertices, normal_pts, normals, scale=0.2)
    toc = time.time() - tic
    print 'Spin image library construction took {} seconds'.format(toc)

    # Choose a test image
    # test_spin = spin(intf, normal_pts[0], normals[0])
    n = np.random.randint(len(spin_images))
    test_spin = spin_images[n]

    tic = time.time()
    similarity = np.array(map(lambda q: rpq(test_spin, q), spin_images))
    toc = time.time() - tic
    print 'Similarity estimation took {} seconds'.format(toc)
    print np.min(similarity), np.max(similarity)
    similarity = np.clip(similarity, 0.0, 1.0)

    visualize.color_points3d(normal_pts, similarity, scale_factor=0.1)
    visualize.points3d(normal_pts[n], scale_factor=0.05, opacity=0.7)

    visualize.show()
