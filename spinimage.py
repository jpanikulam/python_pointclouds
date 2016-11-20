import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa
import cv2
from mayavi import mlab
import load
import time

# pcl
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
            mlab.show()

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

    # plt.figure(3)
    # plt.imshow(image)
    # plt.show()


def rpq(im1, im2):
    """Linear correlation of two images."""
    p = np.squeeze(np.reshape(im1, (-1, 1))) / np.max(im1)
    q = np.squeeze(np.reshape(im2, (-1, 1))) / np.max(im2)
    # import IPython; IPython.embed()

    N = 10 * 25
    qsum = np.sum(q)

    correlation = (N * np.dot(p, q)) - (np.sum(p) * np.sum(q))
    # print correlation

    p_normalizer = (N * np.sum(np.square(p))) - np.square(np.sum(p))
    q_normalizer = (N * np.sum(np.square(q))) - np.square(np.sum(q))

    normalization = np.sqrt(p_normalizer * q_normalizer)
    # print normalization, p_normalizer, q_normalizer
    return correlation / normalization


def spin(vertices, point, direction):
    """Cylindrical deprojection about a point and normal.

    TODO: occlusion
    """
    difference = (vertices - point).transpose()
    xmp = np.linalg.norm(vertices - point, axis=1) ** 2

    projection = (direction.dot(difference)) ** 2

    xmp_m_proj = xmp - projection
    if np.any(xmp_m_proj < 0):
        print 'Encountered invalid vertex in spinimage.'
        xmp_m_proj[xmp_m_proj < 0] = 0

    alpha = np.sqrt(xmp - projection)
    beta = direction.dot(difference)

    spin_points = np.vstack((alpha, beta)).transpose()
    spin_image = render_points(spin_points, 25, 10)
    return spin_image

if __name__ == '__main__':
    mesh = load.load_mesh(load.catalog['stealth'])

    vertices = mesh[0]

    upsample_mesh(mesh)
