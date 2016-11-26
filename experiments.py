from matplotlib import pyplot as plt

import visualize
import time
import numpy as np

import load
import clouds
import geometry

from mayavi import mlab


def error(a, b):
    return np.linalg.norm(a - b, axis=1) ** 2


def derror(a, b):
    return np.vstack([
        -2 * a[:, 0] + 2 * b[:, 0],
        -2 * a[:, 1] + 2 * b[:, 1],
        -2 * a[:, 2] + 2 * b[:, 2],
    ]).transpose()


def err_cost(a, b):
    return np.sum(geometry.huber(error(a, b), delta=0.2), axis=0)


def derr_cost(a, b):
    db_ddelta = a + b

    # return np.sum(geometry.dhuber(2 * - ))


def dT(x, y):
    print err_cost(x, y)
    jacobian = np.zeros((3, 3))
    # for pt in y:
    for n in range(y.shape[0]):
        y_n = y[n]
        x_n = x[n]
        jacobian += -geometry.so3.skew_symmetrize(y_n)

    print jacobian
    return jacobian


def incremental_transform(from_vertices, to_vertices):
    """Robust registration of `from_vertices` onto `to_vertices`

    (Assuming correspondence)
    """

    T = np.identity(3)
    delta = np.array([0.1, 0.0, 0.1])
    geometry.so3.exp(delta)

    cost = err_cost(to_vertices, from_vertices)


    # differential_transform = dT(to_vertices, from_vertices)
    # dT.dot()


@mlab.animate
def anim():
    f = mlab.gcf()
    k = 0.9

    R0 = geometry.so3.exp(np.array([0.0, 0.0, 1.0]) * k)
    v1 = visualize.points3d(vertices.dot(R0.transpose()), scale_factor=0.01, color=(0.0, 1.0, 0.0))

    for a in range(20):
        k += 0.1
        R = geometry.so3.exp(np.array([0.0, 0.0, 1.0]) * k)

        rotated_pts = vertices.dot(R.transpose())
        visualize.update(v1, rotated_pts)
        print error(rotated_pts, vertices)
        yield


if __name__ == '__main__':
    mesh = load.load_mesh('Y3043_Finch.obj')
    vertices = mesh[0]

    R0 = geometry.so3.exp(np.array([0.0, 0.0, 1.0]) * 1.5)
    rotated_pts = vertices.dot(R0.transpose())
    incremental_transform(rotated_pts, vertices)

    # visualize.points3d(vertices, scale_factor=0.01, color=(1.0, 0.0, 0.0), opacity=0.6)
    # an = anim()  # Starts the animation.
    # visualize.show()
