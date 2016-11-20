from mayavi import mlab
import numpy as np


def quiver3d(x, n, **kwargs):
    return mlab.quiver3d(
        x[:, 0],
        x[:, 1],
        x[:, 2],
        n[:, 0],
        n[:, 1],
        n[:, 2],
        **kwargs
    )


def points3d(x, **kwargs):
    if 'color' in kwargs.keys():
        kwargs['color'] = tuple(kwargs['color'])

    # if it's a single point, just make it work
    if x.ndim == 1:
        x = np.reshape(x, (1, -1))

    return mlab.points3d(
        x[:, 0],
        x[:, 1],
        x[:, 2],
        **kwargs
    )


def triangular_mesh(pts, faces, **kwargs):
    return mlab.triangular_mesh(pts[:, 0], pts[:, 1], pts[:, 2], faces, **kwargs)


def mesh(mesh, **kwargs):
    return triangular_mesh(mesh[0], mesh[1], **kwargs)


def line(a, b, colors=None):
    a = np.array(a) * np.ones(1)
    b = np.array(b) * np.ones(1)

    if colors is None:
        colors = [(1.0, 1.0, 1.0)] * len(a)

    for n, (start, end) in enumerate(zip(a, b)):
        mlab.plot3d(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=tuple(colors[n])
        )


def color_points3d(x, scalars, **kwargs):
    nodes = points3d(x, **kwargs)
    nodes.glyph.scale_mode = 'scale_by_vector'
    if 'scale_factor' in kwargs.keys():
        nodes.mlab_source.dataset.point_data.vectors = np.ones(x.shape) * kwargs['scale_factor']
    nodes.mlab_source.dataset.point_data.scalars = scalars
    return nodes


def show(axis_scale=1.0):
    """So you don't have to import mlab."""
    diag = np.diag([1.0, 1.0, 1.0])
    line(np.zeros((3, 3)), diag * axis_scale, colors=diag)
    mlab.show()
