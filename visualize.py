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


def color_points3d(x, scalars, **kwargs):
    nodes = points3d(x, **kwargs)
    nodes.glyph.scale_mode = 'scale_by_vector'
    # nodes.mlab_source.dataset.point_data.vectors = np.tile(np.random.random((5000,)), (3, 1))
    if 'scale_factor' in kwargs.keys():
        nodes.mlab_source.dataset.point_data.vectors = np.ones(x.shape) * kwargs['scale_factor']
    nodes.mlab_source.dataset.point_data.scalars = scalars
    return nodes


def show():
    """So you don't have to import mlab."""
    mlab.show()
