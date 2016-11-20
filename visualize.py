from mayavi import mlab


def quiver3d(x, n, **kwargs):
    mlab.quiver3d(
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

    mlab.points3d(
        x[:, 0],
        x[:, 1],
        x[:, 2],
        **kwargs
    )


def triangular_mesh(pts, faces, **kwargs):
    mlab.triangular_mesh(pts[:, 0], pts[:, 1], pts[:, 2], faces, **kwargs)


def mesh(mesh, **kwargs):
    triangular_mesh(mesh[0], mesh[1], **kwargs)


def show():
    """So you don't have to import mlab."""
    mlab.show()
