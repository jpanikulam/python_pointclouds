"""Vectorized numpy geometry functions."""
import numpy as np


def unitize(vec, order=2):
    """A point."""
    return vec / np.linalg.norm(vec, order)


def vunitize(vecs, order=2):
    """Each row is a point."""
    return vecs / np.linalg.norm(vecs, order, axis=1)[:, None]


def vnormal_from_faces(vertices, faces):
    """Vectorized normal from a matrix of vertices and an index list."""
    # Group num, index in group, (x y z)
    points = vertices[faces]
    ab = points[:, 0, :] - points[:, 1, :]
    ac = points[:, 0, :] - points[:, 2, :]
    n = np.cross(ab, ac)
    return vunitize(n)


def vcentroids_from_faces(vertices, faces):
    """Vectorized centroid of faces

    where faces is an array of index triplets
    """
    return np.mean(vertices[faces], axis=1)


def interpolate_faces(vertices, faces, amt=50):
    """Generate valid convex combinations of triads."""

    alpha_set = vunitize(np.random.random((amt, 3)), order=1)

    intf_list = []
    for n, face in enumerate(faces):
        pts = vertices[face]
        intf_list.append(alpha_set.dot(pts))

    intf = np.vstack(intf_list)
    return intf


def vunique_rows(data):
    ncols = data.shape[1]
    dtype = data.dtype.descr * ncols
    struct = data.view(dtype)

    uniq, cts = np.unique(struct, return_counts=True)
    uniq = uniq.view(data.dtype).reshape(-1, ncols)
    return uniq, cts
