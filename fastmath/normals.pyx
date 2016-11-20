import numpy as np
from scipy.spatial import cKDTree
cimport scipy
cimport numpy as np

DTYPE = np.float32
ctypedef np.int_t DTYPE_t

# def kdtree_query(pts, ):
#     kdtree = cKDTree(np.copy(points))
#     point_clusters = kdtree.query_ball_point(points, leaf_size)
def ncov(np.ndarray[DTYPE_t, ndim=2] pts, float leaf_size=0.01):
    kdtree = scipy.spatial.cKDTree(points)
    point_clusters = kdtree.query_ball_point(points, leaf_size)
