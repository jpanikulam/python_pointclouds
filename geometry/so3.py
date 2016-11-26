import numpy as np


def natlog(R):
    """ln: SO3 -> so3"""
    theta = np.arccos((np.trace(R) - 1) / 2.0)
    theta_factor = theta / (2 * np.sin(theta))
    r_factor = R - R.transpose()
    ln = theta_factor * r_factor
    return ln


def exp(w):
    """exp: so3 -> SO3

    TODO: use taylor expansion instead when theta is small."""
    wx = skew_symmetrize(w)
    wx2 = wx.dot(wx)
    theta = np.linalg.norm(w)

    sin_term = (np.sin(theta) / theta) * wx
    cos_term = ((1 - np.cos(theta)) / (theta ** 2)) * wx2

    result = np.identity(3) + sin_term + cos_term
    return result


def dexp(w):
    wx = skew_symmetrize(w)
    wx2 = wx.dot(wx)
    norm_w = np.linalg.norm(w)
    norm_w2 = norm_w ** 2

    cos_term = ((1 - np.cos(norm_w)) / (norm_w2)) * wx
    sin_term = ((1 - (np.sin(norm_w) / norm_w)) / norm_w2) * wx2
    result = np.identity(3) + cos_term + sin_term
    return result


def dnatlog(w):
    theta = np.linalg.norm(w)
    atheta = (np.sin(theta) / theta)
    btheta = (1 - np.cos(theta)) / (theta ** 2)
    ctheta = (1 - atheta) / (theta ** 2)

    return (btheta - (0.5 * atheta)) / (1 - np.cos(theta))


def skew_symmetrize(w):
    wx = np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0]
    ])
    return wx


def deskew(wx):
    return np.array([wx[2, 1], wx[0, 2], wx[1, 0]])


def skew_product(w, v):
    """Compute skew_symmetrize(w).dot(v)."""
    return np.array([
        -v[1] * w[2] + v[2] * w[1], v[0] * w[2] - v[2] * w[0], -v[0] * w[1] + v[1] * w[0]
    ])


def test_so3():
    w_0 = np.array([1.0, 1.5, 2.2])
    e = exp(w_0)
    d = deskew(natlog(e))
    np.testing.assert_almost_equal(w_0, d)

    eps = np.array([0.0, 0.0, 0.01])
    print exp(w_0)
    print exp(w_0 + eps)
    print exp(w_0) + (dexp(w_0).dot(eps))


if __name__ == '__main__':
    test_so3()

    # print skew_symmetrize([1.0, 2.0, 3.0]).dot(np.array([1.0, 1.0, 1.0]))
    # print skew_product([1.0, 2.0, 3.0], np.array([1.0, 1.0, 1.0]))

    delta = np.array([0.5, 0.1, 0.1])

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 5.0, -7.0])
    print np.cross(a, b)
    print np.cross(a, b + delta)

    dcross = skew_symmetrize(a)
    print np.cross(a, b) + dcross.dot(delta)
