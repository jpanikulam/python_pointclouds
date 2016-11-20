"""Common Non-Geometric Point Cloud Math.

Includes:
    - Robust loss functions
"""
import numpy as np


def huber(x, delta):
    results = np.zeros(x.shape)
    abs_x = np.fabs(x)

    le = abs_x <= delta
    results[le] = np.square(x[le]) * 0.5
    results[~le] = delta * (abs_x[~le] - (delta * 0.5))

    return results


def dhuber(x, delta):
    results = np.zeros(x.shape)
    abs_x = np.fabs(x)

    le = abs_x <= delta
    results[le] = x[le]
    results[~le] = delta * np.sign(x[~le])

    return results


def tukey(x, c):
    results = np.zeros(x.shape)
    abs_x = np.fabs(x)

    le = abs_x <= c
    c2_over_6 = (c ** 2) / 6

    inner = 1 - ((x[le] / c) ** 2)
    results[le] = c2_over_6 * (1 - (inner ** 3))
    results[~le] = c2_over_6

    return results


def dtukey(x, c):
    results = np.zeros(x.shape)
    abs_x = np.fabs(x)

    le = abs_x <= c
    inner = 1 - ((x[le] / c) ** 2)
    results[le] = x[le] * (inner ** 2)
    results[~le] = 0

    return results


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    r = np.linspace(-10, 10, 1000)
    plt.plot(r, huber(r, 1.0), 'g')
    plt.plot(r, dhuber(r, 1.0), 'g--')

    plt.plot(r, tukey(r, 1.0), 'r')
    plt.plot(r, dtukey(r, 1.0), 'r--')

    plt.show()
