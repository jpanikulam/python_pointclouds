import numpy as np
import time
import pyximport
pyximport.install()  # noqa
import normals


if __name__ == '__main__':
    tic = time.time()
    points = np.random.random((3, 500))
    toc = time.time() - tic

