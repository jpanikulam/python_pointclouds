import visualize
import time
import numpy as np

import load
import geometry
import clouds


if __name__ == '__main__':
    mesh = load.load_mesh('Y3043_Finch.obj')

    vertices = mesh[0]
    faces = mesh[1]
    visualize.mesh(mesh)
    visualize.quiver3d(vertices, mesh[2])

    visualize.show()
