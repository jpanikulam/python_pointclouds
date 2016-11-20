"""Load and cache meshes.

Nice mesh website: http://3d.csie.ntu.edu.tw/~dynamic/database/index.html
"""
from vispy import io
import os
import numpy as np

this_path = os.path.dirname(os.path.realpath(__file__))


def load_and_cache_mesh(file_path, url=None):
    '''Load and save a binary-cached version of a mesh'''
    vertices, faces, normals, texcoords = io.read_mesh(file_path)
    if url is None:
        np.savez(file_path, vertices=vertices, faces=faces, normals=normals)
    else:
        np.savez(file_path, vertices=vertices, faces=faces, normals=normals, url=url)
    return vertices, faces, normals, texcoords


def load_from_cache(path):
    array_data = np.load(path + '.npz')
    vertices = array_data['vertices']
    faces = array_data['faces']
    normals = array_data['normals']
    texcoords = None
    return vertices, faces, normals, texcoords


def load_mesh(path):
    if path in os.listdir(os.path.join(this_path, 'meshes')):
        path = os.path.realpath(os.path.join(this_path, 'meshes', path))

    dir_, file_ = os.path.split(path)

    # It's not a file we downloaded, load it normally
    if os.path.exists(path + '.npz'):
        return load_from_cache(path)

    elif os.path.exists(path):
        print 'Loading and then caching {}'.format(path)
        return load_and_cache_mesh(path)

    else:
        raise Exception('Something went wrong, we could not load a mesh at {}'.format(file_))


catalog = {
    'table': os.path.join(this_path, 'meshes', 'Y3991_table9.obj'),
    'stealth': os.path.join(this_path, 'meshes', 'Y3962_stealth.obj'),
}

if __name__ == '__main__':
    path = os.path.join(this_path, 'meshes', 'Y3991_table9.obj')
    mesh = load_mesh(path)
