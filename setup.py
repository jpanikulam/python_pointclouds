# from distutils.core import setup
from Cython.Build import cythonize

# setup(
#     ext_modules=cythonize("helloworld.pyx")
# )
import setuptools

description = "A Python implementation of many common point-cloud"


setuptools.setup(
    name='clouds',
    version='0.1',
    license='MIT',
    long_description=__doc__,
    url='jakepanikul.am',
    author_email='jpanikul@gmail.com',
    packages=setuptools.find_packages(),
    description=description,
    keywords="pointcloud pcl",
    platforms='any',
    ext_modules=cythonize("fastmath/normals.pyx")
    # zip_safe=True
)
