from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "CIC",
    ext_modules = cythonize('cpy.pyx'),  # accepts a glob pattern
)
