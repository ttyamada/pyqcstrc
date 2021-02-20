from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

"""
ext_modules = cythonize(Extension("qcmath", 
                        sources=["qcmath.pyx"],
                        include_dirs=['.', get_include()]),
                        compiler_directives={'language_level':"3"})
setup(name="qcmath", ext_modules = ext_modules)
"""

ext_modules = [
                Extension("test", 
                sources=["test.pyx"],
                include_dirs=['.', get_include()]),
                Extension("test2", 
                sources=["test2.pyx"],
                include_dirs=['.', get_include()])
                ]

setup(name="test",
        ext_modules = cythonize(ext_modules,compiler_directives={'language_level':"3"})
)
