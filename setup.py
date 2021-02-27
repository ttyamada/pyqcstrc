#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import os
import platform

from setuptools import setup, find_packages, Extension
from numpy import get_include

from pyqcstrc.version import __version__, __authors__, __copyright__, __license__, __date__, __docformat__

from Cython.Distutils import build_ext
from Cython.Build import cythonize

__package_name__ = 'pyqcstrc'

extensions = [
    Extension('pyqcstrc.icosah.math1', 
    sources=['pyqcstrc/icosah/math1.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.numericalc', 
    sources=['pyqcstrc/icosah/numericalc.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.symmetry', 
    sources=['pyqcstrc/icosah/symmetry.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.utils', 
    sources=['pyqcstrc/icosah/utils.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.intsct', 
    sources=['pyqcstrc/icosah/intsct.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.mics', 
    sources=['pyqcstrc/icosah/mics.pyx'],
    include_dirs=['.', get_include()]
    )
]

extensions = cythonize(extensions,compiler_directives={'language_level':"3"})

long_description = open('README.md').read()

setup(
    name=__package_name__,
    version=__version__,
    description="PyQCstrc provides Python tools for Quasi-Crystal structure.",
    long_description = long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    license = __license__,
    #scripts = ['pyqcstrc/icosah/occupation_domain.py',
    #            'pyqcstrc/icosah/two_occupation_domains.py'],
    author='Tsunetomo Yamada',
    #author_email='tsunetomo.yamada@rs.tus.ac.jp',
    url = "https://github.com/ttyamada/PyQCstrc",
    ext_modules = extensions,
    #packages=find_packages(),
    packages = [
        'pyqcstrc/icosah',
        'pyqcstrc/examples',
        'pyqcstrc/xyz',
        'pyqcstrc/scripts'
    ],
    include_package_data=True,
    #scripts=[],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.20.1',
        'scipy>=1.6.0',
        'cython>=0.29.21',
    ]
)
