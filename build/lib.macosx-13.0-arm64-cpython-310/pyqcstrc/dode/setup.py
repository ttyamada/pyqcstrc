#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import os
import platform

from setuptools import setup, find_packages, Extension
from numpy import get_include

from Cython.Distutils import build_ext
from Cython.Build import cythonize

extensions = [
    Extension('math12',
    sources=['math12.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('symmetry12',
    sources=['symmetry12.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('utils12',
    sources=['utils12.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('numericalc12',
    sources=['numericalc12.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('strc12',
    sources=['strc12.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('intsct12',
    sources=['intsct12.pyx'],
    include_dirs=['.', get_include()]
    ),
]

comp_direct = {  # compiler_directives
        'language_level': 3,  # use python 3
        #'embedsignature': True,  # write function signature in doc-strings
}

extensions = cythonize(extensions,
                        compiler_directives=comp_direct)

setup(
    name = 'PyQCstrc.dode',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.6.0',
        'cython>=0.29.21',
    ],
    ext_modules = extensions,
    python_requires='>=3.7',
)
