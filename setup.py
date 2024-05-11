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

from pyqcstrc.version import __version__, __authors__, __copyright__, __license__, __date__, __docformat__

from Cython.Distutils import build_ext
from Cython.Build import cythonize

__package_name__ = 'pyqcstrc'

extensions = [
    Extension('pyqcstrc.ico.math1', 
    sources=['src/pyqcstrc/ico/math1.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.ico.numericalc', 
    sources=['src/pyqcstrc/ico/numericalc.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.ico.symmetry', 
    sources=['src/pyqcstrc/ico/symmetry.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.ico.utils', 
    sources=['src/pyqcstrc/ico/utils.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.ico.intsct', 
    sources=['src/pyqcstrc/ico/intsct.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.ico.mics', 
    sources=['src/pyqcstrc/ico/mics.pyx'],
    include_dirs=['.', get_include()]
    ),
    #Extension('pyqcstrc.ico.reciprocal', 
    #sources=['src/pyqcstrc/ico/reciprocal.pyx'],
    #include_dirs=['.', get_include()]
    #),
    Extension('pyqcstrc.dode.symmetry12',
    sources=['src/pyqcstrc/dode/symmetry12.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.dode.utils12',
    sources=['src/pyqcstrc/dode/utils12.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.dode.math12',
    sources=['src/pyqcstrc/dode/math12.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.dode.numericalc12',
    sources=['src/pyqcstrc/dode/numericalc12.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.dode.strc12',
    sources=['src/pyqcstrc/dode/strc12.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.dode.intsct12',
    sources=['src/pyqcstrc/dode/intsct12.pyx'],
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
    name=__package_name__,
    version=__version__,
    description="PyQCstrc provides Python library for Quasi-Crystal structure.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    license = __license__,
    #scripts = ['pyqcstrc/ico/occupation_domain.py',
    #            'pyqcstrc/ico/two_occupation_domains.py'],
    author='Tsunetomo Yamada',
    author_email='tsunetomo.yamada@rs.tus.ac.jp',
    url = "https://www.rs.tus.ac.jp/tsunetomo.yamada/pyqcstrc",
    #url = "https://github.com/ttyamada/PyQCstrc",
    #
    # Comment out when including ico and dode
    #ext_modules = extensions,
    #
    #
    #packages=find_packages(),
    packages = [
        'src/pyqcstrc/ico2',
        #
        # Comment out when including ico and dode
        #'src/pyqcstrc/ico',
        #'src/pyqcstrc/dode',
        #
        #
        #'src/pyqcstrc/deca',
        #'src/pyqcstrc/ico/examples',
        #'src/pyqcstrc/ico/xyz',
        #'src/pyqcstrc/ico/tests',
    ],
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.6.0',
        'cython>=0.29.21',
    ]
)
