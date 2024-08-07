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

VERSION="0.0.2a11"

# Package meta-data.
NAME = "pyqcstrc"
DESCRIPTION = "PyQCstrc - Python library for Quasi-Crystal structure"
URL = "https://www.rs.tus.ac.jp/tsunetomo.yamada/pyqcstrc"
AUTHOR = "Tsunetomo Yamada"
EMAIL = "tsunetomo.yamada@rs.tus.ac.jp"
REQUIRES_PYTHON = ">=3.7.0"

REQUIRED=[
    'numpy>=1.20.0',
    'scipy>=1.6.0',
]

EXTRAS = {
    "dev": [
        # Jupyter notebook
        "notebook",
        "matplotlib",
        "ipython",
        "ipykernel",
    ],
}

comp_direct = {  # compiler_directives
        'language_level': 3,  # use python 3
        #'embedsignature': True,  # write function signature in doc-strings
}

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    license="MIT",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["pyqcstrc"]),
    package_data={},
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    zip_safe=False,
    #test_suite="tests",
    #use_scm_version=True,
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
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
