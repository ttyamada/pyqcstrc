PyQCstrc: python library for quasi-crystal structure
====================================================

Introduction
============
PyQCstrc is a Python library for quasicrystal structure. It provides tools commonly used to build initial structure of quasicrystals, which includes translation and symmetry operations in the nD (n = 5 or 6) space, and intersection operations on occupation domains in (n-3)D space, perpendicular to 3D physical space.

PyQCstrc supports export and import of the occupation domains: the occupation domains can be exported in VESTA format (.vesta) and XYZ format (.xyz), and these can be visualized by utilizing `VESTA <https://jp-minerals.org/vesta/en/>`_ (Momma & Izumi, 2011, J. Appl. Crystallogr., 44, 1272-1276.). The latter can be imported so that users can recall their occupation domains in Python scripts.

Basic usage of the PyQCstrc is described by providing Python scripts to obtain a rhombic icosahedron occupation domain. See :doc:`Examples <examples/index>`.

Changelog
=========
.. include:: ../CHANGELOG.rst
  :end-before: Version 0.0.1

See the full :doc:`Changelog<changelog>`

License
=======
.. license section

The PyQCstrc library is distributed with a MIT license.
See https://opensource.org/licenses/mit-license.php

.. license end

Download
========
.. download section

PyQCstrc is available from:
 * https://pypi.python.org/pypi/pyqcstrc

.. download end

PyQCstrc.ico: icosahedral quasicrystals
=======================================
.. pyqcstrc.ico section

Description
-----------
This modules provides algorithm for building 6D structural models of the icosahedral quasicrystals:

* Translation operation of occupation domain
* Symmetry operations of occupation domain
* Intersection of two occupation domains

.. pyqcstrc.ico end

PyQCstrc.deca: decagonal quasicrystals
======================================
.. pyqcstrc.deca section

Under construction

.. pyqcstrc.deca end


PyQCstrc.dode: dodecagonal quasicrystals
========================================
.. pyqcstrc.dode section

Under construction

.. pyqcstrc.dode end
