.. toctree::
   :maxdepth: 2

.. _examples:


Examples
=================

Basic usage of the PyQCstrc is described by providing Python scripts to obtain a rhombic icosahedron occupation domain.

`example1.py`:
^^^^^^^^^^^^^^^^^
The simple script to obtain a rhombic icosahedron occupation domain, as a set of tetrahedra.

.. literalinclude:: ../../pyqcstrc/examples/example1.py
   :language: text
   
`example2.py`:
^^^^^^^^^^^^^^^^^
Since the rhombic icosahedron forms a convex polyhedron, it is tetrahedralizable by 3D Delaunay triangulation.

.. literalinclude:: ../../pyqcstrc/examples/example2.py
   :language: text

`example3.py`:
^^^^^^^^^^^^^^^^^
The simple script to obtain a portion of rhombic triacontahedron inside the asymmetric unit of 5m at (1, 0, 0, 0, 0, 0)/2.

.. literalinclude:: ../../pyqcstrc/examples/example3.py
   :language: text
