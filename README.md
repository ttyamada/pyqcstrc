# PyQCstrc - Python library for Quasi-Crystal structure

### Main websites:

- https://github.com/ttyamada/PyQCstrc


### Installation using pip

`pip install pyqcstrc`

### Supported versions

- Python: 3.9
- Operating systems: Linux, macOS, and Windows


### Requirements

- Cython>=0.29.21
- numpy>=1.20.1
- scipy>=1.6.0

### Python example

```python

# import libraries
import numpy as np
import pyqcstrc.icosah.occupation_domain as od
import pyqcstrc.icosah.two_occupation_domains as ods

# Vertices of tetrahedron, v0,v1,v2,v3, which
# defines the asymmetric part.
v0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],
               [ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) 
v1 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],
               [-1, 0, 2],[-1, 0, 2],[-1, 0, 2]]) 
v2 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],
               [ 1, 0, 2],[-1, 0, 2],[-1, 0, 2]]) 
v3 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],
               [ 0, 0, 2],[-1, 0, 2],[ 0, 0, 2]]) 
seed = np.vstack([v0,v1,v2,v3]).reshape(4,6,3)
od0 = od.as_it_is(seed)

# Creating the basic occupation domain (OD) from its asymmetric unit 
# by applying the symmetry operations of m35 around the origin:
# generate symmetric OD, symmetric centre is v0.
od1 = od.symmetric(od, v0)

# Creating the basic OD at (1,0,0,0,0,0):
# 6D coordinates of position_1
pos1 = np.array([[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],
                 [ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) 
od2 = od.shift(od1,pos1)  # move to position_1

# Intersection operation on od1 and od2:
od3 = ods.intersection(od1, od2)

# Export od3 in .vesta and .xyz files:
od.write(od3, basename='od3', format='vesta')
od.write(od3, basename='od3', format='xyz')
```

### Citation:

If you use this tool in a program or publication, please acknowledge its
author:

```bibtex
@article{pyqcstrc,
  title     = {PyQCstrc.icosah: A computing package for structure model of icosahedral quasicrystals},
  author    = {Yamada, Tsunetomo},
  journal   = {},
  volume    = {},
  number    = {},
  pages     = {},
  year      = {},
  publisher = {},
  version   = {},
  doi       = {},
  url       = {}
}
```


### License
This work is licensed under an [MIT license](https://mit-license.org/).
