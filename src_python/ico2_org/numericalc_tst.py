#if __name__ == '__main__':

# test
import numpy as np
from numpy.typing import NDArray
import random
from numericalc import (
    inside_outside_tetrahedron,
    projection3_numerical
    )

def generate_random_value():
    """ generate value in TAU-style
    """
    nmax=10
    v=np.zeros((3),dtype=np.int64)
    for i1 in range(2):
        v[i1]=random.randrange(-nmax,nmax) # a and b in (a+b*TAU)/c.
    v[2]=random.randrange(1,nmax) # c in (a+b*TAU)/c.
    return v
    
def generate_random_vector(ndim=6):
    """ generate ndim vector in TAU-style
    ndim: dimension of vectors
    """
    nmax=10
    v=np.zeros((ndim,3), dtype=np.int64)
    for i1 in range(ndim):
        v[i1]=generate_random_value()
    return v
    
def generate_random_vectors(n,ndim=6):
    """
    num: number of generated vectors.
    ndim: dimension of vectors
    """
    v=np.zeros((n,ndim,3), dtype=np.int64)
    for i1 in range(n):
        v[i1]=generate_random_vector(ndim)
    return v

def generate_random_tetrahedron():
    return generate_random_vectors(4)

"""
print('check projection')

nset=2
vst=generate_random_vectors(nset)
vsn=numerical_vectors(vst)
#
# TAU-style
vset=get_internal_component_sets_numerical(vst)
for v in vset:
    print(v)
# float
viset=projection3_sets_numerical(vsn)
for v in viset:
    print(v)
vset=projection_sets_numerical(vsn)
for v in vset:
    print(v)
"""

"""
print('check tetrahedron')
tetrahedron=generate_random_tetrahedron() # in TAU-style
tetrahedron_num=numerical_vectors(tetrahedron) # in float
#print(tetrahedron_num)
vol=tetrahedron_volume_6d_numerical(tetrahedron_num)
print(vol)
"""

"""
ln=np.array([\
[1.61803399, -1.,          0. ],\
[3.73607, -0.19098,1.30902 ]])

tr=np.array([\
[3.61803399, -1.,          0.38197],\
[2.61803399, -0.38196601,  0.],\
[2.61803399, 0,1. ]])

a=check_intersection_segment_surface_numerical(ln,tr)
print(a)
"""

eps=1e-3
tetrahedron=np.array([[ -1, -1, -1],\
                      [  1,  1, -1],\
                      [ -1,  1,  1],\
                      [  1, -1,  1]])
#point=np.array([0,0,np.sqrt(3)+eps])
point=np.array([ -1+eps, -1+eps, -1+eps])
print('inside_outside_tetrahedron_rough:')
if inside_outside_tetrahedron_rough(point,tetrahedron):
    print(' inside')
else:
    print(' outside')

print('inside_outside_tetrahedron:')
if inside_outside_tetrahedron(point,tetrahedron):
    print(' inside')
else:
    print(' outside')

point = np.array([ 0.03, -0.03, -0.03, 0.00, -0.03, -0.02])
# Vertices of tetrahedron, v0,v1,v2,v3, which
# defines the asymmetric part.
v0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v1 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
v2 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
v3 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 0, 0, 2],[-1, 0, 2],[ 0, 0, 2]])
tetrahedron = np.vstack([v0,v1,v2,v3]).reshape(4,6,3)
point = projection3_numerical(point)
if inside_outside_tetrahedron_tau_v2(point,tetrahedron):
    print(' inside')
else:
    print(' outside')
print(point)
