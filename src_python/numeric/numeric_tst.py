#if __name__ == '__main__':

# test
import numpy as np
import cython
import random

def generate_random_value():
    """ generate value in TAU-style  # qnnu
    """
    nmax=10 # maximum int
    v=np.zeros((3),dtype=np.int64)
    for i1 in range(2): # random int v[1] v[2]
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

def generate_random_triangle():
    return generate_random_vectors(3)

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
triangle=generate_random_triangle() # in TAU-style
triangle_num=numerical_vectors(triangle) # in float
#print(triangle_num)
area=triangle_area_6d_numerical(triangle_num)
print(area)
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
    