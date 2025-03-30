# test
import numpy as np
import cython
import random

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnndarray as qna
import prjop as prj

from numeric import (numeric_init,
                    check_intersection_segment_surface_numerical)

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

isys=3  # decagonal
crs.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qnm.qnmat_init()
prj.prjop_init()
numeric_init()

M0=qnn.zero()
M1=qnn.Qnnum([1,1,2])  # tau
M2=qnn.Qnnum([-1,0,1]) # -1
M3=qnn.pow(M1,-3)+qnn.Qnnum([7,0,2])  # 3.73607
M4=-qnn.pow(M1,-3)/2   #-0.19098
M5=qnn.pow(M1,2)/2     # 1.30903

vec1=np.array([M1,M2,M0],dtype=qnn.Qnnum)    
vec2=np.array([M3,M4,M5],dtype=qnn.Qnnum)

# line segment  (this should be Qnvec array)
ln=[vec1,vec2]
#ln=np.array([\
#[1.61803399, -1.,          0. ],\
#[3.73607, -0.19098,1.30902 ]])

M0=qnn.zero()          # 0
M1=qnn.Qnnum([1,1,2])  # tau        
M2=M1+qnn.Qnnum([3,0,1]) # 3.61804
M3=qnn.Qnnum([-1,0,1])   # -1
M4=qnn.pow(M1,-2)        # 0.38196
M5=qnn.pow(M1,2)         # 2.62804
M6=-M4                   # -0.38196
M7=qnn.Qnnum([1,0,1])    # 1

vec3=np.array([M2,M3,M4],dtype=qnn.Qnnum)    
vec4=np.array([M5,M6,M0],dtype=qnn.Qnnum)
vec5=np.array([M5,M0,M7],dtype=qnn.Qnnum)

# triangle vertices (this should be Qnvec array)
tr0=np.array([vec3,vec4,vec5])
tr=qna.anya(tr0,(3,3))
#tr=np.array([\
#[3.61803399, -1.,          0.38197],\
#[2.61803399, -0.38196601,  0.],\
#[2.61803399, 0,1. ]])

a=check_intersection_segment_surface_numerical(ln,tr)
print(a)
    