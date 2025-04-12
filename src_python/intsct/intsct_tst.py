import numpy as np
from numpy.typing import NDArray
import time # in object_subtraction_dev1, tetrahedron_not_obj
import itertools
import cython

import crsys as crs
import intsct as isct
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import numeric as num
import qnndarray as qna
import prjop as prj

#if __name__ == '__main__':

# test

import random

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

def generate_random_triangle():
    return generate_random_vectors(3)

#segment_1=np.array(\
#[[[1, 0, 1],[1, 0, 1],[0, 0, 1],[0, 0, 1],[0, 0, 1],[0, 0, 1]],\
#[[3, 0, 2],[1, 0, 2],[0, 0, 1],[0, 0, 1],[0, 0, 1],[0, 0, 1]]]) # edge1
  
#segment_2=np.array(\
#[[[ 0,  0,  1],[ 0,  0,  1],[ 0,  0,  1],[ 1,  0,  1],[ 0,  0,  1],[ 0,  0,  1]],\
#[[ 0,  0,  1],[ 0,  0,  1],[-1,  0,  2],[ 1,  0,  2],[ 0,  0,  1],[ 0,  0,  1]]]) # edge2

isys=4  # for octabonal
crs.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qnm.qnmat_init()
prj.prjop_init()
qna.qnndarray_init()

# M0=qnn.Qnnum([0,0,1])
# M1=qnn.Qnnum([1,0,1])
# M2=qnn.Qnnum([3,0,2])
# M3=qnn.Qnnum([1,0,2])
# M4=qnn.Qnnum([-1,0,2])

# seg_1=np.array([
#         [M1,M1,M0,M0,M0],
#         [M2,M3,M0,M0,M0]
#         ],dtype=qnn.Qnnum)
# seg_2=np.array([\
#         [M0,M0,M0,M1,M0],
#         [M0,M0,M4,M3,M0]
#         ],dtype=qnn.Qnnum)

#test_dir='../../tests/octa2/tests'
#xyz_dir='../../xyz/octa'
M0=qnn.Qnnum([0,0,1])  # 0
M1=qnn.Qnnum([1,0,2])  # 1/2
M2=qnn.Qnnum([-1,0,2]) #-1/2
oc_asym0=np.array([\
    [M0,M0,M0,M0,M0],\
    [M1,M2,M0,M1,M0],\
    [M1,M2,M2,M1,M0],\
    ],dtype=qnn.Qnnum)
oc_asym1=np.array([\
    [M0,M0,M0,M0,M0],\
    [M2,M1,M0,M2,M0],\
    [M2,M1,M2,M2,M0],\
    ],dtype=qnn.Qnnum)


od_asym0_nd=qna.anya(oc_asym0,(3,5))
od_asym1_nd=qna.anya(oc_asym1,(3,5))

triang_1=prj.projection3_sets_numerical(od_asym0_nd)  #.reshape((1,3,2))
triang_2=prj.projection3_sets_numerical(od_asym1_nd)  #.reshape((1,3,2))

#a=num.check_intersection_two_segment_numerical_nd_tau(segment_1,segment_2)
#print(a)
x=isct.intersection_two_triangles(triang_1, triang_2) 
qnv.printqnvs("cross points of two triangles",x)

print('TEST1')

# seg_1=np.array([\
#         [M1,M1,M0,M0,M0],\
#         [M3,M2,M0,M0,M0]
#         ],dtype=qnn.Qnnum)
# seg_2=np.array([\
#         [M0,M1,M0,M0,M0],\
#         [M0,M3,M4,M0,M0]
#         ],dtype=qnn.Qnnum)

# segment_1=qna.anya(seg_1,(2,5))
# segment_2=qna.anya(seg_2,(2,5))
# qna.printqndm("segment_1",segment_1)
# qna.printqndm("segment_2",segment_2)

#segment_1=np.array(\
#[[[1, 0, 1],[1, 0, 1],[0, 0, 1],[0, 0, 1],[0, 0, 1],[0, 0, 1]],\
#[[1, 0, 2],[3, 0, 2],[0, 0, 1],[0, 0, 1],[0, 0, 1],[0, 0, 1]]])
#segment_2=np.array(\
#[[[ 0,  0,  1],[ 1,  0,  1],[ 0,  0,  1],[ 0,  0,  1],[ 0,  0,  1],[ 0,  0,  1]],\
#[[ 0,  0,  1],[ 1,  0,  2],[-1,  0,  2],[ 0,  0,  1],[ 0,  0,  1],[ 0,  0,  1]]])


#a=num.check_intersection_two_segment_numerical_nd_tau(segment_1,segment_2)
#print(a)
# qna.printqndm("segment_1",segment_1)
# qna.printqndm("segment_2",segment_2)
# qnseg_1=prj.projection3_sets_numerical(segment_1)
# qnseg_2=prj.projection3_sets_numerical(segment_2)

# a=isct.intersection_two_triangles(qnseg_1, qnseg_2) 
# qnv.printqnv("a",a)

"""
s: [ 2 -1  1] 0.5857864376269049
tmp1 [[ 0  1  2]
 [ 4 -1  2]
 [ 0  0  1]
 [ 0  0  1]
 [ 0  0  1]
 [ 0  0  1]]
"""
