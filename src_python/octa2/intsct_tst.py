import numpy as np
from numpy.typing import NDArray
import time # in object_subtraction_dev1, tetrahedron_not_obj
import itertools
import cython
from intsct import (check_intersection_two_segment_numerical_6d_tau,
                    intersection_two_segment,
                    )

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

segment_1=np.array(\
[[[1, 0, 1],\
  [1, 0, 1],\
  [0, 0, 1],\
  [0, 0, 1],\
  [0, 0, 1],\
  [0, 0, 1]],\
 [[3, 0, 2],\
  [1, 0, 2],\
  [0, 0, 1],\
  [0, 0, 1],\
  [0, 0, 1],\
  [0, 0, 1]]])
  
segment_2=np.array(\
[[[ 0,  0,  1],\
  [ 0,  0,  1],\
  [ 0,  0,  1],\
  [ 1,  0,  1],\
  [ 0,  0,  1],\
  [ 0,  0,  1]],\
 [[ 0,  0,  1],\
  [ 0,  0,  1],\
  [-1,  0,  2],\
  [ 1,  0,  2],\
  [ 0,  0,  1],\
  [ 0,  0,  1]]])

a=check_intersection_two_segment_numerical_6d_tau(segment_1,segment_2)
print(a)
a=intersection_two_segment(segment_1, segment_2) 
print(a)



print('TEST1')
segment_1=np.array(\
[[[1, 0, 1],\
  [1, 0, 1],\
  [0, 0, 1],\
  [0, 0, 1],\
  [0, 0, 1],\
  [0, 0, 1]],\
 [[1, 0, 2],\
  [3, 0, 2],\
  [0, 0, 1],\
  [0, 0, 1],\
  [0, 0, 1],\
  [0, 0, 1]]])
segment_2=np.array(\
[[[ 0,  0,  1],\
  [ 1,  0,  1],\
  [ 0,  0,  1],\
  [ 0,  0,  1],\
  [ 0,  0,  1],\
  [ 0,  0,  1]],\
 [[ 0,  0,  1],\
  [ 1,  0,  2],\
  [-1,  0,  2],\
  [ 0,  0,  1],\
  [ 0,  0,  1],\
  [ 0,  0,  1]]])
a=check_intersection_two_segment_numerical_6d_tau(segment_1,segment_2)
print(a)
a=intersection_two_segment(segment_1, segment_2) 
print(a)

"""
s: [ 2 -1  1] 0.5857864376269049
tmp1 [[ 0  1  2]
 [ 4 -1  2]
 [ 0  0  1]
 [ 0  0  1]
 [ 0  0  1]
 [ 0  0  1]]
"""
