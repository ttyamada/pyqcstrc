#if __name__ == '__main__':
    
# test
import sys
import itertools
import cython
import numpy as np
import random

from numpy.typing import NDArray

#EPS=1e-6
#V0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)

from numericalc import (numerical_vectors,numerical_vector,numeric_value)
from symmetry import (symop_vec,symop_vecs,octasymop_array)
                            
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


symop = octasymop_array()


