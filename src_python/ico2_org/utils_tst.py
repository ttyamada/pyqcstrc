#if __name__ == '__main__':

# test
import sys
#sys.path.append('.')
from ico2.projection import projection3
from ico2.math1 import (add,
                        sub,
                        mul,
                        div,
                        add_vectors,
                        sub_vectors,
                        outer_product,
                        inner_product,
                        centroid,
                        coplanar_check,
                        )
from ico2.numericalc import (numeric_value,
                                    numerical_vector,
                                    numerical_vectors,
                                    point_on_segment,
                                    coplanar_check_numeric_tau,
                                    get_internal_component_numerical,
                                    get_internal_component_sets_numerical,
                                    )
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
import itertools
import time

import random
DTYPE_int = int
#DTYPE_int = np.int64

def generate_random_value():
    """ generate value in TAU-style
    """
    nmax=10
    v=np.zeros((3),dtype=DTYPE_int)
    for i1 in range(2):
        v[i1]=random.randrange(-nmax,nmax) # a and b in (a+b*TAU)/c.
    v[2]=random.randrange(1,nmax) # c in (a+b*TAU)/c.
    return v
    
def generate_random_vector(ndim=6):
    """ generate ndim vector in TAU-style
    ndim: dimension of vectors
    """
    nmax=10
    v=np.zeros((ndim,3), dtype=DTYPE_int)
    for i1 in range(ndim):
        v[i1]=generate_random_value()
    return v
    
def generate_random_vectors(n,ndim=6):
    """
    num: number of generated vectors.
    ndim: dimension of vectors
    """
    v=np.zeros((n,ndim,3), dtype=DTYPE_int)
    for i1 in range(n):
        v[i1]=generate_random_vector(ndim)
    return v

def generate_random_tetrahedron():
    return generate_random_vectors(4)



#================
# ソートのテスト
#================
nset=10
vts=generate_random_vectors(nset)
vns=get_internal_component_sets_numerical(vts)
for vn in vns:
    print(vn)
print('\n')
vts1=sort_vctors(vts)
vns1=get_internal_component_sets_numerical(vts1)
for vn in vns1:
    print(vn)

#================
# 重複のテスト
#================
nset=10
vst=generate_random_vectors(nset)
vst_d3=np.concatenate([vst,vst]) # doubling dim3 vectors
vst_d4=np.stack([vst_d3,vst_d3]) # doubling dim4 vectors

a=remove_doubling(vst_d4)
if len(a)==nset:
    print('remove_doubling: pass')
else:
    print('remove_doubling: error')
    
a=remove_doubling_in_perp_space(vst_d4)
if len(a)==nset:
    print('remove_doubling_in_perp_space: pass')
else:
    print('remove_doubling_in_perp_space: error')

#================
# 面と辺のテスト
#================
tetrahedron=generate_random_tetrahedron()

# doubled tetrahedon
obj=np.stack([tetrahedron,tetrahedron]) # doubled tetrahedon
generator_surface_1(obj)

# a tetrahedon
obj=tetrahedron
surface=generator_surface_1(obj.reshape(1,4,6,3))
generator_edge(surface)

    