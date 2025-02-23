
#if __name__ == '__main__':

# test

import numpy as np
import math1
import random
from utils import (get_internal_component_sets_numerical,
                   generator_all_edges,
                   remove_doubling_in_perp_space,
                   remove_doubling,
                   sort_vctors,
                   )

DTYPE_int = int
#DTYPE_int = np.int64

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



#================
# ソートのテスト
#================
nset=10
vts=generate_random_vectors(nset)
vns=get_internal_component_sets_numerical(vts)
print(vns.shape)
for vn in vns:
    print(vn)
print('\n')
print(vts.shape)
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
triangle=generate_random_triangle()

# doubled tetrahedon
obj=np.stack([triangle,triangle]) # doubled tetrahedon
#generator_surface_1(obj)

# a tetrahedon
obj=triangle
#surface=generator_surface_1(obj.reshape(1,3,6,3))
surface=obj.reshape(1,3,6,3)
#generator_edge(surface)
generator_all_edges(surface)
    