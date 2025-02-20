    
#if __name__ == '__main__':

# test
import sys
import numpy as np
try:
    from ico2.symmetry import (icosasymop,
                                        icosasymop_array,
                                        icosasymop3_array,
                                        icosasymop_array_1,
                                        icosasymop3_array_1,
                                        remove_overlaps,
                                        find_overlaps,
                                        similarity,
                                        )
except ImportError:
    print('import error in symmetry_numerical.py\n')
    
import random
from numericalc import (numerical_vectors,
                        numerical_vector,
                        numeric_value,)
                        
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

cen0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])




"""
print("TEST: symop_vec()")
symop=icosasymop()
vt=generate_random_vector()
counter=0
for sop in symop:
    #
    # calc using symop_vec
    svt=symop_vec(sop,vt,cen0)
    svn1=numerical_vector(svt)
    #print(svn1)
    #
    # calc using no.dot with float values
    vn=numerical_vector(vt)
    svn2=np.dot(sop,vn)
    #print(svn2)
    if np.allclose(svn1,svn2):
        pass
    else:
        counter+=1
if counter==0:
    print('symop_vec: correct')
else:
    print('symop_vec: worng')
    
nset=4
vts=generate_random_vectors(nset)
print(vts)
svts=symop_vecs(symop[1],vts,cen0)
print(svts)
"""


############################
# TEST symmetry operations
############################
print('')
print('Symmetry operation on 6D axial vector\n')
print("TEST :generator_obj_symmetric_vector_specific_symop_1()")
flag='axial'
lst=[0,108,22,92,119,16,14,111,98,28,102,5]
vn=np.array([1,0,0,0,0,0])
vns=generator_obj_symmetric_vector_specific_symop_1(vn,V0,lst,flag)
for vn in vns:
    print(vn)
"""
print("TEST :generator_obj_symmetric_vectors_specific_symop_1()")
print('Identity')
lst=[0]
vnss=generator_obj_symmetric_vectors_specific_symop_1(vns,V0,lst,flag)
for vns in vnss:
    for vn in vns:
        print(vn)
print('Inversion')
lst=[60]
vnss=generator_obj_symmetric_vectors_specific_symop_1(vns,V0,lst,flag)
for vns in vnss:
    for vn in vns:
        print(vn)
"""
    
print('')
print('Symmetry operation on 3D axial vector\n')
V0=np.array([0.,0.,0.],dtype=np.float64)
print("TEST :generator_obj_symmetric_vector_specific_symop_1()")
flag='axial'
lst=[0,108,22,92,119,16,14,111,98,28,102,5]
vn=np.array([1.,1.618034,0.])
vns=generator_obj_symmetric_vector_specific_symop_1(vn,V0,lst,flag)
for i1,vn in enumerate(vns):
    print('%d %8.6f %8.6f %8.6f'%(i1,vn[0],vn[1],vn[2]))
"""
print("TEST :generator_obj_symmetric_vectors_specific_symop_1()")
lst=[0,60] # identity and inversion
vnss=generator_obj_symmetric_vectors_specific_symop_1(vns,V0,lst,flag)
for i1,vns in enumerate(vnss):
    for i2,vn in enumerate(vns):
        print('%d %d %8.6f %8.6f %8.6f'%(i1,i2,vn[0],vn[1],vn[2]))
"""
    