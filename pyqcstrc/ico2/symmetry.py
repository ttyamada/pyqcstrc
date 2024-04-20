#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
sys.path.append('..')
from math1 import add, matrixpow, dot_product, sub_vectors, add_vectors
from utils import remove_doubling_dim4_in_perp_space, remove_doubling_dim3
import numpy as np


def symop_obj(symop,obj,centre):
    """ Apply a symmetric operation on an object around given centre. in TAU-style
    
    """
    ndim=obj.ndim
    if ndim==3:
        return symop_vecs(symop,obj,centre)
    elif ndim==4:
        obj1=np.zeros(obj.shape,dtype=np.int64)
        i=0
        for tetrahedron in obj:
            obj1[i]=symop_tetrahedron(symop,tetrahedron,centre)
            i+=1
        return obj1
    else:
        return 

def symop_vecs(symop,tetrahedron,centre):
    """ Apply a symmetric operation on set of vectors around given centre. in TAU-style
    
    """
    tetrahedron1=np.zeros(tetrahedron.shape,dtype=np.int64)
    i=0
    for vt in tetrahedron:
        obj1[i]=symop_vec(symop,vt,centre)
        i+=1
    return obj1

def symop_vec(symop,vt,centre):
    """ Apply a symmetric operation on a vector around given centre. in TAU-style
    
    """
    vt=sub_vectors(vt,centre)
    vt=dot_product(symop,vt)
    return add_vectors(vt,centre)





def generator_obj_symmetric_obj(obj,centre):
    mop=icosasymop()
    a=np.zeros((len(mop),obj.shape)),dtype=int.64)
    i=0
    for op in mop:
        a[i]=symop_obj(op,obj,centre)
    i+=1
    return a

def generator_obj_symmetric_surface(obj,centre):
    return generator_obj_symmetric_obj(obj,centre)
    
def np.ndarray generator_obj_symmetric_tetrahedron(obj,centre):
    return generator_obj_symmetric_obj(obj,centre)

def generator_obj_symmetric_tetrahedron_specific_symop(obj,centre,list_of_symmetry_operation_index):
    # using specific symmetry operations
    mop=icosasymop()
    a=np.zeros((len(index_of_symmetry_operation),obj.shape)),dtype=int.64)
    i=0
    for i in index_of_symmetry_operation:
        a[i]=symop_obj(mop[i],obj,centre)
    i+=1
    return a

def np.ndarray generator_obj_symmetric_tetrahedron_0(obj,centre,symmetry_operation_index):
    mop=icosasymop()
    return symop_obj(mop[symmetry_operation_index],obj,centre)

def np.ndarray generator_obj_symmetric_vec(vectors, centre):
    return generator_obj_symmetric_obj(vectors,centre)

def np.ndarray generator_equivalent_vectors(vectors,centre):
    a=generator_obj_symmetric_obj
    return remove_doubling_in_perp_space(a)

def np.ndarray generator_equivalent_vec(vector,centre):
    a=generator_obj_symmetric_obj(vector,centre)
    return remove_doubling(a)

def icosasymop():
    # icosahedral symmetry operations
    m1=np.array([[ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 1, 0, 0, 0, 0]])
    # mirror
    m2=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0,-1],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0,-1, 0, 0, 0]])
    # c2
    m3=np.array([[ 0, 0, 0, 0, 0,-1],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [-1, 0, 0, 0, 0, 0]])
    # c3
    m4=np.array([[ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0]])
    # inversion
    m5=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0,-1]])
    symop=[]
    for m in range(2):
        for l in range(3):
            for k in range(2):
                for j in range(2):
                    for i in range(5):
                        s1=matrixpow(m1,i) # c5
                        s2=matrixpow(m2,j) # mirror
                        s3=matrixpow(m3,k) # c2
                        s4=matrixpow(m4,l) # c3
                        s5=matrixpow(m5,m) # inversion
                        tmp=np.dot(s5,s4)
                        tmp=np.dot(tmp,s3)
                        tmp=np.dot(tmp,s2)
                        tmp=np.dot(tmp,s1)
                        symop.append(tmp)
    return symop

if __name__ == '__main__':
    
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
    
    def generate_random_tetrahedron():
        return generate_random_vectors(4)
    