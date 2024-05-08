#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
#sys.path.append('.')
from pyqcstrc.ico2.math1 import (add, 
                                matrixpow, 
                                dot_product, 
                                dot_product_1, 
                                sub_vectors, 
                                add_vectors,
                                )
from pyqcstrc.ico2.utils import (remove_doubling_in_perp_space, 
                                remove_doubling,
                                )
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
        for vts in obj:
            obj1[i]=symop_vecs(symop,vts,centre)
            i+=1
        return obj1
    else:
        print('object has an incorrect shape!')
        return 

def symop_vecs(symop,tetrahedron,centre):
    """ Apply a symmetric operation on set of vectors around given centre. in TAU-style
    
    """
    tetrahedron1=np.zeros(tetrahedron.shape,dtype=np.int64)
    i=0
    for vt in tetrahedron:
        tetrahedron1[i]=symop_vec(symop,vt,centre)
        i+=1
    return tetrahedron1

def symop_vec(symop,vt,centre):
    """ Apply a symmetric operation on a vector around given centre. in TAU-style
    
    """
    vt=sub_vectors(vt,centre)
    vt=dot_product_1(symop,vt)
    return add_vectors(vt,centre)





def generator_obj_symmetric_obj(obj,centre):
    
    if obj.ndim==3 or obj.ndim==4:
        mop=icosasymop()
        num=len(mop)
        shape=tuple([num])
        a=np.zeros(shape+obj.shape,dtype=np.int64)
        i=0
        for op in mop:
            a[i]=symop_obj(op,obj,centre)
            i+=1
        if obj.ndim==4:
            n1,n2,_,_=obj.shape
            a=a.reshape(num*n1,n2,6,3)
        else:
            pass
        return a
    else:
        print('object has an incorrect shape!')
        return

def generator_obj_symmetric_surface(obj,centre):
    return generator_obj_symmetric_obj(obj,centre)
    
def generator_obj_symmetric_tetrahedron(obj,centre):
    return generator_obj_symmetric_obj(obj,centre)

def generator_obj_symmetric_tetrahedron_specific_symop(obj,centre,list_of_symmetry_operation_index):
    # using specific symmetry operations
    if obj.ndim==3 or obj.ndim==4:
        mop=icosasymop()
        shape=tuple([len(index_of_symmetry_operation)])
        a=np.zeros(shape+obj.shape,dtype=np.int64)
        i=0
        for i in index_of_symmetry_operation:
            a[i]=symop_obj(mop[i],obj,centre)
            i+=1
        return a
    else:
        print('object has an incorrect shape!')
        return
    
def generator_obj_symmetric_tetrahedron_0(obj,centre,symmetry_operation_index):
    mop=icosasymop()
    return symop_obj(mop[symmetry_operation_index],obj,centre)

def generator_obj_symmetric_vec(vectors, centre):
    return generator_obj_symmetric_obj(vectors,centre)

def generator_equivalent_vectors(vectors,centre):
    a=generator_obj_symmetric_obj
    return remove_doubling_in_perp_space(a)

def generator_equivalent_vec(vector,centre):
    a=generator_obj_symmetric_obj(vector,centre)
    return remove_doubling(a)

def icosasymop():
    # icosahedral symmetry operations
    m1=np.array([[ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 1, 0, 0, 0, 0]],dtype=np.int64)
    # mirror
    m2=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0,-1],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0,-1, 0, 0, 0]],dtype=np.int64)
    # c2
    m3=np.array([[ 0, 0, 0, 0, 0,-1],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [-1, 0, 0, 0, 0, 0]],dtype=np.int64)
    # c3
    m4=np.array([[ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0]],dtype=np.int64)
    # inversion
    m5=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0,-1]],dtype=np.int64)
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
    