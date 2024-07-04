#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
sys.path.append('.')
from math1 import (add, 
                                matrixpow, 
                                dot_product, 
                                dot_product_1, 
                                sub_vectors, 
                                add_vectors,
                                )
from dode2.utils import (remove_doubling_in_perp_space, 
                                remove_doubling,
                                )
"""
from pyqcstrc.dode2.math1 import (add, 
                                matrixpow, 
                                dot_product, 
                                dot_product_1, 
                                sub_vectors, 
                                add_vectors,
                                )
from pyqcstrc.dode2.utils import (remove_doubling_in_perp_space, 
                                remove_doubling,
                                )
"""
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
    elif ndim==2:
        return symop_vec(symop,obj,centre)
    else:
        print('object has an incorrect shape!')
        return 

def symop_vecs(symop,triangle,centre):
    """ Apply a symmetric operation on set of vectors around given centre. in TAU-style
    
    """
    triangle1=np.zeros(triangle.shape,dtype=np.int64)
    i=0
    for vt in triangle:
        triangle1[i]=symop_vec(symop,vt,centre)
        i+=1
    return triangle1

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

def generator_obj_symmetric_triangle(obj,centre):
    return generator_obj_symmetric_obj(obj,centre)
    
#def generator_obj_symmetric_tetrahedron(obj,centre):
#    return generator_obj_symmetric_obj(obj,centre)

def generator_obj_symmetric_triangle_specific_symop(obj,centre,list_of_symmetry_operation_index):
    # using specific symmetry operations
    if obj.ndim==3 or obj.ndim==4:
        mop=dodesymop()
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
    
def generator_obj_symmetric_triangle_0(obj,centre,symmetry_operation_index):
    mop=dodesymop()
    return symop_obj(mop[symmetry_operation_index],obj,centre)

def generator_obj_symmetric_vec(vectors, centre):
    return generator_obj_symmetric_obj(vectors,centre)

def generator_equivalent_vectors(vectors,centre):
    a=generator_obj_symmetric_obj
    return remove_doubling_in_perp_space(a)

def generator_equivalent_vec(vector,centre):
    a=generator_obj_symmetric_obj(vector,centre)
    return remove_doubling(a)

def dodesymop():
    # dodecagonal symmetry operations
    # c12
    m1=np.array([[ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [-1, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    # mirror
    m2=np.array([[ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 1, 0, 0, 0, 0],\
                [ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    symop=[]
    for i1 in range(2):
        for i2 in range(12):
            s1=matrixpow(m1,i2) # c12
            s2=matrixpow(m2,i1) # mirror
            tmp=np.dot(s2,s1)
            symop.append(tmp)
    return symop

def site_symmetry(site):
    
    vec1=[]
    centre=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    dev=np.array([[-3,4,1],[2,-4,1],[-4,3,1],[3,1,1],[0,0,1],[0,0,1]])
    
    # translation
    v1=np.array([[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (1,0,0,0)
    trans=generator_equivalent_vec(v1,centre)
    
    for k in range(6):
        [b1,b2,b3]=add(site[k][0],site[k][1],site[k][2],dev[k][0],dev[k][1],dev[k][2])
        vec1.extend([b1,b2,b3])
    vec=np.array(vec1).reshape(6,3)
    tmp3a=generator_equivalent_vec(vec,centre)
    
    # translational symmetry
    vec1=[]
    for i in range(len(tmp3a)):
        for j in range(len(trans)):
            for k in range(6):
                [b1,b2,b3]=sub(tmp3a[i][k][0],tmp3a[i][k][1],tmp3a[i][k][2],trans[j][k][0],trans[j][k][1],trans[j][k][2])
                vec1.extend([b1,b2,b3])
    tmp3b=np.array(vec1).reshape(int(len(vec1)/18),6,3)
    tmp3a=np.vstack([tmp3a,tmp3b])
    
    #print('1: len(tmp3a)=',len(tmp3a))
    vec1=[]
    for i in range(len(tmp3a)):
        for k in range(6):
            [b1,b2,b3]=sub(tmp3a[i][k][0],tmp3a[i][k][1],tmp3a[i][k][2],site[k][0],site[k][1],site[k][2])
            vec1.extend([b1,b2,b3])
    tmp3a=np.array(vec1).reshape(len(tmp3a),6,3)
    #print(vec1)
    
    #print('2: len(tmp3a)=',len(tmp3a))
    mop=dodesymop()
    
    numlst=[]
    for i in range(len(tmp3a)):
        for k in range(len(mop)):
            vec1=symop_vec(mop[k],dev,centre)
            if np.array_equal(np.array(vec1).reshape(6,3),tmp3a[i]):
                numlst.extend([k])
            else:
                pass
    return numlst

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
    