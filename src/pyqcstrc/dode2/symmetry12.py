#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np

from pyqcstrc.dode2.math12 import (add,
                                sub,
                                matrixpow,
                                )
from pyqcstrc.dode2.utils12 import (remove_doubling, 
                                    remove_doubling_in_perp_space,
                                    )

DTYPE_int = np.int64

def symop_obj(symop,obj,centre):
    obj1=[]
    for i in range(len(obj)):
        obj1.extend(symop_vec(symop,obj[i],centre))
    return obj1

def symop_vec(symop,vec,centre):
    vec1=shift_vec_pull(vec,centre)
    vec2=symop_vec_0(symop,vec1)
    vec1=shift_vec_push(vec2,centre)
    return vec1.tolist()

def shift_vec_push(vt,shift):
    a=[]
    for i1 in range(6):
        b=add(vt[i1],shift[i1])
        a.append(b)
    return np.array(a)

def shift_vec_pull(vec,shift):
    a=[]
    for i1 in range(6):
        b=sub(vec[i1][0],vec[i1][1],vec[i1][2],shift[i1][0],shift[i1][1],shift[i1][2])
        a.append(b)
    return np.array(a)

def symop_vec_0(symop,vec):
    """
    vector: from the origin to a point defined by vec
    symmetric operation on the vector. Note that the symmetric center is the origin.
    """
    vec1=[]
    for k in range(6):
        b1,b2,b3=0,0,1
        for j in range(6):
            a1=vec[j][0]*symop[k][j]
            a2=vec[j][1]*symop[k][j]
            a3=vec[j][2]
            [b1,b2,b3]=add(b1,b2,b3,a1,a2,a3)
        vec1.extend([b1,b2,b3])
    return np.array(vec1).reshape(6,3)

def generator_obj_symmetric_triangle(obj,centre):
    od=[]
    mop=dodesymop()
    for i in range(len(mop)):
        od.extend(symop_obj(mop[i],obj,centre))
    return np.array(od).reshape(int(len(od)/54),3,6,3) # 54=3*6*3

def generator_obj_symmetric_triangle_specific_symop(obj,centre,symmetry_operation):
    od=[]
    for i in range(len(symmetry_operation)):
        od.extend(symop_obj(symmetry_operation[i],obj,centre))
    return np.array(od).reshape(int(len(od)/54),3,6,3) # 54=3*6*3
    
def generator_symmetric_obj_0(obj,centre,numop):
    for i in range(len(obj)):
        tmp4a=generator_obj_symmetric_triangle_0(obj[i],centre,numop)
        if i==0:
            tmp4b=tmp4a
        else:
            tmp4b=np.vstack([tmp4b,tmp4a])
    return tmp4b

def generator_obj_symmetric_triangle_0(obj,centre,numop):
    od=[]
    mop=dodesymop()
    od.extend(symop_obj(mop[numop],obj,centre))
    return np.array(od).reshape(1,3,6,3) # 54=3*6*3

def generator_obj_symmetric_vec(vts,centre):
    od=[]
    mop=dodesymop()
    for i1 in range(len(mop)):
        for i2 in range(len(vts)):
            od.extend(symop_vec(mop[i1],vectors[i2],centre))
    return np.array(od).reshape(24,int(len(od)/18/24),6,3) # 18=6*3

def generator_obj_symmetric_vec_0(vt, numop):
    centre=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,0,0,0)
    mop=dodesymop()
    return symop_vec(mop[numop],vt,centre)
    
def similarity_obj(obj,m):
    obj1=np.zeros((obj1.shape),dtype=DTYPE_int)
    for i in range(len(obj)):
        obj1=similarity_vec(obj[i],m)
    return obj1

def similarity_vec(vt, m):
    op=similarity(m)
    vec1=[]
    for k in range(6):
        b1,b2,b3=0,0,1
        for j in range(6):
            a1=vt[j][0]*op[k][j]
            a2=vt[j][1]*op[k][j]
            a3=vt[j][2]
            [b1,b2,b3]=add(b1,b2,b3,a1,a2,a3)
        vec1.extend([b1,b2,b3])
    return np.array(vec1)

def generator_equivalent_vec(vt,centre):
    mop=dodesymop()
    vts=np.zeros((len(mop),6,3),dtype=DTYPE_int)
    for i1 in range(len(mop)):
        vts[i1]=(symop_vec(mop[i1],vector,centre))
    return remove_doubling(vts)
    
def similarity(m):
    # Similarity transformation of Dodecagonal QC
    if m>0:
        m1=np.array([[ 1, 0, 0, -1, 0, 0],\
                    [ 1, 1, 0, 0, 0, 0],\
                    [ 0, 1, 1, 1, 0, 0],\
                    [ 0, 0, 1, 1, 0, 0],\
                    [ 0, 0, 0, 0, 1, 0],\
                    [ 0, 0, 0, 0, 0, 1]])
        return matrix_pow(m1.T,m)
    elif m<0:
        m1=np.array([[ 0, 1,-1, 1, 0, 0],\
                   [ 0, 0, 1,-1, 0, 0],\
                   [ 1,-1, 1, 0, 0, 0],\
                   [-1, 1,-1, 1, 0, 0],\
                   [ 0, 0, 0, 0, 1, 0],\
                   [ 0, 0, 0, 0, 0, 1]])
        return matrix_pow(m1.T,-m)
    else:
        m1=np.array([[ 1, 0, 0, 0, 0, 0],\
                   [ 0, 1, 0, 0, 0, 0],\
                   [ 0, 0, 1, 0, 0, 0],\
                   [ 0, 0, 0, 1, 0, 0],\
                   [ 0, 0, 0, 0, 1, 0],\
                   [ 0, 0, 0, 0, 0, 1]])
        return m1.T

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
            s1=matrix_pow(m1,i2) # c12
            s2=matrix_pow(m2,i1) # mirror
            tmp=matrix_dot(s2,s1)
            symop.append(tmp)
    return symop




def list site_symmetry(site):
    
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


