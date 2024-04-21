#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
sys.path.append('.')
from math1 import (projection3,
                    sub_vectors,
                    add_vectors,
                    outer_product,
                    inner_product,
                    add,
                    sub,
                    mul,
                    div)
from numericalc import numeric_value
import numpy as np




def shift_object(obj,shift):
    """shift an object
    """
    vol0=obj_volume_6d(obj)
    obj_new=np.zeros(obj.shape,dtype=np.int64)
    i1=0
    for tetrahedron in obj:
        i2=0
        for vertex in tetrahedron:
            obj_new[i1][i2]=add_vectors(vertex,shift)
            i2+=1
        i1+=1
    vol1=obj_volume_6d(obj_new)
    if np.all(vol0==vol1):
        return obj_new
    else:
        return 


def obj_volume_6d(obj):
    w=np.array([0,0,1])
    if obj.ndim==4:
        for tetrahedron in obj:
            v=tetrahedron_volume_6d(tetrahedron)
            w=add(w,v)
        return w
    elif obj.ndim==5:
        for tset in obj:
            for tetrahedron in tset:
                v=tetrahedron_volume_6d(tetrahedron)
                w=add(w,v)
        return w
    else:
        print('object has an incorrect shape!')
        return 
    
def tetrahedron_volume_6d(tetrahedron):
    vts=np.zeros((4,3,3),dtype=np.int64)
    for i in range(4):
        vts[i]=projection3(tetrahedron[i])
    return tetrahedron_volume(vts)

def tetrahedron_volume(vts):
    # This function returns volume of a tetrahedron
    # input: vertex coordinates of the tetrahedron (x0,y0,z0),(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)
    v1=sub_vectors(vts[1],vts[0])
    v2=sub_vectors(vts[2],vts[0])
    v3=sub_vectors(vts[3],vts[0])
    
    v=outer_product(v1,v2)
    v=inner_product(v,v3)
    #[a1,a2,a3]=det_matrix(a,b,c) # determinant of 3x3 matrix
    
    # avoid a negative value
    val=numeric_value(v)
    if val<0.0: # to avoid negative volume
        return mul(v,np.array([-1,0,6]))
    else:
        return mul(v,np.array([1,0,6]))




# こちら2つに統一した方が良い。
def remove_doubling(vst):
    """remove doubling 6d coordinates
    
    Parameters
    ----------
    obj: array
        set of 6-dimensional vectors in TAU-style
    
    Returns
    -------
    obj: array
        set of 6-dimensional vectors in TAU-style
    """
    
    ndim=vst.ndim
    if ndim==4:
        n1,n2,_,_=vst.shape
        num=n1*n2
        vst=vst.reshape(num,6,3)
    elif ndim==3:
        num,_,_=vst.shape
    else:
        print('ndim should be larger than 3.')
    
    if num>1:
        lst=[0]
        for i1 in range(1,num):
            counter=0
            vst1=vst[i1]
            for i2 in lst:
                if np.all(vst1==vst[i2]):
                    counter+=1
                    break
                else:
                    pass
            if counter==0:
                lst.append(i1)
            else:
                pass
        num=len(lst)
        vst_new=np.zeros((num,6,3),dtype=np.int64)
        for i in range(num):
            vst_new[i]=vst[lst[i]]
        return vst_new
    else:
        return vst

def remove_doubling_in_perp_space(vst):
    """ remove 6d coordinates which is doubled in Eperp.
    
    Parameters
    ----------
    obj: array
        set of 6-dimensional vectors in TAU-style
    
    Returns
    -------
    obj: array
        set of 6-dimensional vectors in TAU-style
    """
    #print('remove_doubling')
    # 先にremove_doubling()を走らすと速くなる。
    vst=remove_doubling(vst)
    
    #print('remove_doubling_in_perp_space')
    ndim=vst.ndim
    if ndim==4:
        n1,n2,_,_=vst.shape
        vst=vst.reshape(n1*n2,6,3)
    elif ndim==3:
        num,_,_=vst.shape
    
    if num>1:
        lst=[0]
        for i1 in range(1,num):
            xyzi1=projection3(vst[i1])
            counter=0
            for i2 in lst:
                if np.all(xyzi1==projection3(vst[i2])):
                    counter+=1
                    break
                else:
                    pass
            if counter==0:
                lst.append(i1)
            else:
                pass
        num=len(lst)
        vst_new=np.zeros((num,6,3),dtype=np.int64)
        for i in range(num):
            vst_new[i]=vst[lst[i]]
        return vst_new
    else:
        return vst






# 下の関数を使用しているが、上の2つに統一した方が良い。

def remove_doubling_dim4(vst):
    """remove doubling 6d coordinates
    
    Parameters
    ----------
    obj: array
        set of 6-dimensional vectors in TAU-style
    
    Returns
    -------
    obj: array
        set of 6-dimensional vectors in TAU-style
    """
    n1,n2,_,_=vst.shape()
    a=vst.reshape(n1*n2,6,3)
    return remove_doubling_dim3(a)

def remove_doubling_dim3(vst):
    """remove doubling 6d coordinates
    
    Parameters
    ----------
    obj: array
        set of 6-dimensional vectors in TAU-style
    
    Returns
    -------
    obj: array
        set of 6-dimensional vectors in TAU-style
    """
    num=len(vst)
    if num>1:
        lst=[0]
        for i1 in range(1,num):
            counter=0
            vst1=vst[i1]
            for i2 in lst:
                if np.all(vst1==vst[i2]):
                    counter+=1
                    break
                else:
                    pass
            if counter==0:
                lst.append(i1)
            else:
                pass
        num=len(lst)
        vst_new=np.zeros((num,6,3),dtype=np.int64)
        for i in range(num):
            vst_new[i]=vst[lst[i]]
        return vst_new
    else:
        return vst

def remove_doubling_dim4_in_perp_space(vst):
    """ remove 6d coordinates which is doubled in perpendicular space
    
    Parameters
    ----------
    obj: array
        set of 6-dimensional vectors in TAU-style
    
    Returns
    -------
    obj: array
        set of 6-dimensional vectors in TAU-style
    """
    n1,n2,_,_=vst.shape
    a=vst.reshape(n1*n2,6,3)
    return remove_doubling_dim3_in_perp_space(a)

def remove_doubling_dim3_in_perp_space(vst):
    """ remove 6d coordinates which is doubled in Eperp.
    
    Parameters
    ----------
    obj: array
        set of 6-dimensional vectors in TAU-style
    
    Returns
    -------
    obj: array
        set of 6-dimensional vectors in TAU-style
    """
    num=len(vst)
    if num>1:
        lst=[0]
        for i1 in range(1,num):
            xyzi1=projection3(vst[i1])
            counter=0
            for i2 in lst:
                #xyzi2=projection3(vst[i2])
                if np.all(xyzi1==projection3(vst[i2])):
                    counter+=1
                    break
                else:
                    pass
            if counter==0:
                lst.append(i1)
            else:
                pass
        num=len(lst)
        vst_new=np.zeros((num,6,3),dtype=np.int64)
        for i in range(num):
            vst_new[i]=vst[lst[i]]
        return vst_new
    else:
        return vst


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
    
    #================
    # 重複のテスト
    #================
    nset=5
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
    
    