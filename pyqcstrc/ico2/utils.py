#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
sys.path.append('..')
#import numericalc import obj_volume_6d_numerical, inside_outside_obj
#import math1 import det_matrix, projection, projection3, add, sub, mul, div
from math1 import projection3
import numpy as np


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
        num,n2,_,_=vst.shape
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
    
    ndim=vst.ndim
    if ndim==4:
        n1,n2,_,_=vst.shape
        vst=vst.reshape(n1*n2,6,3)
    elif ndim==3:
        pass
    else:
        print('ndim should be larger than 3.')
    
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