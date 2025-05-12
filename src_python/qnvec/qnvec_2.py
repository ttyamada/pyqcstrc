import sys
import numpy as np
from numpy.typing import NDArray
from typing import Self

import crsys as crs
import qnnum as qnn
import qnndarray as qna

def copy(v: Qnvec) -> Qnvec:
    shape=v.shape
    v1=Qnvec(shape)
    for i in range(shape[0]):
        v1[i]=v[i]
    return v1
    #return np.deepcopy(v1)
    
def add(v1:Qnvec, v2:Qnvec) -> Qnvec:
    n=v1.shape[0]
    a=Qnvec(n)
    for i in range(n):
        a[i]=v1[i]+v2[i]
    return a

def iadd(self:Qnvec, b:Qnvec):
    self=add(self,b)
    return self

def sub(v1:Qnvec, v2:Qnvec)-> Qnvec:
    a=Qnvec(n)
    for i in range(n):
        a[i]=v1[i]-v2[i]
    return a

def isub(self:Qnvec, b:Qnvec):
    self=sub(self,b)
    return self

def sub_vectors_qn(v:qna.QnNdarray, v0:Qnvec):
    shape=v.shape
    ndim=len(shape)
    v1=qna.zeros(shape)
    if ndim==2:
        for i in range(shape[0]):
            v1[i]=v[i]-v0
    elif ndim==3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                v1[i][j]=v[i][j]-v0
    return v1

def mul_vector_i(v:Qnvec, coeff) -> Qnvec:
    if v.ndim==1:
        a=Qnvec(n)  #np.zeros(v.shape,dtype=np.int64)
        for i in range(n):
            a[i]=v[i]*coeff  #mul(v,coeff)
        return a
    else:
        print('incorrect shape in mul_vector_i')
        return

def mul_vector_qn(v:Qnvec, coeff:qnn.Qnnum) -> Qnvec:
    if v.ndim==1:
        a=Qnvec(n)  #np.zeros(v.shape,dtype=np.int64)
        for i in range(n):
            a[i]=v[i]*coeff  #mul(v,coeff)
        return a
    else:
        print('incorrect shape in mul_vector_qn')
        return

def mul_vectors_i(vs:qna.QnNdarray, coeff:int) -> qna.QnNdarray:
    shape=vs.shape
    ndim=len(shape)
    if vs.ndim==2:
        a=qna.zeros(shape)
        la=vs.shape
        for i in range(la[0]):
            for j in range(n):
                a[i][j]=vs[i][j]*coeff #mul_vector(v,coeff)
        return a
    else:
        print('incorrect shape in mul_vectors_i')
        return

def mul_vectors_qn(vs:Qnvec, coeff:qnn.Qnnum):
    if vs.ndim==2:
        #n=vs.shape[0]
        #N=vs[0].N
        a=[Qnvec(n)]*vs.shape
        la=vs.shape
        for i in range(la[0]):
            for j in range(n):
                a[i][j]=vs[i][j]*coeff #mul_vector(v,coeff)
        return a
    else:
        print('incorrect shape mul_vectors_qn')
        return

def div_vector_i(v:Qnvec, coeff: int) -> Qnvec:
    if v.ndim==1:
        a=np.zeros(v.shape,dtype=np.int64)
        la=v.shape
        for i in range(la[0]):
            a[i]=v[i]/coeff  #mul(v,coeff)
        return a
    else:
        print('incorrect shape in div_vector_i')
        return

    
    
    
