import sys
import numpy as np
import cython
from typing import Self
from numpy.typing import NDArray

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnndarray as qna
from numpy.typing import(NDArray)
#def zeroms(shape:np.int64,N:np.int64) -> Qnmat: # qnmat ndarray
#    return np.zeros(shape,dtype=Qnmat)

def anym(m:NDArray[qnn.Qnnum]) -> Qnmat:
    shape=m.shape
    m1=Qnmat(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            m1[i][j]=m[i][j]
    return m1

def unitm(n_:np.int64) -> Qnmat:
    qn1=qnn.one()
    shape=(n_,n_)
    qnm=zerom(shape)
    for i in range(n_):
        qnm[i][i]=qn1
    return qnm

def matrix_2d(v1:qnv.Qnvec, v2: qnv.Qnvec) -> Qnmat:
    m=zerom((2,2))
    for i in range(2):
        m[0][i]=v1[i]
        m[1][i]=v2[i]
    return m

def matrix_3d(v1:qnv.Qnvec, v2:qnv.Qnvec, v3:qnv.Qnvec) -> Qnmat:
    m=zerom((3,3))
    for i in range(3):
        m[0][i]=v1[i]
        m[1][i]=v2[i]
        m[2][i]=v3[i]
    return m
        
def copy(m: Qnmat) -> Qnmat:
    shape=m.shape
    m1=Qnmat(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            m1[i][j]=m[i][j]
    return m1

def copyms(ms: Qnmat) -> Qnmat:
    shape=ms.shape
    print("ms.shape",ms.shape)  # for test
    m1s=Qnmat(shape)
    for i in range(shape[0]):
        m1s[i]=ms[i]
    return m1s
                
def int2qnm(r:np.ndarray, n_:np.int64) -> Qnmat:
    qnr=Qnmat(n_) # n_ x n_ matrix
    for i in range(n_):
        for j in range(n_):
            qnr[i][j]=qnn.int2qnn(r[i][j])
    return qnr

def qnm2qnv(qnm: Qnmat) -> qnv.Qnvec:
    if len(qnm.shape)!=1:
        print("size(shape) != 1 so cannt convert to Qnvec")
    n=qnm.shape[0]
    #print("n",n,"n",n)  # for test
    qnvt=qnv.zerov((n))
    for i in range(n):
        qnvt[i]=qnm[i]
    return qnvt
    
def add(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    #a=np.empty(mat1.shape, dtype=qnn.Qnnum)
    shape=ma1.shape
    n1=len(ma1.shape)
    n2=len(ma2.shape)
    a=Qnmat(shape)
    if(n1==1 and n2==1): # vectors
        for i in range(shape[0]):
            a[i]=ma1[i]+ma2[i]  #add(v1[i],v2[i])
        return a
    elif(n1==2 and n2==2): # matrices
        for i in range(shape[0]):
            for j in range(shape[1]):
                a[i][j]=ma1[i][j]+ma2[i][j]  #add(v1[i],v2[i])
        return a 
