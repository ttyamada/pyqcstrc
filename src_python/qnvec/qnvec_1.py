import sys
import numpy as np
from numpy.typing import NDArray
from typing import Self

import crsys as crs
import qnnum as qnn
import qnndarray as qna

class Qnvec(qna.QnNdarray):
#class Qnvec(np.ndarray):
    def __new__(cls, n:np.int64):
        global shape
        shape=(n)
        return super().__new__(cls,shape)
    
    def __init__(self, n:np.int64):
        qn0=qnn.zero() #int2qnn(0,N)
        self.n=n
        self.shape=shape
        for i in range(self.shape[0]):
            self[i]=qn0
        #print("self.shape",self.shape)  # for test
        #print("self.ndim",self.ndim)    # for test
        #print("self.dtype",self.dtype)  # for test

    def __add__(a:Self, b:Self):
        return add(a,b)
    
    def __sub__(a:Self, b:Self):
        return sub(a,b)
    
    def __iadd__(self:Self, b:Self):
        return iadd(self,b)
        
    def __isub__(self:Self, b:Self):
        return isub(self,b)
    
    def __mul__(a:Self, b:Self): # b should be int or qnnum
        if isinstance(b, int):
            return mul_vector_i(a,b)
        elif isinstance(b, qnn.Qnnum):
            return mul_vector_qn(a,b)
        
    def __truediv__(a:Self,b:np.int64): # b should be int
        if isinstance(b, int):
            return div_vector_i(a,b)
        
    def __eq__(a:Self, b:Self):
        return eq(a,b)
    
    def __not__(a:Self,b:Self):
        return not_eq(a,b)
    
    def __copy__(a:Self):
        return copy(a)
    
def qnvec_init():
    global n,N,isys,n_e,n_i
    isys=crs.isys
    n=crs.n
    N=crs.N
    if isys==2:
        n_e=3; n_i=3
    else:
        n_e=2; n_i=2
   
def zerovs(shape:np.int64) -> qna.QnNdarray:  # qnvec ndarray
    #print("shape in zerovs",shape)  # for test
    nv=shape[0]
    n=shape[1]
    qnvs=qna.zeros(shape)
    #qnvs = [Qnvec(n) for i in range(nv)] # list
    #print("type(qnvs)",type(qnvs))  # for test
    #print("type(qnvs[0])",type(qnvs[0]))  # for test
    for i in range(nv):
        qnvs[i]=zerov(n)
    return qnvs

def zerov(n: np.int64)->Qnvec: # qnnumber zero vector
    qnv=Qnvec(n)
    return qnv

def anyv(v:NDArray[qnn.Qnnum]) -> qna.QnNdarray:
    shape=v.shape
    #print("shape in anyv",shape)  # for test
    n=shape[0]
    v1=Qnvec(n)
    for i in range(n):
        v1[i]=v[i]
    #printqnv("v1",v1)  # for test
    return v1

