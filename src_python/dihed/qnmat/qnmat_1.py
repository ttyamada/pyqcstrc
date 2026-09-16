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

# this includes nD Qnvec as a special case (shape=(n))
#class Qnmat(np.ndarray):
class Qnmat(qna.QnNdarray):
    def __new__(cls, shape:np.int64):
        return super().__new__(cls,shape)
        #return super().__new__(cls,shape,dtype=qnn.Qnnum)

    def __init__(self,shape:np.int64):
        #global n,N
        qn0=qnn.zero()
        self.N=N
        self.shape=shape
        ndim=len(shape)
        #print("ndim in Qnmat",ndim)  # for test
        if ndim==1:
            for i in range(shape[0]):
                self[i]=qn0
        
        if ndim==2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    self[i][j]=qn0
        #print("self.shape",self.shape)
        #print("self.ndim",self.ndim)
        #print("self.dtype",self.dtype)
        #printqnm("Qnmat self",self) # for test

    def __add__(ma1:Self, ma2:Self):  #  for ma1+ma2
        return add(ma1,ma2)
    
    def __iadd__(ma1:Self, ma2:Self):  #  for ma1+=ma2
        return iadd(ma1,ma2)
    
    def __isub__(ma1:Self, ma2:Self):  #  for ma1-=ma2
        return isub(ma1,ma2)
        
    def __matmul__(ma1:Self, ma2:Self):  #  for ma1@ma2
        return mul(ma1,ma2)
        #if isinstance(ma2,qna.QnNdarray):
        #    return mul(ma1,ma2)
        #if isinstance(ma2,NDArray[np.int64]):
        #    return mul_i(ma1,ma2)
    
    def set_mt(self:Self,mt:np.array): # QnNdarray*
        shape=self.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.mt[i][j]=qnn.copy(mt[i][j]) # copy qnnum
                
    def __copy__(a:Self):
        return copy(a)
    
def qnmat_init():
    global n,N,scly,isys
    n=crs.n
    N=crs.N
    isys=crs.isys
    if isys==3:
        scly=2.0*np.sin(np.pi/5)
    else:
        scly=1.0

def zerom(shape:np.int64) -> Qnmat:
    qnm=Qnmat(shape)
    return qnm

def zeroms(shape:np.int64) -> Qnmat:
    nm=shape[0]
    n=shape[1]
    #qnms=[Qnmat((shape[1],shape[2])) for i in range(n)]
    qnms=qna.zeros((nm),dtype=qnn.Qnmat)
    for i in range(nm):
        qnvs[i]=zerom((shape[1],shape[2]))
    return qnvs
    return qnms

