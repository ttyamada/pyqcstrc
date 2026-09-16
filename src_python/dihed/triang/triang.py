import numpy as np
from numpy.typing import NDArray
from typing import Self

import crsys
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnndarray as qna

class Qntri(qna.QnNdarray):
    def __new__(cls):
        global shape
        shape=(3,n_i)
        return super().__new__(cls,shape)
    
    def __init__(self):
        qn0=qnn.zero() #int2qnn(0,N)
        self.n=n_i
        self.N=N
        self.shape=shape
        for i in range(3):
            for j in range(n_i):
                self[i][j]=qn0

    def __eq__(a:Self, b:Self):
        return eq(a,b)
    
    def __not__(a:Self,b:Self):
        return not_eq(a,b)
      
def qntri_init():
    global n,N,isys,n_e,n_i
    isys=crsys.isys
    n=crsys.n
    N=crsys.N
    if isys==2:
        n_e=3; n_i=3
    else:
        n_e=2; n_i=2
        
def zerotri(n_i:np.int64):
    qn0=qnn.zero()
    v1=Triangle(n_i)
    for i in range(3):
        for j in range(n_i):
            v1[i][j]=qn0
    return v1
        
def zerotris(shape:np.int64) -> Qntri:  # qntri ndarray
    print("shape in zerovs",shape)  # for test
    nv=shape[0]
    n=shape[1]
    #qntrs = [Qntr(n_i) for i in range(nv)] # list
    qntrs=qna.zeros((nv),dtype=Qntri)
    #print("type(qnvs)",type(qnvs))  # for test
    #print("type(qnvs[0])",type(qnvs[0]))  # for test
    for i in range(nv):
        qntrs[i]=zerotr(n)
    return qnvs


def anytri(v:NDArray[qnv.Qnvec]):
    #shape=v.shape
    #n=shape[1]
    n=len(v)
    print("n",n)  # for test
    v1=Qntri()
    print("v1.shape",v1.shape)  # for test
    for i in range(3):
        #for j in range(n):
        v1[i]=v[i]
    return v1
  
def eq(qnv1:Qntri, qnv2:Qntri):
    n_=qnv1.n
    # angular sort necessary
    for i in range(3):
        #for j in range(n_):
        if qnv1[i]!=qnv2[i]:
            return False
    return True

def not_eq(qnv1:Qntri, qnv2:Qntri):
    n_=qnv1.n
    # angular sort necessary
    for i in range(3):
        #for j in range(n_):
        if qnv1[i]!=qnv2[i]:
            return True
    return False

def wt_qntri(tri:Qntri):
    shap=tri.shape
    print("shape",shape)
    for i in range(3):
        qnv.printqnv("tri",tri[i])
