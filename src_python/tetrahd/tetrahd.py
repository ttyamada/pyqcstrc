import numpy as np
from numpy.typing import NDArray
from typing import Self

import crsys
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnndarray as qna

class Qntet(qna.QnNdarray):
    def __new__(cls):
        global shape
        shape=(4,n_i)
        return super().__new__(cls,shape)
    
    def __init__(self):
        qn0=qnn.zero() #int2qnn(0,N)
        self.n=n_i
        self.N=N
        self.shape=shape
        for i in range(4):
            for j in range(n_i):
                self[i][j]=qn0

def qntet_init():
    global n,N,isys,n_e,n_i
    isys=crsys.isys
    n=crsys.n
    N=crsys.N
    if isys==2:
        n_e=3; n_i=3
    else:
        print("isys shoud be 2 for tetrahd")
        exit()
        
def zerotet(n_i:np.int64):
    qn0=qnn.zero()
    v1=Qntet(n_i)
    for i in range(4):
        for j in range(n_i):
            v1[i][j]=qn0
    return v1
        
def zerotets(shape:np.int64) -> Qntet:  # qntri ndarray
    #print("shape in zerovs",shape)  # for test
    nv=shape[0]
    n=shape[1]
    #qnvs=np.zeros(shape,dtype=Qnvec)
    #qntets = [Qntet(n_i) for i in range(nv)] # list
    qntrs=qna.zeros((nv),dtype=Qntet)
    #print("type(qnvs)",type(qnvs))  # for test
    #print("type(qnvs[0])",type(qnvs[0]))  # for test
    for i in range(nv):
        qntets[i]=zerotet(n)
    return qnvs

def anytet(v:NDArray[qnn.Qnnum]):
    shape=v.shape
    n=shape[1]
    v1=Qntet()
    for i in range(4):
        #for j in range(n):
        v1[i]=v[i]
    return v1

def eq(qntr1:Qntet, qntr2:Qntet):
    n_=qnv1.n
    # angular sort necessary
    for i in range(4):
        #for j in range(n_):
        if qnv1[i]!=qnv2[i]:
            return False
    return True

def not_eq(qnv1:Qntet, qnv2:Qntet):
    n_=qnv1.n
    # angular sort necessary
    for i in range(4):
        #for j in range(n_):
        if qnv1[i]!=qnv2[i]:
            return True
    return False

def wt_qntet(tet:Qntet):
    shap=tet.shape
    print("shape",shape)
    for i in range(4):
        qnv.printqnv("tet",tet[i])
