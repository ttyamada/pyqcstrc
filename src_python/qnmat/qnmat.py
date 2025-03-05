import sys
import numpy as np
import cython

import qnnum as qnn
import qnvec as qnv
#import qnmath as qmt
import qnndarray as qna
from typing import Self
from numpy.typing import NDArray

#class Qnmat(np.ndarray):
class Qnmat(qna.QnNdarray):
    def __new__(cls, shape:np.int64, N:np.int64):
        return super().__new__(cls,shape,N)
        #return super().__new__(cls,shape,dtype=qnn.Qnnum)

    def __init__(self,shape:np.int64, N:np.int64):
        #global n,N
        qn0=qnn.zero(N)
        self.N=N
        self.shape=shape
        for i in range(n):
            for j in range(n):
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
    
    def set_mt(self:Self,mt:np.array): # QnNdarray*
        n=self.shape[0]
        #N=self[0][0].N
        for i in range(n):
            for j in range(n):
                self.mt[i][j]=qnn.copy(mt[i][j]) # copy qnnum

def zerom(shape:np.int64, N: np.int64) -> Qnmat:
    qnm=Qnmat(shape,N)
    return qnm

#def zeroms(shape:np.int64,N:np.int64) -> Qnmat: # qnmat ndarray
#    return np.zeros(shape,dtype=Qnmat)

def anym(m:NDArray[qnn.Qnnum]):
    N=m.N
    shape=m.shape
    m1=Qnmat(shape,N)
    for i in range(shape[0]):
        for j in range(shape[1]):
            m1[i][j]=m[i][j]
    return m1

def unitm(n:np.int64, N: np.int64) -> Qnmat:
    qn1=qnn.Qnnum([1,0,1],N)
    shape=(n,n)
    qnm=Qnmat(shape,N)
    for i in range(n):
        qnm[i][i]=qn1
    return qnm
        
def copy(qnm: Qnmat) -> Qnmat:
    return np.copy(qnm)
    # original code
    #n=qnm.n
    #N=qnm.N
    #qnm1=Qnmat(n,N)
    #print("qnm1.N",qnm1.N,"qnm1.shape",qnm1.shape,"ndim",qnm1.ndim) # for test
    #for i in range(n):
    #    for j in range(n):
    #        qnm1[i][j]=qnn.copy(qnm[i][j])
    #return qnm1

def int2qnm(r:np.ndarray,n:np.int64,N: np.int64) -> Qnmat:
    qnr=Qnmat(n,N)
    for i in range(n):
        for j in range(n):
            qnr[i][j]=qnn.int2qnn(r[i][j],N)
    return qnr
    
def add(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    #a=np.empty(mat1.shape, dtype=qnn.Qnnum)
    la1=ma1.shape[0]
    la2=ma1.shape[1]
    n1=ma1.ndim
    n2=ma2.ndim
    N=ma1.N
    a=Qnmat(n1,N)
    if(n1==1 and n2==1): # vectors
        for i in range(la1):
            a[i]=ma1[i]+ma2[i]  #add(v1[i],v2[i])
        return a
    elif(n1==2 and n2==2): # matrices
        for i in range(la1):
            for j in range(la2):
                a[i][j]=ma1[i][j]+ma2[i][j]  #add(v1[i],v2[i])
        return a 
    
def sub(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    #a=np.empty(mat1.shape, dtype=qnn.Qnnum)
    la1=ma1.shape[0]
    la2=ma2.shape[1]
    n1=ma1.ndim
    n2=ma2.ndim
    N=ma1[0][0].N
    a=Qnmat(n1,N)
    if(n1==1 and n2==1): # vectors
        for i in range(la1):
            a[i]=ma1[i]-ma2[i]  #add(v1[i],v2[i])
        return a
    elif(n1==2 and n2==2): # matrices
        for i in range(la1):
            for j in range(la2):
                a[i][j]=ma1[i][j]-ma2[i][j]  #add(v1[i],v2[i])
        return a 
    
def iadd(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    ma1=add(ma1,ma2)
    return ma1

def isub(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    ma1=sub(ma1,ma2)
    return ma1
    
# for ma1@ma2 (ma1 and ma2 should be qnvec or qnmat)
def mul(ma1: Qnmat, ma2: Qnmat, dtype=qnn.Qnnum) -> Qnmat: 
    ma3=np.matmul(ma1,ma2,dtype=qnn.Qnnum)
    return ma3

# for similarity transformation
# not confirmed yet
def pow(ma: Qnmat, n: int) -> Qnmat:
    """
    """
    (mx,my)=ma.shape
    N=ma[0][0].N
    if mx==my:
        if n==0:
            return np.identity(mx)
#        elif n<0:
#            tmp=unitm(n,N)
#            mai = copy(ma) # copy for qnmatinv
#            qmt.qnmatinv(mai,n) #???
#            for i in range(-n):
#                tmp=np.dot(tmp,inva)
#            return tmp
        else:
            tmp=np.unitm(n,N)
            for i in range(n):
                tmp=np.dot(tmp,ma)
            return tmp
    else:
        print('matrix has not regular shape')
        return 

def qnm2npa(a) -> np.ndarray:
    # Qnmatrix to np.array converter
    la=a.shape #len(a)
    b=np.zeros(la[0],la[1],3) #la x la qnnum matrix 
    for i in range(la[0]):
        for j in range(la[1]):
            b[i][j]=[a[i][j].n[0],a[i][j].n[1],a[i][j].n[2]]
    return b

def qnm2flt(a) -> np.ndarray:
    la=a.shape #len(a)
    b=np.zeros(la[0],la[1],3)  #la x la qnnum matrix 
    N=a[0].N
    for i in range(la[0]):
        for j in range(la[1]):
            b[i][j]=(a[i][j].n[0]+a[i][j].n[1]*np.sqrt(N))/a[i][j].n[2]
    return b

# get qnmat from int matrix
def intm2qnm(a:np.array,n:np.int64,N:np.int64) -> Qnmat:
    b=Qnmat(n,N) #qnnum zero vector
    for i in range(n):
        for j in range(n):
            b[i][j]=qnn.int2qn(a[i][j],N)
    return b

#def printqnm(str:str,qnm:Qnmat):
#def printqnm(str:str,qna:QnNdarray):
def printqnm(str:str,qnm:qna.QnNdarray):
    #ndim=2  # 
    ndim=qnm.ndim
    shape=qnm.shape
    #if shape[1]==0:
    #    ndim=1
    #elif shape[2]==0:
    #    ndim=2
    #else:
    #    ndim=3

    print(str)
    if ndim==1:
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            print(qnn.qn2npa(qnm[i]),end=" ")
        print("]")
    elif ndim==2:
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            for j in range(qnm.shape[1]):
                print(qnn.qn2npa(qnm[i][j]),end=" ")
            print("]")
        print("")
    elif ndim==3:
        for i in range(qnm.shape[0]):
            print("")
            for j in range(qnm.shape[1]):
                print("[",end=" ")
                for k in range(qnm.shape[2]):
                    print(qnn.qn2npa(qnm[i][j][k]),end=" ")
                print("]")
        print("")
    else:
        print("ord in printqnm should be 1 2 or 3 but",ord); exit()

