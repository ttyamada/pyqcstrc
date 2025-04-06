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
    n1=ma1.ndim
    n2=ma2.ndim
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
    
def sub(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    #a=np.empty(mat1.shape, dtype=qnn.Qnnum)
    shape=ma1.shape
    n1=ma1.ndim
    n2=ma2.ndim
    #N=ma1[0][0].N
    a=Qnmat(shape)
    if(n1==1 and n2==1): # vectors
        for i in range(shape[0]):
            a[i]=ma1[i]-ma2[i]  #add(v1[i],v2[i])
        return a
    elif(n1==2 and n2==2): # matrices
        for i in range(shape[0]):
            for j in range(shape[1]):
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
    ndm1=ma1.ndim
    ndm2=ma2.ndim
    #print("ndm1=",ndm1,"ndm2=",ndm2)  # for test
    if ndm1==1 and ndm2==1:  # dot product
        if ma1.shape[0]==ma2.shape[0]:
            ma3=qnn.zero()
            for i in range(ma1.shape[0]):
                ma3+=ma1[i]*ma2[i]
        else:
            print("ma1.shape[0] should be equal to ma2.shape[0] for ndim1==ndim2"); exit()
    elif ndm1==1 and ndm2==2:  # vec*matrix
        if ma1.shape[0]==ma2.shape[0]:
            ma3=qnv.zerov(ma2.shape[1])
            for j in range(ma2.shape[1]):
                for i in range(ma1.shape[0]):
                    ma3[j]+=ma1[i]*ma2[i][j]
        else:
            print("ma1.shape[0] should be equal to ma2.shape[0] for ndim1=1 and ndim2=2"); exit()
    elif ndm1==2 and ndm2==1:  # matrix*vec
        if ma1.shape[1]==ma2.shape[0]:
            ma3=qnv.zerov(ma1.shape[0])
            for j in range(ma2.shape[0]):
                for i in range(ma1.shape[0]):
                    ma3[j]+=ma1[j][i]*ma2[i]
        else:
            print("ma1.shape[1] should be equal to ma2.shape[0] for ndim1=2 and ndim2=1"); exit()
    elif ndm1==2 and ndm2==2:
        if ma1.shape[0]==ma1.shape[1] and ma2.shape[0]==ma2.shape[1] and ma1.shape[0]==ma2.shape[0]:
            ma3=qnv.zerov(ma1.shape)
            for k in range(ma2.shape[0]):
                for j in range(ma1.shape[0]):
                    for i in range(ma1.shape[0]):
                        ma3[k][j]+=ma1[k][i]*ma2[i][j]
        else:
             print("square matrix is assumed for ndm1=ndm2=2"); exit()

    #ma3=np.matmul(ma1,ma2,dtype=qnn.Qnnum)
    return ma3

# for similarity transformation
# not confirmed yet
def pow(ma: Qnmat, n_: int) -> Qnmat:
    """
    """
    (mx,my)=ma.shape
    #N=ma[0][0].N
    if mx==my:
        if n_==0:
            return np.identity(mx)
#        elif n<0:
#            tmp=unitm(n,N)
#            mai = copy(ma) # copy for qnmatinv
#            qmt.qnmatinv(mai,n) #???
#            for i in range(-n):
#                tmp=np.dot(tmp,inva)
#            return tmp
        else:
            tmp=np.unitm(n_,N)
            for i in range(n_):
                tmp=np.dot(tmp,ma)
            return tmp
    else:
        print('matrix has not regular shape')
        exit()

def qnm2npa(a) -> np.ndarray:
    # Qnmatrix to np.array converter
    shape=a.shape
    b=np.zeros(shape,dtype=qnn.Qnnum)
    for i in range(shape[0]):
        for j in range(shape[1]):
            b[i][j]=[a[i][j].n[0],a[i][j].n[1],a[i][j].n[2]]
    return b

def qnm2flt(a) -> np.ndarray:
    shape=a.shape
    b=np.zeros(shape,dtype=qnn.Qnnum)
    for i in range(shape[0]):
        for j in range(shape[1]):
            b[i][j]=(a[i][j].n[0]+a[i][j].n[1]*np.sqrt(N))/a[i][j].n[2]
    return b

# get qnmat from int matrix
def intm2qnm(a:np.array,n_:np.int64) -> Qnmat:
    shape=a.shape
    b=Qnmat(shape) #qnnum zero vector
    for i in range(shape[0]):
        for j in range(shape[1]):
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
    elif ndim==3: # for several matrices
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


def printfm(str:str,fm:NDArray[np.float64]):
    #ndim=2  # 
    ndim=fm.ndim
    shape=fm.shape
    #if shape[1]==0:
    #    ndim=1
    #elif shape[2]==0:
    #    ndim=2
    #else:
    #    ndim=3

    print(str)
    if ndim==1:
        for i in range(fm.shape[0]):
            print("[",end=" ")
            print(fm[i],end=" ")
        print("]")
    elif ndim==2:
        for i in range(fm.shape[0]):
            print("[",end=" ")
            for j in range(fm.shape[1]):
                print(fm[i][j],end=" ")
            print("]")
        print("")
    elif ndim==3: # for several matrices
        for i in range(fm.shape[0]):
            print("")
            for j in range(fm.shape[1]):
                print("[",end=" ")
                for k in range(fm.shape[2]):
                    print(fm[i][j][k],end=" ")
                print("]")
        print("")
    else:
        print("ord in printfm should be 1 2 or 3 but",ord); exit()
