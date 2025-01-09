import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
from numpy.typing import NDArray

class Qnmat:
    #def __init__(self, mt:np.ndarray[qnnum], shape:np.array):
    def __init__(self, mt:np.ndarray[qnn]):
        self.mt=mt
        self.shape=mt.shape
        self.ndim=mt.ndim
        
    def __add__(ma1, ma2):  #  for ma1+ma2
        return add(ma1,ma2)
    
    def __iadd__(ma1, ma2):  #  for ma1+ma2
        return iadd(ma1,ma2)
    
    def __isub__(ma1, ma2):  #  for ma+=ma2
        return isub(ma1,ma2)
        
    def __matmul__(ma1, ma2):  #  for ma1@ma2
        return matmul(ma1,ma2)
    
def zerom(n:np.int64, N: np.int64):
    qnzero=qnn.Qnnum([0,0,1],N)
    qnm=Qnmat(np.full(n,qnzero))
    return qnm

def unitm(n:np.int64, N: np.int64):
    qnzero=qnn.Qnnum([0,0,1],N)
    qnone=qnn.Qnnum([1,0,1],N)
    qnm=Qnmat(np.full(n,qnzero)) # qn zero matrix
    for i in range(n):
        qnm.mt[i][i]=qnone
    
def add(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    a=np.empty(mat1.shape, dtype=qnn.Qnnum)
    la1=ma1.shape
    la2=ma2.shape
    n1=ma1.ndim
    n2=ma2.ndim
    if(n1==1 and n2==1):
        for i in range(la[0]):
            a[i]=ma1.mt[i]+ma2.mt[i]  #add(v1[i],v2[i])
        return Qnmat(a)
    elif(n1==2 and n2==2):
        for i in range(la[0]):
            for j in range(la[1]):
                a[i][j]=ma1.mt[i][j]+ma2.mt[i][j]  #add(v1[i],v2[i])
        return Qnmat(a) 
    
def iadd(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    ma1=add(ma1,ma2)
    return ma1

def isub(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    ma1=sub(ma1,ma2)
    return ma1
    
# for ma1@ma2 (ma1 and ma2 should be qnvec or qnmat)
def matmul(ma1: Qnmat, ma2: Qnmat) -> Qnmat: 
    la1=ma1.shape
    la2=ma2.shape
    n1=ma1.ndim
    n2=ma2.ndim
    print("la1",la1,"la2",la2,"n1",n1,"n2",n2)
    if(n1==1 and n2==1): # inner product of qnvec
        N=ma1.mt[0].N
        qnzero=qnn.Qnnum([0,0,1],N) # qnnumber zero
        sum=qnzero
        for i in range(la1[0]):
            sum=sum+ma1.mt[i]*ma2.mt[i]
            return sum
    elif(n1==2 and n2==1): # qnmat@qnvec
        N=ma2.mt[0].N
        qnzero=qnn.Qnnum([0,0,1],N) # qnnumber zero
        ma3=zerov(qnzero,la1[0]) #"qnnumber zero vector"
        for i in range(la1[0]):
            for j in range(la1[1]):
                #ma3.mt[i]+=ma1.mt[i][j]*ma2.mt[j]
                ma3.mt[i]=ma3.mt[i]+ma1.mt[i][j]*ma2.mt[j]
        return ma3
    elif(n1==1 and n2==2): # qnvec@qnmat
        N=ma1.mt[0].N
        qnzero=qnn.Qnnum([0,0,1],N) # qnnumber zero
        ma3=zerov(qnzero,la1[0]) #"qnnumber zero vector"
        for i in range(la2[0]):
            for j in range(la2[1]):
                #ma3.mt[i]+=ma1.mt[j]*ma2.mt[j][i]
                ma3.mt[i]=ma3.mt[i]+ma1.mt[j]*ma2.mt[j][i]
        return ma3
                
# for similarity transformation
# not confirmed yet
def matrixpow(ma: Qnmat, n: int) -> Qnmat:
    """
    """
    (mx,my)=ma.shape
    if mx==my:
        if n==0:
            return np.identity(mx)
        elif n<0:
            tmp=np.identity(mx)
            inva = np.linalg.inv(ma) # matrix inversion
            for i in range(-n):
                tmp=np.dot(tmp,inva)
            return tmp
        else:
            tmp=np.identity(mx)
            for i in range(n):
                tmp=np.dot(tmp,ma)
            return tmp
    else:
        print('matrix has not regular shape')
        return 

def det_matrix(mtx: Qnmat) -> Qnmat:
    """Determinant of 3x3 matrix, mtx, in TAU style
    
    Parameters
    ----------
    mtx: array
        3x3 matrix in SQRT3-style

    Returns
    -------
    6d vectors projected onto Eperp in SQRT3-style.
    """
    
    t3=mtx[0][0]*mtx[1][1]  #mul(mtx[0][0],mtx[1][1])
    t1=t3*mtx[2][2]         #mul(t3,mtx[2][2])
    #
    t3=mtx[0][2]*mtx[1][0]  #mul(mtx[0][2],mtx[1][0])
    t2=t3*c[1]              #mul(t3,c[1])
    #
    t1=t1+t2                #add(t1,t2)
    
    t3=mtx[0][1]*mtx[1][2]  #mul(mtx[0][1],mtx[1][2])
    t3=t3*mtx[2][0]         #mul(t3,mtx[2][0])
    #
    t1=t1+t3                #add(t1,t3)
    
    t3=mtx[0][2]*mtx[1][1]  #mul(mtx[0][2],mtx[1][1])
    t2=t3*mtx[2][0]         #mul(t3,mtx[2][0])
    #
    t1=t1-t2                #sub(t1,t2)
    
    t3=mtx[0][1]*mtx[1][0]  #mul(mtx[0][1],mtx[1][0])
    t2=t3*mtx[2][2]         #mul(t3,mtx[2][2])
    #
    t1=t1-t2                #sub(t1,t2)
    
    t3=mtx[0][0]*mtx[1][2]  #mul(mtx[0][0],mtx[1][2])
    t2=t3*mtx[2][1]         #mul(t3,mtx[2][1])
    #
    t1=t1-t2                #sub(t1,t2)
    #
    return t1
    
def qnm2npa(a):
    # Qnmatrix to np.array converter
    la=a.shape #len(a)
    b=np.zeros(la[0],la[1],3) #la x la qnnum matrix 
    for i in range(la[0]):
        for j in range(la[1]):
            b[i][j]=[a[i][j].n[0],a[i][j].n[1],a[i][j].n[2]]
    return b

def qnm2flt(a):
    la=a.shape #len(a)
    b=np.zeros(la[0],la[1],3)  #la x la qnnum matrix 
    N=a[0].N
    for i in range(la[0]):
        for j in range(la[1]):
            b[i][j]=(a[i][j].n[0]+a[i][j].n[1]*np.sqrt(N))/a[i][j].n[2]
    return b

def intm2qnm(a:np.array,N:np.int64):
    qn0=qnn.Qnnum([0,0,1],N) # qnnum zero
    la=a.shape
    b=Qnmat(np.full(la,qn0)) #qnnum zero vector
    for i in range(la[0]):
        for j in range(la[1]):
            b.mt[i][j]=qnn.int2qnn(a[i][j],N)
    return b

def printqnm(str,qnm):
    print(str)
    n1=qnm.ndim
    if n1==1:
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            print(qnn.qn2npa(qnm.mt[i]),end=" ")
        print("]")
    elif n1==2:
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            for j in range(qnm.shape[1]):
                print(qnn.qn2npa(qnm.mt[i][j]),end=" ")
            print("]")
        print("]")

if __name__ == '__main__':
    # test
    import random
    sys.path.append('.')
    import numericalc

    ncycle=20
    eps=1e-3
    