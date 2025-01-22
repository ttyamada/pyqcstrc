import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
from numpy.typing import NDArray

class Qnmat:

    def __init__(self,n:np.int64, N:np.int64):
        #print("n",n)
        self.mt=np.ndarray((n,n),dtype=qnn.Qnnum) # 2D array
        #mt=np.array((n,n),dtype=qnn.Qnnum)
        qn0=qnn.int2qnn(0,N)
        for i in range(n):
            for j in range(n):
                self.mt[i][j]=qn0
        #self.mt=copy(mt,N)
        self.shape=np.copy((n,n))
        self.n=np.copy(n)
        self.N=np.copy(N)
        print("self.shape",self.shape[0],self.shape[1])
        print("self.n",self.n)

    def set_mt(self,mt:np.ndarray):
        n=self.n
        N=self.N
        for i in range(n):
            for j in range(n):
                self.mt[i][j]=qnn.copy(mt[i][j]) # copy qnnum


    def __add__(ma1, ma2):  #  for ma1+ma2
        return add(ma1,ma2)
    
    def __iadd__(ma1, ma2):  #  for ma1+=ma2
        return iadd(ma1,ma2)
    
    def __isub__(ma1, ma2):  #  for ma1-=ma2
        return isub(ma1,ma2)
        
    def __matmul__(ma1, ma2):  #  for ma1@ma2
        return matmul(ma1,ma2)
    
def zerom(n:np.int64, N: np.int64):
    qnm=Qnmat(n,N)
    return qnm

def unitm(n:np.int64, N: np.int64):
    qn1=qnn.Qnnum([1,0,1],N)
    qnm=zerom(n,N)
    for i in range(n):
        qnm.mt[i][i]=qn1
        
def copy(qnm: Qnmat) -> Qnmat:
#    return np.copy(qnm)
    n=qnm.n
    N=qnm.N
    qnr=Qnmat(n,N)
    qnr.n=np.copy(qnm.n)
    qnr.N=np.copy(qnm.N)
    qnr.shape=np.copy(qnm.shape)
    print("qnr.n",qnr.n,"qnr.N",qnr.N,"qnr.shape",qnr.shape)
    for i in range(n):
        for j in range(n):
            qnr.mt[i][j]=qnn.copy(qnm.mt[i][j])
    return qnr

def int2qnm(r:np.ndarray,n:np.int64,N: np.int64):
    qnr=Qnmat(n,N)
    for i in range(n):
        for j in range(n):
            qnr.mt[i][j]=int2qnn(r[i][j],N)
    return qnr
    
def add(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    #a=np.empty(mat1.shape, dtype=qnn.Qnnum)
    la1=ma1.shape
    la2=ma2.shape
    n1=ma1.n
    n2=ma2.n
    N=mat1.N
    a=Qnmat(n1,N)
    if(n1==1 and n2==1):
        for i in range(la[0]):
            a.mt[i]=ma1.mt[i]+ma2.mt[i]  #add(v1[i],v2[i])
        return a
    elif(n1==2 and n2==2):
        for i in range(la[0]):
            for j in range(la[1]):
                a.mt[i][j]=ma1.mt[i][j]+ma2.mt[i][j]  #add(v1[i],v2[i])
        return a 
    
def sub(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    #a=np.empty(mat1.shape, dtype=qnn.Qnnum)
    la1=ma1.shape
    la2=ma2.shape
    n1=ma1.n
    n2=ma2.n
    N=mat1.N
    a=Qnmat(n1,N)
    if(n1==1 and n2==1):
        for i in range(la[0]):
            a.mt[i]=ma1.mt[i]-ma2.mt[i]  #add(v1[i],v2[i])
        return a
    elif(n1==2 and n2==2):
        for i in range(la[0]):
            for j in range(la[1]):
                a.mt[i][j]=ma1.mt[i][j]-ma2.mt[i][j]  #add(v1[i],v2[i])
        return a 
    
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
    n1=ma1.n
    n2=ma2.n
    print("la1",la1,"la2",la2,"n1",n1,"n2",n2)
    if n1==1 and n2==1: # inner product of qnvec
        N=ma1.mt[0].N
        qnzero=qnn.Qnnum([0,0,1],N) # qnnumber zero
        sum=qnzero
        for i in range(la1[0]):
            sum=sum+ma1.mt[i]*ma2.mt[i]
            return sum
    elif n1==2 and n2==2:
        N=ma1.mt[0][0].N
        qn0=qnn.Qnnum([0,0,1],N) # qnnumber zero
        ma3=zeromat(qn0,la1[0],la2[1]) #"qnnumber zero vector"
        for i in range(la1[0]):
            for j in range(la2[1]):
                for k in range(la2[0]):
                    #ma3.mt[i]+=ma1.mt[i][j]*ma2.mt[j]
                    ma3.mt[i][j]=ma3.mt[i][j]+ma1.mt[i][k]*ma2.mt[k][j]
        return ma3
    elif n1==2 and n2==1 : # qnmat@qnvec
        N=ma2.mt[0].N
        qnzero=qnn.Qnnum([0,0,1],N) # qnnumber zero
        ma3=zerov(qnzero,la1[0]) #"qnnumber zero vector"
        for i in range(la1[0]):
            for j in range(la1[1]):
                #ma3.mt[i]+=ma1.mt[i][j]*ma2.mt[j]
                ma3.mt[i]=ma3.mt[i]+ma1.mt[i][j]*ma2.mt[j]
        return ma3
    elif n1==1 and n2==2 : # qnvec@qnmat
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
            mai = copy(ma) # copy for qnmatinv
            qnmatinv(mai,n)
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

#def det_matrix(mtx: Qnmat) -> Qnmat:
#    """Determinant of 3x3 matrix, mtx, in TAU style
#    
#    Parameters
#    ----------
#    mtx: array
#        3x3 matrix in SQRT3-style
#
#    Returns
#    -------
#    6d vectors projected onto Eperp in SQRT3-style.
#    """
#    
#    t3=mtx[0][0]*mtx[1][1]  #mul(mtx[0][0],mtx[1][1])
#    t1=t3*mtx[2][2]         #mul(t3,mtx[2][2])
#    #
#    t3=mtx[0][2]*mtx[1][0]  #mul(mtx[0][2],mtx[1][0])
#    t2=t3*c[1]              #mul(t3,c[1])
#    #
#    t1=t1+t2                #add(t1,t2)
#    
#    t3=mtx[0][1]*mtx[1][2]  #mul(mtx[0][1],mtx[1][2])
#    t3=t3*mtx[2][0]         #mul(t3,mtx[2][0])
#    #
#    t1=t1+t3                #add(t1,t3)
#    
#    t3=mtx[0][2]*mtx[1][1]  #mul(mtx[0][2],mtx[1][1])
#    t2=t3*mtx[2][0]         #mul(t3,mtx[2][0])
#    #
#    t1=t1-t2                #sub(t1,t2)
#    
#    t3=mtx[0][1]*mtx[1][0]  #mul(mtx[0][1],mtx[1][0])
#    t2=t3*mtx[2][2]         #mul(t3,mtx[2][2])
#    #
#    t1=t1-t2                #sub(t1,t2)
#    
#    t3=mtx[0][0]*mtx[1][2]  #mul(mtx[0][0],mtx[1][2])
#    t2=t3*mtx[2][1]         #mul(t3,mtx[2][1])
#    #
#    t1=t1-t2                #sub(t1,t2)
#    #
#    return t1
    
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

def intm2qnm(a:np.array,n:np.int64,N:np.int64):
    b=Qnmat(n,N) #qnnum zero vector
    for i in range(n):
        for j in range(n):
            b.mt[i][j]=qnn.int2qnn(a[i][j],N)
    return b

def printqnm(str,qnm:Qnmat):
    print(str)
    n=qnm.n
    if n==1:
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            print(qnn.qn2npa(qnm.mt[i]),end=" ")
        print("]")
    elif n==2:
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            for j in range(qnm.shape[1]):
                print(qnn.qn2npa(qnm.mt[i][j]),end=" ")
            print("]")
        print("]")

if __name__ == '__main__':
    # test
    
    N=np.int64(2)
    n=np.int64(6)
    print("n=",n)
    qnm=zerom(n,N) #qnm=Qnmat(n,N)  # nxn qmnum zero matrix
    print("qnm.shape",qnm.shape[0],qnm.shape[0])
    print("qnm.n",qnm.n)
    printqnm("zero matrix",qnm)
    