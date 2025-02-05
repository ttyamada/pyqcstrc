import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnndarray.qnndarray as qna
#from numpy.typing import NDArray

#class Qnmat(np.ndarray):
class Qnmat(qna.QnNdarray):
    def __new__(cls, n:np.int64, N:np.int64):
        shape=(n,n)
        return super().__new__(cls,shape,N)
        #return super().__new__(cls,shape,dtype=qnn.Qnnum)

    def __init__(self,n:np.int64, N:np.int64):
        qn0=qnn.Qnnum([0,0,1],N)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self[i][j]=qn0
        #print("self.shape",self.shape)
        #print("self.ndim",self.ndim)
        #print("self.dtype",self.dtype)
        #printqnm("Qnmat self",self) # for test

    def __add__(ma1, ma2):  #  for ma1+ma2
        return add(ma1,ma2)
    
    def __iadd__(ma1, ma2):  #  for ma1+=ma2
        return iadd(ma1,ma2)
    
    def __isub__(ma1, ma2):  #  for ma1-=ma2
        return isub(ma1,ma2)
        
    def __matmul__(ma1, ma2):  #  for ma1@ma2
        return matmul(ma1,ma2)
    
    def set_mt(self,mt:np.array): # QnNdarray*
        n=self.shape[0]
        #N=self[0][0].N
        for i in range(n):
            for j in range(n):
                self.mt[i][j]=qnn.copy(mt[i][j]) # copy qnnum

def zerom(n:np.int64, N: np.int64):
    qnm=Qnmat(n,N)
    return qnm

def unitm(n:np.int64, N: np.int64):
    qn1=qnn.Qnnum([1,0,1],N)
    qnm=Qnmat(n,N)
    for i in range(n):
        qnm[i][i]=qn1
    return qnm
        
#def copy(qna1: Qnmat):
#    return np.copy(qna1)
        
def copy(qnm: Qnmat) -> Qnmat:
#    return np.copy(qnm)
#   original code
    n=qnm.shape[0]
    N=qnm[0][0].N
    qnr=Qnmat(n,N)
    print("qnr.N",qnr[0][0].N,"qnr.shape",qnr.shape,"ndim",qnr.ndim) # for test
    for i in range(n):
        for j in range(n):
            qnr[i][j]=qnn.copy(qnm[i][j])
    return qnr

def int2qnm(r:np.ndarray,n:np.int64,N: np.int64):
    qnr=Qnmat(n,N)
    for i in range(n):
        for j in range(n):
            qnr[i][j]=int2qnn(r[i][j],N)
    return qnr
    
def add(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    #a=np.empty(mat1.shape, dtype=qnn.Qnnum)
    la1=ma1.shape[0]
    la2=ma1.shape[1]
    n1=ma1.ndim
    n2=ma2.ndim
    N=mat1.N
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
    N=mat1[0][0].N
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
def matmul(ma1: Qnmat, ma2: Qnmat, dtype=qnn.Qnnum) -> Qnmat: 
    ma3=np.matmul(ma1,ma2,dtype=qnn.Qnnum)
    return ma3

# for similarity transformation
# not confirmed yet
def matrixpow(ma: Qnmat, n: int) -> Qnmat:
    """
    """
    (mx,my)=ma.shape
    N=ma[0][0].N
    if mx==my:
        if n==0:
            return np.identity(mx)
        elif n<0:
            tmp=unitm(n,N)
            mai = copy(ma) # copy for qnmatinv
            qnmatinv(mai,n)
            for i in range(-n):
                tmp=np.dot(tmp,inva)
            return tmp
        else:
            tmp=np.unitm(n,N)
            for i in range(n):
                tmp=np.dot(tmp,ma)
            return tmp
    else:
        print('matrix has not regular shape')
        return 

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

# get qnmat from int matrix
def intm2qnm(a:np.array,n:np.int64,N:np.int64):
    b=Qnmat(n,N) #qnnum zero vector
    for i in range(n):
        for j in range(n):
            b[i][j]=qnn.int2qnn(a[i][j],N)
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

if __name__ == '__main__':
    # test
    
    N=2
    n=5
    print("n=",n,"N",N)
    qnm=Qnmat(n,N) # nxn qmnum zero matrix
    print("qnm.ndim",qnm.ndim)
    print("qnm.shape",qnm.shape)
    printqnm("qnm",qnm)
    
    qnm1=copy(qnm)
    print("qnm1.ndim",qnm1.ndim)
    print("qnm1.shape",qnm1.shape)
    printqnm("qnm1",qnm1)
    
    unm1=unitm(n,N) # nxn qmnum zero matrix
    print("unm1.ndim",unm1.ndim)
    print("unm1.shape",unm1.shape)
    printqnm("unm1",unm1)
    
    unm2=unitm(n,N) # nxn qmnum zero matrix
    print("unm2.ndim",unm2.ndim)
    print("unm2.shape",unm2.shape)
    printqnm("unm2",unm2)
    
    unm3=unm2@unm1
