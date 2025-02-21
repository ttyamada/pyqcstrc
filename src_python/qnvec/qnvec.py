import sys
import numpy as np
from numpy.typing import NDArray
import qnnum.qnnum as qnn
import qnvec.qnvec as qnv
import qnndarray.qnndarray as qna
from typing import Self

#class Qnvec(np.ndarray):
#class Qnvec(np.ndarray):
class Qnvec(qna.QnNdarray):
    def __new__(cls, n:np.int64, N:np.int64):
        global shape
        shape=(n)
        return super().__new__(cls,shape,N)
    
    def __init__(self, n:np.int64, N:np.int64):
        qn0=qnn.Qnnum([0,0,1],N) #int2qnn(0,N)
        self.n=n
        self.N=N
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
        
def zerovs(shape) -> Qnvec:  # qnvec ndarray
    return np.zeros(shape,dtype=Qnvec)

def zerov(n:np.int64,N:np.int64)->Qnvec: # qnnumber zero vector
    qnv=Qnvec(n,N)
    return qnv

def anyv(n:np.int64,N:np.int64,vec:qnn.Qnnum)->Qnvec:
    qnv=Qnvec(n,N)
    for i in range(n):
        qnv[i]=vec[i]
    return qnv

def copy(v1: Qnvec) -> Qnvec:
    return np.copy(v1)
    
def add(v1:Qnvec, v2:Qnvec) -> Qnvec:
    n=v1.shape[0]
    N=v1[0].N
    a=Qnvec(n,N)
    for i in range(n):
        a[i]=v1[i]+v2[i]
    return a

def sub(v1:Qnvec, v2:Qnvec)-> Qnvec:
    n=v1.shape[0]
    N=v1[0].N
    a=Qnvec(n,N)
    for i in range(n):
        a[i]=v1[i]-v2[i]
    return a

def mul_vector_i(v:Qnvec, coeff:int) -> qnv.Qnvec:
    if v.ndim==1:
        n=v.shape
        N=v.N
        a=Qnvec(n,N)  #np.zeros(v.shape,dtype=np.int64)
        for i in range(n):
            a[i]=v[i]*coeff  #mul(v,coeff)
        return a
    else:
        print('incorrect shape')
        return

def mul_vector_qn(v:Qnvec, coeff:qnn.Qnnum) -> Qnvec:
    if v.ndim==1:
        n=v.shape
        N=v.N
        a=Qnvec(n,N)  #np.zeros(v.shape,dtype=np.int64)
        for i in range(n):
            a[i]=v[i]*coeff  #mul(v,coeff)
        return a
    else:
        print('incorrect shape')
        return

def mul_vectors_i(vs:Qnvec, coeff:int) -> Qnvec:
    if vs.ndim==2:
        n=vs.shape[0]
        N=vs[0].N
        a=[Qnvec(n,N)]*vs.shape
        la=vs.shape
        for i in range(2):
            for j in range(n):
                a[i]=vs[i][j]*coeff #mul_vector(v,coeff)
        return a
    else:
        print('incorrect shape')
        return

def mul_vectors_qn(vs:Qnvec, coeff:qnn.Qnnum):
    if vs.ndim==2:
        n=vs.shape[0]
        N=vs[0].N
        a=[Qnvec(n,N)]*vs.shape
        la=vs.shape
        for i in range(2):
            for j in range(n):
                a[i]=vs[i][j]*coeff #mul_vector(v,coeff)
        return a
    else:
        print('incorrect shape')
        return

def div_vector_i(v:Qnvec, coeff: int) -> Qnvec:
    if v.ndim==1:
        a=np.zeros(v.shape,dtype=np.int64)
        la=v.shape
        for i in range(la[0]):
            a[i]=v[i]/coeff  #mul(v,coeff)
        return a
    else:
        print('incorrect shape')
        return
    
# cros == outer_product (cross product) 
def cros(v1:Qnvec, v2:Qnvec) -> Qnvec:
    a=v1[1]*v2[2] #mul(v1[1],v2[2])
    b=v1[2]*v2[1] #mul(v1[2],v2[1])
    c1=a-b        #sub(a,b)
    #
    a=v1[2]*v2[0] #mul(v1[2],v2[0])
    b=v1[0]*v2[2] #mul(v1[0],v2[2])
    c2=a-b        #sub(a,b)
    #
    a=v1[0]*v2[1] #mul(v1[0],v2[1])
    b=v1[1]*v2[0] #mul(v1[1],v2[0])
    c3=a-b        #sub(a,b)
    #return np.array([c1,c2,c3],dtype=qnn.Qnnumber)
    return np.array([c1,c2,c3])

#def dot(v1:Qnvec, v2:Qnvec) -> qnn.Qnnum:
#    N=v1.N
#    qnn.Qnnum([0,0,1],N) # zero qnnumber
#    return   v1[0]*v2[0]+v1[1]*v2[1]-v1[1]*v2[0]-v1[0]*v2[1]
    
# for octagonal and dodecagonal
def dot(v1:Qnvec, v2:Qnvec) -> qnn.Qnnum:
    N=v1.N
    n=v1.shape[0]
    v=qnn.Qnnum([0,0,1],N) # qnnum zero
    for i in range(n):
        v=v+v1[i]*v2[i]
    return v
    
# equivalent to cross
def outer_product(v1:Qnvec, v2:Qnvec)-> Qnvec:
    return cros(v1,v2)

# equivalent to dot
def inner_product(v1:Qnvec, v2:Qnvec) -> qnn.Qnnum:
    return dot(v1,v2)

def qnv2npa(a:Qnvec):
    # Qnvector to np.array converter
    la=a.shape
    #print("la",la)
    print("la",la,"la[0]",la[0],"range(la[0])",range(la[0]))
    
    b=np.empty((la[0],3),dtype=np.int64)
    print("b",b)
    for i in range(la[0]):
        ai=a[i]
        b[i]=[ai.n[0],ai.n[1],ai.n[2]]
    return b
    
def qnv2flt(a:Qnvec):
    la=a.shape
    b=np.zeros(la, dtype=np.float64)
    #print("b",b)
    N=a[0].N
    for i in range(la):
        ai=a[i]
        b[i]=(ai.n[0]+ai.n[1]*np.sqrt(N))/ai.n[2]
    return b

def intv2qnv(a:np.ndarray,N:np.int64):
    n=a.shape[0]
    qn0=qnn.Qnnum([0,0,1],N) # qnnum zero
    b=Qnvec(np.full(n,qn0)) #qnnum zero vector
    for i in range(n):
        b[i]=qnn.int2qnn(a[i],N)
    return b

#def printqnv(str:str,qnv:Qnvec):
def printqnv(str:str,qnv:qna.QnNdarray):
    ndim=qnv.ndim
    if ndim==1 :
        print(str,"[",end=" ")
        for i in range(qnv.shape[0]):
            print(qnn.qn2npa(qnv[i]),end=" ")
        print("]")
    elif ndim==2:
        for i in range(qnv.shape[0]):
            print("[",end=" ")
            for j in range(qnv.shape[1]):
                print(qnn.qn2npa(qnv[i][j]),end=" ")
            print("]")
        print("")

def printqnv2(str:str,qnv1:Qnvec,qnv2:Qnvec):
    print(str,"[",end=" ")
    n1=qnv1.shape
    for i in range(n1):
        j=qnv1[i]
        print(qnn.qn2npa(j),end=" ")
    print("] [",end="")
    n2=qnv2.shape
    for i in range(n2):
        j=qnv1[i]
        print(qnn.qn2npa(j),end="]")
        
def printqnvs(str:str,qnv1:qnv.Qnvec):
    shape=qnv1.shape
    print("shape",shape)
    n1=shape[0]
    for j in range(n1):
        print(str+"["+format(j)+"]",end=" ")
        printqnv("",qnv1[j])
     
def eq(qnv1:Qnvec, qnv2:Qnvec):
    n=qnv1.n
    for i in range(n):
        if qnv1[i]!=qnv2[i]:
            return False
    return True

def not_eq(qnv1:Qnvec, qnv2:Qnvec):
    n=qnv1.n
    for i in range(n):
        if qnv1[i]!=qnv2[i]:
            return True
    return False
    
    
    
