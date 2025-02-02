import sys
import numpy as np
from numpy.typing import NDArray
import pyqcstrc.qnnum.qnnum as qnn
#from pyqcstrc.qnvec import qnvec

class Qnvec:
    def __init__(self, n:np.int64, N:np.int64):
 
        self=np.empty(shape=(n),dtype=qnn.Qnnum) # 1D array
        qn0=qnn.Qnnum([0,0,1],N) #int2qnn(0,N)
        for i in range(n):
            self[i]=qn0
        print("self.shape",self.shape)
        print("self.ndim",self.ndim)
        print("self.dtype",self.dtype)

    def __add__(a, b):
        return add(a,b)
    
    def __sub__(a, b):
        return sub(a,b)
    
    def __mul__(a, b): # b should be int or qnnum
        if isinstance(b, int):
            return mul_vector_i(a,b)
        elif isinstance(b, qnn.Qnnum):
            return mul_vector_qn(a,b)
        
    def __truediv__(a,b): # b should be int
        if isinstance(b, int):
            return div_vector_i(a,b)
    
def zerov(n:np.int64,N:np.int64): # qnnumber zero vector
    qnv=Qnvec(n,N)
    return qnv

def add(v1:Qnvec, v2:Qnvec) -> Qnvec:
    a=np.empty(v1.shape, dtype=qnn.Qnnum)
    la=v1.shape
    for i in range(la[0]):
        a[i]=v1[i]+v2[i]  #add(v1[i],v2[i])
    return Qnvec(a)

def sub(v1:Qnvec, v2:Qnvec)-> Qnvec:
    a=np.empty(v1.shape, dtype=qnn.Qnnum)
    la=v1.shape
    for i in range(la[0]):
        a[i]=v1[i]-v2[i]  #add(v1[i],v2[i])
    return Qnvec(a)

def mul_vector_i(v:Qnvec, coeff:int):
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

def mul_vector_qn(v:Qnvec, coeff:qnn.Qnnum):
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

def mul_vectors_i(vs:Qnvec, coeff:int):
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

def div_vector_i(v:Qnvec, coeff: int):
    if v.ndim==1:
        a=np.zeros(v.shape,dtype=np.int64)
        la=v.shape
        for i in range(la[0]):
            a[i]=v[i]/coeff  #mul(v,coeff)
        return a
    else:
        print('incorrect shape')
        return


def shift_vectors(vs:Qnvec, v:Qnvec) -> Qnvec:
    if vs.ndim==1:  #3:
        a=np.zeros(vs.shape,dtype=qnn.Qnvec)
        la=vs.shape
        for i,v1 in enumerate(vs):  #range(la[0]):
            a[i]=add_vectors(v1,v)
        return a
    elif vs.ndim==2:  #4:
        a=np.zeros(vs.shape,dtype=qnn.Qnvec)
        for i1,v1 in enumerate(vs):
            for i2,v2 in enumerate(v1):
                a[i1][i2]=add_vectors(v2,v)
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
    N=v1[0].N
    n=v1.shape[0]
    v=qnn.Qnnum([0,0,1],N) # qnnum zero
    for i in range(n):
        v=v+v1[i]*v2[i]
    return v
    
# equivalent to cross
def outer_product(v1:Qnvec, v2:Qnvec)-> Qnvec:
    return cross(v1,v2)

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

def printqnv(str:str,qnv1:Qnvec):
    print(str,"[",end=" ")
    n=qnv1.shape[0]
    for i in range(n):
        j=qnv1[i]
        print(qnn.qn2npa(j),end=" ")
    print("]")

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
    
    
if __name__ == '__main__':
    # test
    N=np.int64(2)
    n=np.int64(6)
    print("n=",n)
    #qnv=Qnvec(n,N)  # nD qnnum zero vector
    #print("qnv.shape",qnv.shape)
    #printqnv("Qnvec_Octa",qnv)
    
    M0=qnn.Qnnum([0,0,1],N)
    M1=qnn.Qnnum([1,0,1],N)
    M2=qnn.Qnnum([0,1,1],N)
    
    qnv1=np.array([M0,M1,M2])
    qnv2=np.array([M1,M2,M0])
    print("qnv1.shape",qnv1.shape)
    print("qnv1.ndim",qnv1.ndim)
    printqnv("qnv1",qnv1)
    printqnv("qnv2",qnv2)
    
    qnv3=qnv1+qnv2
    qnv4=qnv1-qnv2
    printqnv("qnv3",qnv3)
    printqnv("qnv4",qnv4)
    
    qnv5=dot(qnv1,qnv2)
    qnn.printqnn("qnv5",qnv5)
    
    qnv6=cros(qnv1,qnv2)
    printqnv("qnv6",qnv6)
    
