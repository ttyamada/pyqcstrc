import sys
import numpy as np
from numpy.typing import NDArray
from typing import Self

import crsys as crs
import qnnum as qnn
import qnndarray as qna

# cros == outer_product (cross product) 
# n should be 3
def cros(v1:Qnvec, v2:Qnvec) -> Qnvec:
    if n!=3:
        print("dimention should be 3 for cross product")
        exit()
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
    
# for dihedral and icosahedral excluding decagonal
def dot(v1:Qnvec, v2:Qnvec) -> qnn.Qnnum:
    #print("v1.shape",v1.shape)  # for test
    #printqnv("v1",v1)  # for test
    #printqnv("v2",v2)  # for test
    isys=crs.isys
    if isys==3:
        s2=crs.scly2 # qnnumber
        v=v1[0]*v2[0]+v1[1]*v2[1]*s2
    else:
        v=qnn.zero() # qnnum zero
        n=v1.shape[0]
        for i in range(n):
            v+=(v1[i]*v2[i])
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
    n=a.shape[0]
    b=np.zeros(n, dtype=np.float64)
    #printqnv("a in qnv2flt",a)  # for test
    for i in range(n):
        ai=a[i]
        b[i]=(ai.n[0]+ai.n[1]*np.sqrt(N))/ai.n[2]
    scly=crs.scly
    isys=crs.isys
    if isys==3:
        b[1]=b[1]*scly
        if n>3:
            b[3]=b[3]*scly
    return b

def intv2qnv(a:np.ndarray):
    n=a.shape[0]
    qn0=qnn.Qnnum([0,0,1]) # qnnum zero
    b=Qnvec(np.full(n,qn0)) #qnnum zero vector
    for i in range(n):
        b[i]=qnn.int2qnn(a[i])
    return b

