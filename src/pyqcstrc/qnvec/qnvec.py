import sys
import numpy as np
from numpy.typing import NDArray
from pyqcstrc.qnnum import qnnum
#from pyqcstrc.qnvec import qnvec
    
class Qnvec:
    #def __init__(self, vt:np.ndarray[qnnum], shape:np.array):
    def __init__(self, vt:np.ndarray[qnnum]):
        self.vt=vt
        self.shape = vt.shape  #dimension of a vector a
        self.ndim = vt.ndim
    def __add__(a, b):
        return add(a,b)
    
    def __sub__(a, b):
        return sub(a,b)

def add(v1:Qnvec, v2:Qnvec) -> Qnvec:
    a=np.empty(v1.shape, dtype=qnnum.Qnnum)
    la=v1.shape
    for i in range(la[0]):
        a[i]=v1.vt[i]+v2.vt[i]  #add(v1[i],v2[i])
    return Qnvec(a)

def sub(v1:Qnvec, v2:Qnvec)-> Qnvec:
    a=np.empty(v1.shape, dtype=qnnum.Qnnum)
    la=v1.shape
    for i in range(la[0]):
        a[i]=v1.vt[i]-v2.vt[i]  #add(v1[i],v2[i])
    return Qnvec(a)

def mul_vector(v:Qnvec, coeff):
    if v.ndim==2:
        a=np.zeros(v.shape,dtype=np.int64)
        la=v.shape
        for i,v in range(la[0]):
            a[i]=v*coeff  #mul(v,coeff)
        return a
    else:
        print('incorrect shape')
        return

def mul_vectors(vs, coeff):
    if vs.ndim==3:
        a=np.zeros(vs.shape,dtype=np.int64)
        la=v.shape
        for i,v in range(la[0]):
            a[i]=mul_vector(v,coeff)
        return a
    elif vs.ndim==4:
        a=np.zeros(vs.shape,dtype=np.int64)
        for i1,v in enumerate(vs):
            for i2,v in enumerate(v):
                a[i1][i2]=mul_vector(v,coeff)
    else:
        print('incorrect shape')
        return

def shift_vectors(vs, v):
    if vs.ndim==3:
        a=np.zeros(vs.shape,dtype=np.int64)
        la=vs.shape
        for i,v1 in range(la[0]):
            a[i]=add_vectors(v1,v)
        return a
    elif vs.ndim==4:
        a=np.zeros(vs.shape,dtype=np.int64)
        for i1,v1 in enumerate(vs):
            for i2,v2 in enumerate(v1):
                a[i1][i2]=add_vectors(v2,v)
    else:
        print('incorrect shape')
        return

def outer_product(v1, v2):
    a=v1[1]*v2[2] #mul(v1[1],v2[2])
    b=v1[2]*v2[1] #mul(v1[2],v2[1])
    c1=a-b          #sub(a,b)
    #
    a=v1[2]*v2[0] #mul(v1[2],v2[0])
    b=v1[0]*v2[2] #mul(v1[0],v2[2])
    c2=a-b          #sub(a,b)
    #
    a=v1[0]*v2[1] #mul(v1[0],v2[1])
    b=v1[1]*v2[0] #mul(v1[1],v2[0])
    c3=a-b          #sub(a,b)
    #
    return np.array([c1,c2,c3],dtype=np.int64)

def inner_product(v1, v2):
    s1,_=v1.shape
    s2,_=v2.shape
    if s1!=s2:
        print('matrices have not a proper shape.')
        return 
    else:
        a=np.array([0,0,1])
        for i in range(s1):
            b=v1[i]*v2[i]  #mul(v1[i],v2[i])
            a=a+b            #add(a,b)
        return a

def dot_product(vec1, vec2):
    ndim1=vec1.ndim
    ndim2=vec2.ndim
    
    if ndim1==2 and ndim2==2:
        return inner_product(vec1,vec2)
    elif ndim1==3 and ndim2==2:
        s,t1,_=vec1.shape
        t2,_=vec2.shape
        if t1!=t2:
            print('incorrect shape found in dot_product')
            return 	            
    elif ndim1==3 and ndim2==3:
        s,t1,_=vec1.shape
        t2,u,_=vec2.shape
        if t1!=t2:
            print('incorrect shape found in dot_product')
            return 
    else:
        print('incorrect shape found in dot_product')
        return 

def dot_product_1(vec1, vec2):
    ndim1=vec1.ndim
    ndim2=vec2.ndim
    
    if ndim1==2 and ndim2==2:
        s,t1,=vec1.shape
        t2,_=vec2.shape
        if t1!=t2:
            print('incorrect shape found in dot_product')
            return 
    elif ndim1==2 and ndim2==3:
        s,t1,=vec1.shape
        t2,u,_=vec2.shape
        if t1!=t2:
            print('incorrect shape found in dot_product')
            return 
    else:
        print('incorrect shape found in dot_product')
        return 

def qnv2npa(a):
    # Qnvector to np.array converter
    la=a.shape
    #print("la",la)
    print("la",la,"la[0]",la[0],"range(la[0])",range(la[0]))
    
    b=np.empty((la[0],3),dtype=np.int64)
    print("b",b)
    for i in range(la[0]):
        ai=a.vt[i]
        b[i]=[ai.n[0],ai.n[1],ai.n[2]]
    return b
    
def qnv2flt(a):
    la=a.shape
    b=np.zeros(la, dtype=np.float64)
    #print("b",b)
    N=a.vt[0].N
    for i in range(la[0]):
        ai=a.vt[i]
        b[i]=(ai.n[0]+ai.n[1]*np.sqrt(N))/ai.n[2]
    return b

def printqnv(str,qnv1):
    print(str,"[",end=" ")
    la=qnv1.shape
    for i in range(la[0]):
        j=qnv1.vt[i]
        print(qnnum.qn2npa(j),end=" ")
    print("]")

def printqnv2(str,qnv1,qnv2):
    print(str,"[",end=" ")
    la1=qnv1.shape
    for i in range(la[0]):
        j=qnv1.vt[i]
        print(qnnum.qn2npa(j),end=" ")
    print("] [",end="")
    la2=qnv2.shape
    for i in qnv2:
        j=qnv1.vt[i]
        print(qnnum.qn2npa(j),end="]")
    