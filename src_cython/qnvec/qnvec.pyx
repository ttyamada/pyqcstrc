cimport sys
cimport numpy as np
from numpy.typing cimport NDArray
cimport qnnum.qnnum as qnn
#from pyqcstrc.qnvec cimport qnvec

class Qnvec:
    cdef __init__(self, n:np.int64, N:np.int64):
        print("n",n)
        self.vt=np.ndarray(dtype=qnn.Qnnum,shape=(n)) # 1D array
        self.shape=n
        self.ndim=1
        self.N
        qnzero=qnn.int2qnn(0,N)
        for i in range(n):
            self.vt[i]=qnzero
        print("self.shape",self.shape)
        print("self.ndim",self.ndim)

    cdef __add__(a, b):
        return add(a,b)
    
    cdef __sub__(a, b):
        return sub(a,b)
    
cdef zerov(qnzero:qnn.Qnnum,n:np.int64): # qnnumber zero vector
    qnv=Qnvec(np.full(n,qnzero))
    return qnv

cdef add(v1:Qnvec, v2:Qnvec) -> Qnvec:
    a=np.empty(v1.shape, dtype=qnn.Qnnum)
    la=v1.shape
    for i in range(la[0]):
        a[i]=v1.vt[i]+v2.vt[i]  #add(v1[i],v2[i])
    return Qnvec(a)

cdef sub(v1:Qnvec, v2:Qnvec)-> Qnvec:
    a=np.empty(v1.shape, dtype=qnn.Qnnum)
    la=v1.shape
    for i in range(la[0]):
        a[i]=v1.vt[i]-v2.vt[i]  #add(v1[i],v2[i])
    return Qnvec(a)

cdef mul_vector(v:Qnvec, coeff:qnn.Qnnum):
    if v.ndim==2:
        a=np.zeros(v.shape,dtype=np.int64)
        la=v.shape
        for i in range(la[0]):
            a[i]=v.vt[i]*coeff  #mul(v,coeff)
        return a
    else:
        print('incorrect shape')
        return

cdef mul_vectors(vs:Qnvec, coeff:qnn.Qnnum):
    if vs.ndim==3:
        a=np.zeros(vs.shape,dtype=np.int64)
        la=v.shape
        for i in range(la[0]):
            a[i]=v.vt[i]*coeff #mul_vector(v,coeff)
        return a
#    elif vs.ndim==4:
#        a=np.zeros(vs.shape,dtype=np.int64)
#        la=vs.shape
#        for i1 in range(la[0]):
#            for i2 in range(la[1]):
#                a[i1][i2]=v[i1][i2]*coeff  #mul_vector(v,coeff)
    else:
        print('incorrect shape')
        return

cdef shift_vectors(vs:Qnvec, v:Qnvec) -> Qnvec:
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
    
# cros == outer_product (cross product) 
cdef cros(v1:Qnvec, v2:Qnvec) -> Qnvec:
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
    return Qnvec([c1,c2,c3])

cdef outer_product(v1:Qnvec, v2:Qnvec)-> Qnvec:
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
    #return np.array([c1,c2,c3],dtype=qnn.Qnnumber)
    return Qnvec([c1,c2,c3])

cdef inner_product(v1:Qnvec, v2:Qnvec) -> qnn.Qnnum:
    s1,_=v1.shape
    s2,_=v2.shape
    if s1!=s2:
        print('matrices have not a proper shape.')
        return 
    else:
        a=np.array([0,0,1])
        for i in range(s1):
            b=v1[i]*v2[i]  #mul(v1[i],v2[i])
            a=a+b          #add(a,b)
        return a

cdef dot_product(vec1:Qnvec, vec2:Qnvec):
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

cdef dot_product_1(vec1:Qnvec, vec2:Qnvec):
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

cdef qnv2npa(a:Qnvec):
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
    
cdef qnv2flt(a:Qnvec):
    la=a.shape
    b=np.zeros(la, dtype=np.float64)
    #print("b",b)
    N=a.vt[0].N
    for i in range(la[0]):
        ai=a.vt[i]
        b[i]=(ai.n[0]+ai.n[1]*np.sqrt(N))/ai.n[2]
    return b

cdef intv2qnv(a:np.ndarray,N:np.int64):
    n=a.shape[0]
    qn0=qnn.Qnnum([0,0,1],N) # qnnum zero
    b=Qnvec(np.full(n,qn0)) #qnnum zero vector
    for i in range(n):
        b.vt[i]=qnn.int2qnn(a[i],N)
    return b

cdef printqnv(str:str,qnv1:Qnvec):
    print(str,"[",end=" ")
    la=qnv1.shape
    for i in range(la):
        j=qnv1.vt[i]
        print(qnn.qn2npa(j),end=" ")
    print("]")

cdef printqnv2(str:str,qnv1:Qnvec,qnv2:Qnvec):
    print(str,"[",end=" ")
    la1=qnv1.shape
    for i in range(la1):
        j=qnv1.vt[i]
        print(qnn.qn2npa(j),end=" ")
    print("] [",end="")
    la2=qnv2.shape
    for i in qnv2:
        j=qnv1.vt[i]
        print(qnn.qn2npa(j),end="]")
    
    
if __name__ == '__main__':
    # test
    N=np.int64(2)
    n=np.int64(6)
    print("n=",n)
    qnv=Qnvec(n,N)  # nD qnnum zero vector
    print("qnm.shape",qnv.shape)
    printqnv("Qnvec_Octa",qnv)
    