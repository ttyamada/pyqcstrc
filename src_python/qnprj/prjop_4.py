import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnmath as qmt
import qnndarray as qna 

# alias for prjop_i
def projection3(v: qnv.Qnvec) -> qnv.Qnvec:
    return prjvec_i(v)

# alias for prjop
def projection_numerical(vn: qnv.Qnvec) -> qnv.Qnvec:
    return prjvec(vn)

def projection3_sets_numerical(vns: qna.QnNdarray) -> qna.QnNdarray:
    #Parameters
    #vsn: qnvector array
    #    set of nD vectors shape (num,n) assumed or
    #    (num,3,n) or (num,4,n) for triangles or tetrahedra 
    shape=vns.shape
    ndim=len(shape)
    #print("shape",shape,"ndim",ndim)  # for test
    isys=crs.isys
    if isys>2:
        ni=2
    else:
        ni=3
    if ndim==1:
        m=qna.zeros((ni,))
        m=projection3(vns)
    elif ndim==2:
        m=qna.zeros((shape[0],ni))
        for i in range(shape[0]):
            m[i]=projection3(vns[i])
    elif ndim==3:
        m=qna.zeros((shape[0],shape[1],ni))
        for i in range(shape[0]):
            for j in range(shape[1]):
                m[i][j]=projection3(vns[i][j])
    return m

# alias for prjop_e
def projection_numerical_par(v: qnv.Qnvec) -> qnv.Qnvec:
    return prjvec_e(v)

# alias for prjop_i
def get_internal_component_numerical(v: qnv.Qnvec) -> qnv.Qnvec:
    return prjvec_i(v)

def check_ltv(n,N):
    # check lattice vector external and internal space components
    ndv=np.ndarray(3**3,dtype=qnn.Qnnum)
    n=5
    V0=qnn.any(([ 0, 0, 1]))
    for i1 in range(-1,2):
        for i2 in range(-1,2):
            for i3 in range(-1,2):
                for i4 in range(-1,2):
                    V1=qnn.any([ i1, 0, 1])
                    V2=qnn.any([ i2, 0, 1])
                    V3=qnn.any([ i3, 0, 1])
                    V4=qnn.any([ i4, 0, 1])
                    #V0=qnn.any(np.array([ 0, 0, 1]),N)
                    #VT=np.array([v1,v2,v3,V4,V0])
                    vt=qnv.anyv(n,N,[V1,V2,V3,V4,V0])
                    qnn.printqnv("ndv",vt) #print qnvector expression
                    qnv=prj0@vt  #qnv=vt@prj0  #@ndv #external enternal components
                    qnn.printqnv("qnv",qnv) #print qnvector expression
                    n+=1
                    
def print_prj(str:str,prj:qnm.Qnmat):
    #print(prj)
    print(str)
    n=prj.shape[0]
    N=prj.N
    print("n",n,"N",N)
    
    qnm.printqnm(str,prj)
    
    b=qnm.copy(prj)
    print(str)
    qnm.printqnm("str",b)
    
def printfm(str,prj3f,n):
    print(str)
    for i in range(n):
        for j in range(n):
            #print(prj3f[i][j],end=" ")
            print("{:8f}".format(prj3f[i][j]),end=" ")
        print()
    print()
        
def qnm2flnm(prj:qnm.Qnmat):
    n=prj.shape[0]
    prj0f=np.ndarray((n,n),dtype=float)
    for i in range(n):
        for j in range(n):
            a=prj[i][j]
            #print("a",a.n[0],a.n[1],a.n[2])  # for test
            prj0f[i][j]=qnn.qn2flt(a)*scl
            #print("f",prj0f[i][j])  # for test
    return prj0f
