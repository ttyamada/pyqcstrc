import sys
import numpy as np
import cython

import crsys
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnmath as qmt
import qnndarray as qna

class Prjop(qnm.Qnmat):
    def __init__(self):
        if(isys==2): # projection operator for icosahedral
            self=Qnprj_Icos()
        elif(isys==3): # projection operator for decagonal
            self=Qnprj_Deca()
        elif(isys==4): # projection operator for octagonal
            self=Qnprj_Octa()
        elif(isys==5): # projection operator dodecagonal
            self=Qnprj_Dode()

class Qnprj_Octa(qnm.Qnmat):
    def __new__(cls) : 
        shape=(n,n)
        return super().__new__(cls,shape)
 
    def __init__(self):
        M0=qnn.Qnnum([ 0, 0, 1]) #  0
        M1=qnn.Qnnum([ 1, 0, 1]) #  1
        M2=qnn.Qnnum([-1, 0, 1]) # -1
        M3=qnn.Qnnum([ 0, 1, 2]) #  sqrt(2)/2
        M4=qnn.Qnnum([ 0,-1, 2]) # -sqrt(2)/2
        #self=qnm.Qnmat(n,N)
        #mt=[\
        prj0=np.array([\
           [M1,M0,M1,M0,M0],\
           [M3,M3,M4,M3,M0],\
           [M0,M1,M0,M2,M0],\
           [M4,M3,M3,M3,M0],\
           [M0,M0,M0,M0,M1]\
        ],dtype=qnn.Qnnum)
        #]
        #for i in range(n):
        #    for j in range(n):
        #        prj0[i][j]=mt[i][j]
        qnm.printqnm("Qnprj_Octa prj",prj0) # for 
        self.prj0=prj0
        self.prji=qmt.qnmatinv(prj0,n) # get inversion matrix of prj
        self.n=n
        self.N=N
        self.shape=(n,n)
        #prj=self

# for decagonal QCs
#class Qnprj_Deca(np.ndarray):
#class Qnprj_Deca(qna.QnNdarray):
class Qnprj_Deca(qnm.Qnmat):
    def __new__(cls):
        shape=(n,n)
        return super().__new__(cls,shape)
 
    # note that this use orthorhombic coordinate system
    def __init__(self):
        qn2=qnn.Qnnum([2,0,1])   #  2
        M0=qnn.Qnnum([ 0, 0, 1]) #  0
        M1=qnn.Qnnum([ 1, 0, 1]) #  1
        M2=qnn.Qnnum([-1, 0, 1]) # -1
        M3=qnn.Qnnum([1,1,2])    # tau
        M4=qnn.Qnnum([-1,1,2])   # tau^-1
        M5=M4*M4 # tau^-2
        M6=M4-qn2
        M7=-M3-qn2

        #mt=[\
        prj0=np.array([\
           [M6,M4,M7,M5,M0],\
           [M7,M5,M6,-M4,M0],\
           [M7,-M5,M6,M4,M0],\
           [M6,-M4,M7,-M5,M0],\
           [M0,M0,M0,M0,M1]\
        ],dtype=qnn.Qnnum)
        #]
        #for i in range(n):
        #    for j in range(n):
        #        prj0[i][j]=mt[i][j]
        qnm.printqnm("Qnprj_Deca prj",prj0) # for test
        self.prj0=prj0
        self.prji=qmt.qnmatinv(prj0,n) # get inversion matrix of prj
        self.n=n
        self.N=N
        self.shape=(n,n)
        #prj=self
        #print("self.ndim",self.ndim) # fpr test
        #print("self.shape",self.shape) # fpr test

# for dodecagonal QCs
#class Qnprj_Dode(np.ndarray):
#class Qnprj_Dode(qna.QnNdarray):
class Qnprj_Dode(qnm.Qnmat):
    def __new__(cls):
        shape=(n,n)
        return super().__new__(cls,shape)

    def __init__(self):
        M0=qnn.Qnnum([ 0, 0, 1]) #0
        M1=qnn.Qnnum([ 1, 0, 1]) # 1
        M2=qnn.Qnnum([-1, 0, 1]) #-1
        M3=qnn.Qnnum([ 1, 0, 2]) # 1/2
        M4=qnn.Qnnum([-1, 0, 2]) # 1/2
        M5=qnn.Qnnum([ 0, 1, 2]) #  sqrt(3)/2
        M6=qnn.Qnnum([ 0,-1, 2]) # -sqrt(3)/2

        #mt=[\
        prj0=np.array([\
           [M5,M4,M6,M4,M0],\
           [M1,M0,M1,M0,M0],\
           [M0,M1,M0,M1,M0],\
           [M4,M5,M4,M6,M0],\
           [M0,M0,M0,M0,M1]\
        ],dtype=qnn.Qnnum)
        #]

        #for i in range(n):
        #    for j in range(n):
        #        prj0[i][j]=mt[i][j]
        qnm.printqnm("Qnprj_Dode prj",prj0) # for test
        self.prj0=prj0
        self.prji=qmt.qnmatinv(prj0,n) # get inversion matrix of prj
        self.n=n
        self.N=N
        self.shape=(n,n)
        #prj=self
        #print("self.ndim",self.ndim) # fpr test
        #print("self.shape",self.shape) # fpr test

# for icosahedral QCs
#class Qnprj_Icos(npndarray):
#class Qnprj_Icos(qna.QnNdarray):
class Qnprj_Icos(qnm.Qnmat):
    def __new__(cls):
        shape=(n,n)
        return super().__new__(cls,shape)
    
    def __init__(self):
        M0=qnn.Qnnum([ 0, 0, 1]) #  0 
        M1=qnn.Qnnum([ 1, 0, 1]) #  1
        M2=qnn.Qnnum([-1, 0, 1]) # -1
        M3=qnn.Qnnum([ 1, 1, 2]) #  tau=(1+sqrt(5))/2
        M4=qnn.Qnnum([-1,-1, 2]) # -tau

        #mt=[\
        prj0=np.array([\
           [M1,M3,M0,M3,M2,M0],\
           [M3,M0,M1,M2,M0,M3],\
           [M3,M0,M2,M2,M0,M4],\
           [M0,M1,M4,M0,M3,M1],\
           [M2,M3,M0,M4,M2,M0],\
           [M0,M1,M3,M0,M3,M2]\
        ],dtype=qnn.Qnnum)
        #]
        
        #for i in range(n):
        #    for j in range(n):
        #        prj0[i][j]=mt[i][j]
        qnm.printqnm("Qnprj_Icos prj0",prj0) # for test
        self.prj0=prj0
        self.prji=qmt.qnmatinv(prj0,n) # get inversion matrix of prj
        self.n=n
        self.N=N
        self.shape=(n,n)
        #print("self.ndim",self.ndim) # fpr test
        #print("self.shape",self.shape) # fpr test
        
def prjop_init():
    global isys,n,N,prj
    isys=crsys.isys
    n=crsys.n
    N=crsys.N    
    prj=Prjop()
    
def copy(qna1: qnm.Qnmat):
    #return np.copy(qna1,dtype=qnn.Qnnum)
    return qnm.copy(qna1)
    #shape=qna1.shape
    #N=qna1[0][0].N
    #qna2=qnm.zerom(shape,N)
    #for i in range(shape[0]):
    #    for j in range(shape[1]):
    #        qna2[i][j]=qnn.copy(qna1[i][j])
    #return qna2
            
# for class cls cls should be icos octa, deca or dode
def prjop(v: qnv.Qnvec) -> qnv.Qnvec:
    #qnm.printqnm("prj",prj0)
    #qnv.printqnv("v",v)
    vei=v@prj0  #@v # vt assumed to be qnvec
    return vei

# projection into external space for class cls
def prjop_e(v:qnv.Qnvec) -> qnv.Qnvec:
    vei=v@prj0  #@v # vt assumed to be qnvec
    ve=qnv.zerovs(3)
    if isys>2: # dihedral
        ve[0]=vei[0]; ve[1]=vei[1]; ve[2]=vei[4]
        return ve
    elif isys==2: # icosahedral
        ve[0]=vei[0]; ve[1]=vei[1]; ve[2]=vei[2]
        return ve

# projection into internal space for class cls
def prjop_i(v: qnv.Qnvec) -> qnv.Qnvec:
    vei=v@prj0  #@v
    if isys>2: # dihedral
        vi=qnv.zerovs(2)
        vi[0]=vei[2]; vi[1]=vei[3]
        return vi
    elif isys==2: # icosahedral
        vi=qnv.zerovs(3)
        vi[0]=vei[3]; vi[1]=vei[4]; vi[2]=vei[5]
        return vi

# alias for prjop_i
def projection3(v: qnv.Qnvec) -> qnv.Qnvec:
    return prjop_i(v)

# alias for prjop
def projection_numerical(vn: qnv.Qnvec) -> qnv.Qnvec:
    return prjop(vn)

def projection_sets_numerical(vns: qnv.Qnvec) -> qnv.Qnvec:
    #Parameters
    #vsn: array
    #    set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    num=len(vns)
    m=qnv.zeros((1),dtype=qnv.Qnvec)
    for i in range(num):
        m[i]=projection_numerical(vns[i])
    return m

# alias for prjop_e
def projection_numerical_par(v: qnv.Qnvec) -> qnv.Qnvec:
    return prjop_e(v)

# alias for prjop_i
def get_internal_component_numerical(v: qnv.Qnvec) -> qnv.Qnvec:
    return prjop_i(v)

def check_ltv(n,N):
    # check lattice vector external and internal space components
    ndv=np.ndarray(3**3,dtype=qnn.Qnnum)
    n=5
    V0=qnn.Qnnum(np.array([ 0, 0, 1]),N)
    for i1 in range(-1,2):
        for i2 in range(-1,2):
            for i3 in range(-1,2):
                for i4 in range(-1,2):
                    V1=qnn.Qnnum(np.array([ i1, 0, 1]),N)
                    V2=qnn.Qnnum(np.array([ i2, 0, 1]),N)
                    V3=qnn.Qnnum(np.array([ i3, 0, 1]),N)
                    V4=qnn.Qnnum(np.array([ i4, 0, 1]),N)
                    #V0=qnn.Qnnum(np.array([ 0, 0, 1]),N)
                    #VT=np.array([v1,v2,v3,V4,V0])
                    vt=qnv.anyv(n,N,[V1,V2,V3,V4,V0])
                    qnn.printqnv("ndv",vt) #print qnvector expression
                    qnv=vt@prj0  #@ndv #external enternal components
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
            print(prj3f[i][j],end=" ")
        print()
    print()
        
def qnm2flnm(prj:qnm.Qnmat):
    n=prj.shape[0]
    prjf=np.ndarray((n,n),dtype=float)
    for i in range(n):
        for j in range(n):
            a=prj[i][j]
            #print("a",a.n[0],a.n[1],a.n[2])  # for test
            prjf[i][j]=qnn.qn2flt(a)
            #print("f",prjf[i][j])  # for test
    return prjf

