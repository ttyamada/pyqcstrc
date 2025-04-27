import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnmath as qmt
import qnndarray as qna

class Qnprj_Octa(qnm.Qnmat):
    def __new__(cls) : 
        global n
        n=crs.n
        #print("n in __new__",n) # for test
        shape=(n,n)
        return super().__new__(cls,shape)
 
    def __init__(self):
        M0=qnn.any([ 0, 0, 1]) #  0
        M1=qnn.any([ 1, 0, 1]) #  1
        M2=qnn.any([-1, 0, 1]) # -1
        M3=qnn.any([ 0, 1, 2]) #  sqrt(2)/2 t1
        M4=qnn.any([ 0,-1, 2]) # -sqrt(2)/2 t2=-t1
        #self=qnm.Qnmat(n,N)
        
        prj0=np.array([\
           [M1,M0,M1,M0,M0],\
           [M3,M3,M4,M3,M0],\
           [M0,M1,M0,M2,M0],\
           [M4,M3,M3,M3,M0],\
           [M0,M0,M0,M0,M1]\
        ],dtype=qnn.Qnnum)
 
        #qnm.printqnm("Qnprj_Octa prj",prj0) # for 
        self.prj0=prj0
        self.prji=qmt.qnmatinv(prj0,n) # get inversion matrix of prj
        self.n=n
        self.N=crs.N
        self.shape=(n,n)
        self.scl=1.0
        self.scly=1.0
        #prj=self
        prj0f=qnm.qnm2flt(prj0)  # for float number
        #qnm.printfm("prj0f",prj0f)  # for test


class Qnprj_Deca(qnm.Qnmat):
    def __new__(cls):
        global n
        n=crs.n
        shape=(n,n)
        return super().__new__(cls,shape)
 
    # note that this use orthorhombic coordinate system
    def __init__(self):
        M0=qnn.any([ 0,0,1]  ) #  0
        M1=qnn.any([1,1,2])    # tau
        M2=qnn.any([-1,1,2])   # tau^-1
        M3=qnn.any([1,0,1])    # 1
        M5=M2/2     #    tau^-1/2=c1
        M6=M1/(-2)  #   -tau/2=c2
        M7=M2/2     #1   s2/(2sin(pi/5))
        M8=M7*(-1)  #-1 -s2/(2sin(pi/5))
        M9=M3/2     #    s1/(2sin(pi/5))
        M10=M9*(-1) #   -s1/(2sin(pi/5))
        # y axis in external and internal spaces should be scaled by 2sin(pi/5)        

        
        prj0=np.array([\
           [M5,M9,M6,M7,M0],\
           [M6,M7,M5,M10,M0],\
           [M6,M8,M5,M9,M0],\
           [M5,M10,M6,M8,M0],\
           [M0,M0,M0,M0,M3]\
        ],dtype=qnn.Qnnum)
        
        #qnm.printqnm("Qnprj_Deca prj",prj0) # for test
        self.prj0=prj0
        self.prji=qmt.qnmatinv(prj0,n) # get inversion matrix of prj
        self.n=n
        self.N=crs.N
        self.shape=(n,n)
        self.scl=2.0/np.sqrt(5.0)
        self.scly=2.0*np.sin(np.pi/5) #2s1
        self.scly2=(M3-M5**2)*4       #4(1-c1^2)
        #self.scly2=4*(M3-M5**2)       #4(1-c1^2)
        #prj=self
        #print("self.ndim",self.ndim) # for test
        #print("self.shape",self.shape) # for test
        prj0f=qnm.qnm2flt(prj0)
        #qnm.printfm("prj0f",prj0f)  # for test


class Qnprj_Dode(qnm.Qnmat):
    def __new__(cls):
        global n
        n=crs.n
        shape=(n,n)
        return super().__new__(cls,shape)

    def __init__(self):
        M0=qnn.any([ 0, 0, 1]) #0
        M1=qnn.any([ 1, 0, 1]) # 1
        M2=qnn.any([-1, 0, 1]) #-1
        M3=qnn.any([ 1, 0, 2]) # 1/2
        M4=qnn.any([-1, 0, 2]) # 1/2
        M5=qnn.any([ 0, 1, 2]) #  sqrt(3)/2
        M6=qnn.any([ 0,-1, 2]) # -sqrt(3)/2

        
        prj0=np.array([\
           [M1,M0,M1,M0,M0],\
           [M5,M3,M2,M3,M0],\
           [M3,M5,M3,M6,M0],\
           [M0,M1,M0,M1,M0],\
           [M0,M0,M0,M0,M1]\
        ],dtype=qnn.Qnnum)
        
        #qnm.printqnm("Qnprj_Dode prj",prj0) # for test
        self.prj0=prj0
        self.prji=qmt.qnmatinv(prj0,n) # get inversion matrix of prj
        self.n=n
        self.N=crs.N
        self.shape=(n,n)
        self.scl=2.0/np.sqrt(6.0)  # for vesta or qnn2flt
        self.scly=1.0
        #prj=self
        #print("self.ndim",self.ndim) # fpr test
        #print("self.shape",self.shape) # fpr test
        prj0f=qnm.qnm2flt(prj0)
        #qnm.printfm("prj0f",prj0f)  # for test

# for icosahedral QCs
#class Qnprj_Icos(npndarray):
#class Qnprj_Icos(qna.QnNdarray):
class Qnprj_Icos(qnm.Qnmat):
    def __new__(cls):
        global n
        n=crs.n
        shape=(n,n)
        return super().__new__(cls,shape)
    
    def __init__(self):
        M0=qnn.any([ 0, 0, 1]) #  0 
        M1=qnn.any([ 1, 0, 1]) #  1
        M2=qnn.any([-1, 0, 1]) # -1
        M3=qnn.any([ 1, 1, 2]) #  tau=(1+sqrt(5))/2
        M4=qnn.any([-1,-1, 2]) # -tau

        
        prj0=np.array([\
           [M1,M3,M0,M3,M2,M0],\
           [M3,M0,M1,M2,M0,M3],\
           [M3,M0,M2,M2,M0,M4],\
           [M0,M1,M4,M0,M3,M1],\
           [M2,M3,M0,M4,M2,M0],\
           [M0,M1,M3,M0,M3,M2]\
        ],dtype=qnn.Qnnum)
        
        
        #for i in range(n):
        #    for j in range(n):
        #        prj0[i][j]=mt[i][j]
        #qnm.printqnm("Qnprj_Icos prj0",prj0) # for test
        self.prj0=prj0
        self.prji=qmt.qnmatinv(prj0,n) # get inversion matrix of prj
        self.n=n
        self.N=crs.N
        self.shape=(n,n)
        tau=(1.0+np.sqrt(5.0))/2.0
        self.scl=1.0/np.sqrt(2.0+tau)
        self.scly=1.0
        #print("self.ndim",self.ndim) # fpr test
        #print("self.shape",self.shape) # fpr test
        prj0f=qnm.qnm2flt(prj0)
        #qnm.printfm("prj0f",prj0f)  # for test
        
def prjop_init():
    global isys,n,N
    isys=crs.isys
    n=crs.n
    N=crs.N
    prj=Prjop()

    
def Prjop():
    global prj0,prji,prj0t,prjit,scl,scly
    #print("isys in Prjop",isys)  # fpr test
    if(isys==2): # projection operator for icosahedral
        prj=Qnprj_Icos()
    elif(isys==3): # projection operator for decagonal
        prj=Qnprj_Deca()
    elif(isys==4): # projection operator for octagonal
        prj=Qnprj_Octa()
    elif(isys==5): # projection operator dodecagonal
        prj=Qnprj_Dode()
    prj0=prj.prj0
    prji=prj.prji
    prj0t=qmt.matrixtr(prj0) # transposed prj matrix
    prjit=qmt.matrixtr(prji) # transposed prji matrix 
    scl=prj.scl
    scly=prj.scly
    return prj

def tstwt_prjop():
    qnm.printqnm("prj0",prj0)
    qnm.printqnm("prji",prji)
    qnm.printqnm("prj0t",prj0t)
    qnm.printqnm("prjit",prjit)
    unitm=qnm.zerom((n,n))
    unitm=prji@prj0
    qnm.printqnm("unitm",unitm)
    
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
def prjvec(v: qnv.Qnvec) -> qnv.Qnvec:
    #qnm.printqnm("prj",prj0)
    #qnv.printqnv("v",v)
    n=crs.n
    vei=qnv.zerov(n)
    vei[0:3]=prjvec_e(v)
    if isys>2:
        vei[3:5]=prjvec_i(v)
    else:
        vei[3:6]=prjvec_i(v)
    #qnv.printqnv("vei",vei)  # for test
    return vei

# projection into external space for class cls
def prjvec_e(v:qnv.Qnvec) -> qnv.Qnvec:
    #n=crs.n
    #vei=qnv.zerov(n)
    vei=v@prj0  # v assumed to be qnvec
    ve=qnv.zerov(3)
    if isys>2: # dihedral
        ve[0:2]=vei[0:2]
        ve[2]=vei[4]
    elif isys==2: # icosahedral
        ve[0:3]=vei[0:3]
    #qnv.printqnv("ve",ve)  # for test
    return ve

# projection into internal space for class cls
def prjvec_i(v: qnv.Qnvec) -> qnv.Qnvec:
    #n=crs.n
    #vei=qnv.zerov(n)
    #print("v.shape",v.shape)  # for test
    #qnm.printqnm("prj0",prj0)  # for test
    vei=v@prj0
    #qnv.printqnv("v",v)  # for test
    #qnv.printqnv("vei",vei)  # for test
    isys=crs.isys
    #print("isys in prjvec_i",isys)  # for test
    if isys>2: # dihedral
        ni=2
        vi=qnv.zerov(ni)
        vi=vei[2:4]
    elif isys==2: # icosahedral
        ni=3
        vi=qnv.zerov(ni)
        vi=vei[3:6]
    #print("vi.shape",vi.shape)  # for test
    #qnv.printqnv("vi",vi)  # for test
    return vi

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

