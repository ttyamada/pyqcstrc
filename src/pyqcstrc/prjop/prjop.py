import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnmat.qnmat as qnm
import pyqcstrc.qnmath.qnmath as mth
import pyqcstrc.qnndarray.qnndarray as qna

#N=2
#n=5
#prj=Qnprj_Octa(n,N) # projection operator for Qnvector (default)

# for octagonal QCs
#class Qnprj_Octa(np.ndarray):
#class Qnprj_Octa(qna.QnNdarray):
class Qnprj_Octa(qnm.Qnmat):
    def __new__(cls) : 
        global n,N;
        n=5
        N=2
        return super().__new__(cls,n,N)
        #shape=(n,n)
        #return super().__new__(cls,shape,N)

    def __init__(self):
        M0=qnn.Qnnum(np.array([ 0, 0, 1]),N) #  0
        M1=qnn.Qnnum(np.array([ 1, 0, 1]),N) #  1
        M2=qnn.Qnnum(np.array([-1, 0, 1]),N) # -1
        M3=qnn.Qnnum(np.array([ 0, 1, 2]),N) #  sqrt(2)/2
        M4=qnn.Qnnum(np.array([ 0,-1, 2]),N) # -sqrt(2)/2
        #self=qnm.Qnmat(n,N)
        #np.array([\
        #mt=[\
        mt=np.array([\
           [M1,M0,M1,M0,M0],\
           [M3,M3,M4,M3,M0],\
           [M0,M1,M0,M2,M0],\
           [M4,M3,M3,M3,M0],\
           [M0,M0,M0,M0,M1]\
        ],dtype=qnn.Qnnum)
        #]
        for i in range(n):
            for j in range(n):
                self[i][j]=mt[i][j]
        #self.set_mt(mt)
        #prj=copy(self)
 
        #print("self.ndim",self.ndim) # fpr test
        #print("self.shape",self.shape) # fpr test
        #print("self.shape[0]",self.shape[0]) # for test
        qnm.printqnm("Qnprj_Octa self",self) # for test

# for decagonal QCs
#class Qnprj_Deca(np.ndarray):
#class Qnprj_Deca(qna.QnNdarray):
class Qnprj_Deca(qnm.Qnmat):
    def __new__(cls):
        global n,N
        n=5
        N=5
        return super().__new__(cls,n,N)
        #shape=(n,n)
        #return super().__new__(cls,shape,N)

    # note that this use orthorhombic coordinate system
    def __init__(self):
        #n=5
        #N=5
        qn2=qnn.Qnnum([2,0,1],N)   #  2
        M0=qnn.Qnnum([ 0, 0, 1],N) #  0
        M1=qnn.Qnnum([ 1, 0, 1],N) #  1
        M2=qnn.Qnnum([-1, 0, 1],N) # -1
        M3=qnn.Qnnum([1,1,2],N)    # tau
        M4=qnn.Qnnum([-1,1,2],N)   # tau^-1
        M5=M4*M4 # tau^-2
        M6=M4-qn2
        M7=-M3-qn2
        #self=qnm.Qnmat(n,N)
        #mt=[\
        mt=np.array([\
           [M6,M4,M7,M5,M0],\
           [M7,M5,M6,-M4,M0],\
           [M7,-M5,M6,M4,M0],\
           [M6,-M4,M7,-M5,M0],\
           [M0,M0,M0,M0,M1]\
        ],dtype=qnn.Qnnum)
        #]
        for i in range(n):
            for j in range(n):
                self[i][j]=mt[i][j]
        #prj=qnm.copy(self)
        qnm.printqnm("Qnprj_Deca self",self) # for test
        #print("self.ndim",self.ndim) # fpr test
        #print("self.shape",self.shape) # fpr test

# for dodecagonal QCs
#class Qnprj_Dode(np.ndarray):
#class Qnprj_Dode(qna.QnNdarray):
class Qnprj_Dode(qnm.Qnmat):
    def __new__(cls):
        global n,N
        n=5
        N=3
        return super().__new__(cls,n,N)
        #shape=(n,n)
        #return super().__new__(cls,shape,N)

    def __init__(self):
        #n=5
        #N=3
        M0=qnn.Qnnum(np.array([ 0, 0, 1]),N) #0
        M1=qnn.Qnnum(np.array([ 1, 0, 1]),N) # 1
        M2=qnn.Qnnum(np.array([-1, 0, 1]),N) #-1
        M3=qnn.Qnnum(np.array([ 1, 0, 2]),N) # 1/2
        M4=qnn.Qnnum(np.array([-1, 0, 2]),N) # 1/2
        M5=qnn.Qnnum(np.array([ 0, 1, 2]),N) #  sqrt(3)/2
        M6=qnn.Qnnum(np.array([ 0,-1, 2]),N) # -sqrt(3)/2
        #M5=np.array([ 0, 1, 1]) # sqrt(3)
        #M6=np.array([ 0,-1, 1]) # -sqrt(3)
        #self=qnm.Qnmat(n,N)
        #mt=[\
        mt=np.array([\
           [M5,M2,M6,M5,M0],\
           [M1,M0,M1,M0,M0],\
           [M0,M1,M0,M1,M0],\
           [M4,M5,M4,M6,M0],\
           [M0,M0,M0,M0,M1]\
        ],dtype=qnn.Qnnum)
        #]
        #prj=qnm.copy(self)
        for i in range(n):
            for j in range(n):
                self[i][j]=mt[i][j]
        qnm.printqnm("Qnprj_Dode self",self) # for test
        #print("self.ndim",self.ndim) # fpr test
        #print("self.shape",self.shape) # fpr test

# for icosahedral QCs
#class Qnprj_Icos(npndarray):
#class Qnprj_Icos(qna.QnNdarray):
class Qnprj_Icos(qnm.Qnmat):
    def __new__(cls):
        global n,N
        n=6
        N=5
        return super().__new__(cls,n,N)
        #shape=(n,n)
        #return super().__new__(cls,shape,N)
    
    def __init__(self):
        #n=6
        #N=5
        M0=qnn.Qnnum([ 0, 0, 1],N) #  0 in 'TAU-style'
        M1=qnn.Qnnum([ 1, 0, 1],N) #  1
        M2=qnn.Qnnum([-1, 0, 1],N) # -1
        M3=qnn.Qnnum([ 1, 1, 2],N) #  tau=(1+sqrt(5))/2
        M4=qnn.Qnnum([-1,-1, 2],N) # -tau
        
        #self=qnm.Qnmat(n,N)
        #mt=[\
        mt=np.array([\
           [M1,M3,M3,M0,M2,M0],\
           [M3,M0,M0,M1,M3,M1],\
           [M0,M1,M2,M4,M0,M3],\
           [M3,M2,M2,M0,M4,M0],\
           [M2,M0,M0,M3,M2,M3],\
           [M0,M3,M4,M1,M0,M2]\
        ],dtype=qnn.Qnnum)
        #]
        
        for i in range(n):
            for j in range(n):
                self[i][j]=mt[i][j]
        #prj=copy(self)
 
        qnm.printqnm("Qnprj_Icos self",self) # for test
        #print("self.ndim",self.ndim) # fpr test
        #print("self.shape",self.shape) # fpr test
        

#class Prj_Octa:
#    def __init__(self):   
#        TAU=np.sqrt(2)
#        mt=np.array([\
#            [ TAU,  0.0,  TAU,  0.0,  0.0,  0.0],\
#            [ 1.0,  1.0, -1.0,  1.0,  0.0,  0.0],\
#            [ 0.0,  TAU,  0.0, -TAU,  0.0,  0.0],\
#            [-1.0,  1.0,  1.0,  1.0,  0.0,  0.0],\
#            [ 0.0,  0.0,  0.0,  0.0,  1.0,  0.0],\
#            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0],\
#            ])
#        self=mt
#        self.shape=mt.shape
#        self.ndim=mt.ndim
#        self.N=2

#class Prj_Deca:
#    def __init__(self):   
#        PI = np.pi
#        C=np.zeros(4,dtype=float)
#        S=np.zeros(4,dtype=float)
#        C0=np.zeros(4,dtype=float)
#        TAU=(1+np.sqrt(5))/2
#        #SCL=1/np.sqrt(2+TAU)
#        #SCL=1/np.sqrt(5)
#        #SCL=TAU
#        #SCL=1/np.cos(2*PI/5)
#        SCL=1
#        for i in range(4):
#            i1=i+1
#            C0[i] = np.cos(2*PI*i1/5)
#            C[i]=(C0[i]-1)*SCL
#            S[i] = np.sin(2*PI*i1/5)*SCL
#        print("C0",C0)
#        print("C",C)
#        print("S",S)
#        mt=np.array([
#            [ C[0], C[1], C[2], C[3], 0, 0],\
#            [ S[0], S[1], S[2], S[3], 0, 0],\
#            [ C[1], C[3], C[0], C[2], 0, 0],\
#            [ S[1], S[3], S[0], S[2], 0, 0],\
#            [    0,    0,    0,    0, 1, 0],\
#            [    0,    0,    0,    0, 0, 0],\
#            ])
#        self=mt
#        self.shape=mt.shape
#        self.ndim=mt.ndim
#        self.N=5
        
#class Prj_Dode:
#    def __init__(self):
#        mt=np.array([\
#            [ 0.5,          0.577350269,  0.0,         -0.288675135,  0.0,  0.0],\
#            [ 0.288675135,  0.5,          0.288675135,  0.0,          0.0,  0.0],\
#            [ 0.0,          0.288675135,  0.5,          0.288675135,  0.0,  0.0],\
#            [-0.288675135,  0.0,          0.577350269,  0.5,          0.0,  0.0],\
#            [ 0.0,          0.0,          0.0,          0.0,          1.0,  0.0],\
#            [ 0.0,          0.0,          0.0,          0.0,          0.0,  0.0],\
#            ])
#        self=mt
#        self.shape=mt.shape
#        self.ndim=mt.ndim
#        self.N=3
#        # take MT transpose
#        matrixtr(self) 


#class Prj_Icos:
#    #def projection_numerical_par(vn: NDArray[np.float64]) -> NDArray[np.float64]:
#    """This returns 6D vector which corresponds to a projection of vn onto Epar.
#    
#    Parameters
#    ----------
#    v: array
#        6-dimensional vector
#
#    Returns
#    -------
#    6d vectors projected onto Eperp.
#    """
#    def __init__(selfself):
#        mt=np.array([\
#           [0.5,  0.2236068,  0.2236068,  0.2236068,  0.2236068,  0.2236068],\
#           [ 0.2236068,  0.5      ,  0.2236068, -0.2236068, -0.2236068,  0.2236068],\
#           [ 0.2236068,  0.2236068,  0.5      ,  0.2236068, -0.2236068, -0.2236068],\
#           [ 0.2236068, -0.2236068,  0.2236068,  0.5      ,  0.2236068, -0.2236068],\
#           [ 0.2236068, -0.2236068, -0.2236068,  0.2236068,  0.5      ,  0.2236068],\
#           [ 0.2236068,  0.2236068, -0.2236068, -0.2236068,  0.2236068,  0.5      ]\
#           ])
#        self=mt
#        self.shape=mt.shape
#        self.ndim=mt.ndim
#        self.N=5

def prjop_init(isys:np.int64):
    if(isys==2): # projection operator for icosahedral
        prj=Qnprj_Icos()
    elif(isys==3): # projection operator for decagonal
        prj=Qnprj_Deca()
    elif(isys==4): # projection operator for octagonal
        prj=Qnprj_Octa()
    elif(isys==5): # projection operator dodecagonal
        prj=Qnprj_Dode()

# for class cls cls should be icos octa, deca or dode
def prjop(v: qnv.Qnvec) -> qnv.Qnvec:
    qnm.printqnm("prj",prj)
    qnv.printqnv("v",v)
    vei=v@prj  #@v # vt assumed to be qnvec
    return vei

# projection into external space for class cls
def prjop_e(v:qnv.Qnvec) -> qnv.Qnvec:
    vei=v@prj  #@v # vt assumed to be qnvec
    return vei[1:3]

# projection into internal space for class cls
def prjop_i(v: qnv.Qnvec) -> qnv.Qnvec:
    vei=v@prj  #@v
    return vei[4:6]

# alias for prjop_i
def projection3(v: qnv.Qnvec) -> qnv.Qnvec:
    return prjop_i(prj,v)

def projection_numerical(vn: qnv.Qnvec) -> qnv.Qnvec:
    return prjop(vn)

def projection_sets_numerical(vns: qnv.Qnvec) -> qnv.Qnvec:
    #Parameters
    #vsn: array
    #    set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    num=len(vns)
    for i in range(num):
        m[i]=projection_numerical(vns[i])
    return m

def projection_numerical_par(vn: qnv.Qnvec) -> qnv.Qnvec:
    return prjop(prj,vn)

# alias for prjop(vt)
def get_internal_component_numerical(vt: qnv.Qnvec) -> qnv.Qnvec:
    return prjop_i(vt)

def check_ltv(n,N):
    # check lattice vector external and internal space components
    ndv=np.ndarray(3**3,dtype=qnn.Qnnum)
    n=5
    for i1 in range(-1,2):
        for i2 in range(-1,2):
            for i3 in range(-1,2):
                for i4 in range(-1,2):
                    V1=qnn.Qnnum(np.array([ i1, 0, 1]),N)
                    V2=qnn.Qnnum(np.array([ i2, 0, 1]),N)
                    V3=qnn.Qnnum(np.array([ i3, 0, 1]),N)
                    V4=qnn.Qnnum(np.array([ i4, 0, 1]),N)
                    #V0=qnn.Qnnum(np.array([ 0, 0, 1]),N)
                    VT=np.array([v1,v2,v3,V4,V0])
                    ndv[n]=qnv.Qnvec(VT,n,N) # lattice vector
                    qnn.printqnv("ndv",ndv) #print qnvector expression
                    qnv=vt@mto  #@ndv #external enternal components
                    qnn.printqnv("qnv",qnv) #print qnvector expression
                    n+=1
                    
def print_prj(str:str,prj:qnm.Qnmat):
    #print(prj)
    print(str)
    n=prj.shape[0]
    N=prj[0][0].N
    print("n",n,"N",N)
    
    qnm.printqnm(str,prj)
    
    b=qnm.copy(prj)
    print(str)
    qnm.printqnm("str",b)

if __name__ == '__main__':
    # test for qnnum projection operators
    N=2
    prj4=Qnprj_Octa() # qnnum projection operator
    #print("prj4.shape",prj4.shape)
    #print("prj4.ndim",prj4.ndim)
    #print_prj("Octa",prj4)
    qnm.printqnm("prj4",prj4)  #
    qna.printqndm("prj4",prj4)  #
    
    N=5
    prj3=Qnprj_Deca() # float projection operator
    #print("prj3.shape",prj3.shape)
    #print("prj3.ndim",prj3.ndim)
    #print_prj("Deca",prj3)
    qnm.printqnm("prj3",prj3)  #
    qna.printqndm("prj3",prj3)  #
    
    N=3
    prj5=Qnprj_Dode() # float projection operator
    #print("prj5.shape",prj5.shape)
    #print("prj5.ndim",prj5.ndim)
    #print_prj("Dode",prj5)
    
    N=5
    prj2=Qnprj_Icos() # float projection operator
    #print("prj2.shape",prj2.shape)
    #print("prj2.ndim",prj2.ndim)
    #print_prj("Dode",prj5)
    qnm.printqnm("prj2",prj2)  #
    qna.printqndm("prj2",prj2)  #

    # check qnmatinv
    N=2
    n=5
    prj3=Qnprj_Octa()
    #print("prj3.shape",prj3.shape)
    #print("prj3.ndim",prj3.ndim)
    #qnn.printqnn("prj3[0][0].N",prj3[0][0].N)
    #qnm.printqnm("prj3",prj3)
    qnm.printqnm("prj3",prj3)  #
    qna.printqndm("prj3",prj3)  #
    
    prji3=Qnprj_Octa() # copy for matinv
    #print("prji3.shape",prji3.shape)
    #print("prji3.ndim",prji3.ndim)
    qnm.printqnm("prji3",prji3)  #
    qna.printqndm("prji3",prji3)  #
    mth.qnmatinv(prji3,n)
    qnm.printqnm("prji3",prji3)
    
    #check_ltv(n,N)

                    
                    
    
        
