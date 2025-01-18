import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnmat.qnmat as qnm
import pyqcstrc.qnmath.qnmath as mth

prj=Qnprjop_Octa() # projection operator for Qnvector

# for octagonal QCs
class Qnprj_Octa:
    def __init__(self):
        N=2
        M0=qnn.Qnnum(np.array([ 0, 0, 1]),N) #  0
        M1=qnn.Qnnum(np.array([ 1, 0, 1]),N) #  1
        M2=qnn.Qnnum(np.array([-1, 0, 1]),N) # -1
        M3=qnn.Qnnum(np.array([ 0, 1, 2]),N) #  sqrt(2)/2
        M4=qnn.Qnnum(np.array([ 0,-1, 2]),N) # -sqrt(2)/2
        MT=np.array([\
           [M1,M0,M1,M0,M0,M0],\
           [M3,M3,M4,M3,M0,M0],\
           [M0,M1,M0,M2,M0,M0],\
           [M4,M3,M3,M3,M0,M0],\
           [M0,M0,M0,M0,M1,M0],\
           [M0,M0,M0,M0,M0,M0]])
        n=6
        self.mt=qnm.Qnmat(MT,n,N) # matrix of qnnum (qnmat)
        self.shape = MT.shape  #dimension of a vector a
        self.n = n
        self.N=N

# for decagonal QCs
class Qnprj_Deca:
    # note that this use orthorhombic coordinate system
    def __init__(self):
        N=5
        qn2=qnn.Qnnum([2,0,1],N)   #  2
        M0=qnn.Qnnum([ 0, 0, 1],N) #  0
        M1=qnn.Qnnum([ 1, 0, 1],N) #  1
        M2=qnn.Qnnum([-1, 0, 1],N) # -1
        M3=qnn.Qnnum([1,1,2],N)    # tau
        M4=qnn.Qnnum([-1,1,2],N)   # tau^-1
        M5=M4*M4 # tau^-2
        M6=M4-qn2
        M7=-M3-qn2
        MT=np.array([\
           [M6,M4,M7,M5,M0,M0],\
           [M7,M5,M6,-M4,M0,M0],\
           [M7,-M5,M6,M4,M0,M0],\
           [M6,-M4,M7,-M5,M0,M0],\
           [M0,M0,M0,M0,M1,M0],\
           [M0,M0,M0,M0,M0,M0]])
        n=6
        self.mt=qnm.Qnmat(MT,n,N) # matrix of qnnum (qnmat)
        self.shape = MT.shape  #dimension of a vector a
        self.n = n
        self.N=N

# for dodecagonal QCs
class Qnprj_Dode:
    def __init__(self):
        N=3
        """projection of a 6d vector onto Epar and Eperp in "SIN-style"
        NOTE: coefficient (alpha) of the projection matrix is set to be 1.
        alpha = 2*a/np.sqrt(6)
        see Yamamoto ActaCrystal (1997)
        """
        M0=qnn.Qnnum(np.array([ 0, 0, 1]),N) #0
        M1=qnn.Qnnum(np.array([ 1, 0, 1]),N) # 1
        M2=qnn.Qnnum(np.array([-1, 0, 1]),N) #-1
        M3=qnn.Qnnum(np.array([ 1, 0, 2]),N) # 1/2
        M4=qnn.Qnnum(np.array([-1, 0, 2]),N) # 1/2
        M5=qnn.Qnnum(np.array([ 0, 1, 2]),N) #  sqrt(3)/2
        M6=qnn.Qnnum(np.array([ 0,-1, 2]),N) # -sqrt(3)/2
        #M5=np.array([ 0, 1, 1]) # sqrt(3)
        #M6=np.array([ 0,-1, 1]) # -sqrt(3)
        MT=np.array([\
           [M5,M2,M6,M5,M0,M0],\
           [M1,M0,M1,M0,M0,M0],\
           [M0,M1,M0,M1,M0,M0],\
           [M4,M5,M4,M6,M0,M0],\
           [M0,M0,M0,M0,M1,M0],\
           [M0,M0,M0,M0,M0,M0]])
        n=6
        self.mt=qnm.Qnmat(MT,n,N) # matrix of qnnum (qnmat)
        self.shape = MT.shape  #dimension of a vector a
        self.n = n
        self.N=N
        
class Qnprj_Icos:
    def __init__(self):
        N=5
        """projection of a 6d vector onto Epar and Eperp in "TAU-style"
        NOTE: coefficient (alpha) of the projection matrix is set to be 1.
        alpha = a/np.sqrt(2.0+TAU)
        see Yamamoto ActaCrystal (1997)

        Parameters
        ----------
        vt: array
            6-dimensional vector in TAU-style

        Returns
    -    ------
        array containing two 3d vectors projected onto Epar and Eperp in TAU-style.
        """
        m0=qnn.Qnnum([ 0, 0, 1]) #  0 in 'TAU-style'
        m1=qnn.Qnnum([ 1, 0, 1]) #  1
        m2=qnn.Qnnum([-1, 0, 1]) # -1
        m3=qnn.Qnnum([ 1, 1, 2]) #  tau
        m4=qnn.qnnum([-1,-1, 2]) # -tau
        MT=np.array([\
            [m1,m3,m3,m0,m2,m0],\
            [m3,m0,m0,m1,m3,m1],\
            [m0,m1,m2,m4,m0,m3],\
            [m3,m2,m2,m0,m4,m0],\
            [m2,m0,m0,m3,m2,m3],\
            [m0,m3,m4,m1,m0,m2]])
        n=6
        self.mt=qnm.Qnmat(MT,n,N) # matrix of qnnum (qnmat)
        self.shape = MT.shape  #dimension of a vector a
        self.n = n
        self.N=N
        # take MT transpose
        matrixtr(self) 

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
#        self.mt=mt
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
#        self.mt=mt
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
#        self.mt=mt
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
#        self.mt=mt
#        self.shape=mt.shape
#        self.ndim=mt.ndim
#        self.N=5

def prjop_init(n:np.int64):
    if(n==3): # projection operator for decagonal
        prj=Qnprj_Deca()
    elif(n==4): # projection operator for octagonal
        prj=Qnprj_Octa()
    elif(n==5): # projection operator dodecagonal
        prj=Qnprj_Dode()

# for class cls cls should be octa, deca or dode
def prjop(v: qnv.Qnvec) -> qnv.Qnvec:
    qnm.printqnm("prj",prj)
    qnv.printqnv("v.vt",v)
    vei=vt@prj.mt  #@v.vt # vt assumed to be qnvec
    return vei

# projection into external space for class cls
def prjop_e(v:qnv.Qnvec) -> qnv.Qnvec:
    vei=vt@prj.mt  #@v.vt # vt assumed to be qnvec
    return vei[1:3]

# projection into internal space for class cls
def prjop_i(v: qnv.Qnvec) -> qnv.Qnvec:
    vei=vt@prj.mt  #@v.vt
    return vei[4:6]

def projection_numerical_par(vn: qnv.Qnvec) -> qnv.Qnvec:
    return prjop(prj,vn)

def projection3(vt: qnv.Qnvec) -> qnv.Qnvec:
    return prjop_i(prj,vt)

# alias for prjop(vt)
def get_internal_component_numerical(vt: qnv.Qnvec) -> qnv.Qnvec:
    return prjop_i(vt)

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

if __name__ == '__main__':
    # test for qnnum projection operators

    def get_bmt(a,N):
        n=a.shape[0]
        np0=np.zeros((n,n)) # nxn zero matrix
        print("np0",np0)
        b=qnm.Qnmat(np0,n,N) #qnnum zero matrix
        for i in range(n):
            for j in range(n):
                b.mt[i][j]=qnn.flt2qn(a.mt[i][j],N)
        return b
    
    n=6
    N=2
    a=Prj_Octa() # float projection operator
    la=a.shape
    print("la",la)
    
    b=get_bmt(a,N)
    print("Octa.b")
    print(b.mt)
    qnm.printqnm("Octa.mt",b)

    N=5
    a=Prj_Deca() # float projection operator
    la=a.shape
    print("la",la)
    b=get_bmt(a,N)
    print("Deca.b")
    print(b.mt)
    qnm.printqnm("Deca.mt",b)

    N=3
    a=Prj_Dode() # float projection operator
    la=a.shape
    print("la",la)
    b=get_bmt(a,N)
    print("Dode.b")
    print(b.mt)
    qnm.printqnm("Dode.mt",b)
 
    
    # check qnmatinv
    N=2 # octagonal
    n=0
    mto=Qnprj_Octa()
    qnm.printqnm("mto",mto)
    
    mtoi=Qnprj_Octa()
    mth.qnmatinv(mtoi,np.int64(6))
    qnm.printqnm("mtoi",mtoi)
    
    # check lattice vector external and internal space components
    ndv=np.ndarray(3**3,dtype=qnn.Qnnum)
    for i1 in range(-1,2):
        for i2 in range(-1,2):
            for i3 in range(-1,2):
                for i4 in range(-1,2):
                    V1=qnn.Qnnum(np.array([ i1, 0, 1]),N)
                    V2=qnn.Qnnum(np.array([ i2, 0, 1]),N)
                    V3=qnn.Qnnum(np.array([ i3, 0, 1]),N)
                    V4=qnn.Qnnum(np.array([ i4, 0, 1]),N)
                    V0=qnn.Qnnum(np.array([ 0, 0, 1]),N)
                    VT=np.array([v1,v2,v3,V4,V0,V0])
                    ndv[n]=qnv.Qnvec(VT,6,N) # lattice vector
                    qnn.printqnv("ndv",ndv) #print qnvector expression
                    qnv=vt@mto.mt  #@ndv.vt #external enternal components
                    qnn.printqnv("qnv",qnv) #print qnvector expression
                    n+=1

                    
                    
    
        
