import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnmat.qnmat as qnm
import pyqcstrc.qnclass.qnmath

# for octagonal QCs
class Qnprj_Octa:
    def __init__(self):
        M0=qnn.Qnnum(np.array([ 0, 0, 1]),2) #  0
        M1=qnn.Qnnum(np.array([ 1, 0, 1]),2) #  1
        M2=qnn.Qnnum(np.array([-1, 0, 1]),2) # -1
        M3=qnn.Qnnum(np.array([ 0, 1, 2]),2) #  sqrt(2)/2
        M4=qnn.Qnnum(np.array([ 0,-1, 2]),2) # -sqrt(2)/2
        MT=np.array([[M2,M1,M0,M2,M0,M0], #
           [M0,M1,M3,M1,M0,M0], #
           [M0,M0,M0,M0,M0,M0], # 0,0,0,0,0,0 
           [M3,M2,M0,M1,M0,M0], #
           [M0,M1,M4,M1,M0,M0], #
           [M0,M0,M0,M0,M0,M0]]) # 0,0,0,0,0,0
        self.mt=qnm.Qnmat(MT) # matrix of qnnum (qnmat)
        self.shape = MT.shape  #dimension of a vector a
        self.ndim = MT.ndim

# for decagonal QCs
class Qnprj_Deca:
    # note that this use orthorhombic coordinate system
    def __init__(self):
        M0=qnn.Qnnum(np.array([ 0, 0, 1]),5) #  0
        M1=qnn.Qnnum(np.array([ 1, 0, 1]),5) #  1
        M2=qnn.Qnnum(np.array([-1, 0, 1]),5) # -1
        M3=qnn.Qnnum(np.array([ 0, 1, 1]),5) #  sqrt(5)
        M4=qnn.Qnnum(np.array([ 0,-1, 1]),5) # -sqrt(5)
        MT=np.array([[M2,M1,M0,M2,M0,M0], #
           [M0,M1,M3,M1,M0,M0], #
           [M0,M0,M0,M0,M0,M0], # 0,0,0,0,0,0 
           [M3,M2,M0,M1,M0,M0], #
           [M0,M1,M4,M1,M0,M0], #
           [M0,M0,M0,M0,M0,M0]]) # 0,0,0,0,0,0
        self.mt=qnm.Qnmat(MT) # matrix of qnnum (qnmat)

# for dodecagonal QCs
class Qnprj_Dode:
    def __init__(self):
        """projection of a 6d vector onto Epar and Eperp in "SIN-style"
        NOTE: coefficient (alpha) of the projection matrix is set to be 1.
        alpha = 2*a/np.sqrt(6)
        see Yamamoto ActaCrystal (1997)
        """
        M0=qnn.Qnnum(np.array([ 0, 0, 1]),3)
        M1=qnn.Qnnum(np.array([ 1, 0, 1]),3)
        M2=qnn.Qnnum(np.array([-1, 0, 1]),3)
        M3=qnn.Qnnum(np.array([ 1, 0, 2]),3)
        M4=qnn.Qnnum(np.array([-1, 0, 2]),3)
        M5=qnn.Qnnum(np.array([ 0, 1, 2]),3) #  sqrt(3)/2
        M6=qnn.Qnnum(np.array([ 0,-1, 2]),3) # -sqrt(3)/2
        #M5=np.array([ 0, 1, 1]) # sqrt(3)
        #M6=np.array([ 0,-1, 1]) # -sqrt(3)
        MT=np.array([[M5,M1,M0,M4,M0,M0], # sin,1,0,-0.5,0,0
           [M4,M0,M1,M5,M0,M0], # -0.5,0,1,sin,0,0
           [M0,M0,M0,M0,M1,M0], # 0,0,0,0,1,0
           [M6,M1,M0,M4,M0,M0], # -sin,1,0,-0.5,0,0
           [M4,M0,M1,M6,M0,M0], # -0.5,0,1,-sin,0,0
           [M0,M0,M0,M0,M0,M1]]) # 0,0,0,0,0,1
        self.mt=qnm.Qnmat(MT) # matrix of qnnum (qnmat)

class Prj_Octa:
    def __init__(self):   
        TAU=np.sqrt(2)
        mt=np.array([\
            [ TAU,  0.0,  TAU,  0.0,  0.0,  0.0],\
            [ 1.0,  1.0, -1.0,  1.0,  0.0,  0.0],\
            [ 0.0,  TAU,  0.0, -TAU,  0.0,  0.0],\
            [-1.0,  1.0,  1.0,  1.0,  0.0,  0.0],\
            [ 0.0,  0.0,  0.0,  0.0,  1.0,  0.0],\
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0],\
            ])
        self.mt=mt
        self.shape=mt.shape
        self.ndim=mt.ndim

class Prj_Deca:
    def __init__(self):   
        PI = np.pi
        C=np.zeros(4,dtype=float)
        S=np.zeros(4,dtype=float)
        C0=np.zeros(4,dtype=float)
        TAU=(1+np.sqrt(5))/2
        #SCL=1/np.sqrt(2+TAU)
        #SCL=1/np.sqrt(5)
        #SCL=TAU
        #SCL=1/np.cos(2*PI/5)
        SCL=1
        for i in range(4):
            i1=i+1
            C0[i] = np.cos(2*PI*i1/5)
            C[i]=(C0[i]-1)*SCL
            S[i] = np.sin(2*PI*i1/5)*SCL
        print("C0",C0)
        print("C",C)
        print("S",S)
        mt=np.array([
            [ C[0], C[1], C[2], C[3], 0, 0],\
            [ S[0], S[1], S[2], S[3], 0, 0],\
            [ C[1], C[3], C[0], C[2], 0, 0],\
            [ S[1], S[3], S[0], S[2], 0, 0],\
            [    0,    0,    0,    0, 1, 0],\
            [    0,    0,    0,    0, 0, 0],\
            ])
        self.mt=mt
        self.shape=mt.shape
        self.ndim=mt.ndim
        
class Prj_Dode:
    def __init__(self):
        mt=np.array([\
            [ 0.5,          0.577350269,  0.0,         -0.288675135,  0.0,  0.0],\
            [ 0.288675135,  0.5,          0.288675135,  0.0,          0.0,  0.0],\
            [ 0.0,          0.288675135,  0.5,          0.288675135,  0.0,  0.0],\
            [-0.288675135,  0.0,          0.577350269,  0.5,          0.0,  0.0],\
            [ 0.0,          0.0,          0.0,          0.0,          1.0,  0.0],\
            [ 0.0,          0.0,          0.0,          0.0,          0.0,  0.0],\
            ])
        self.mt=mt
        self.shape=mt.shape
        self.ndim=mt.ndim

# for class cls cls should be octa, deca or dode
def prjop(prj,v):
    qnm.printqnm("prj",prj)
    qnv.printqnv("v.vt",v)
    vei=prj.mt@v.vt # vt assumed to be qnvec
    return vei

# projection into external space for class cls
def prjop_e(prj,v):
    vei=prj.mt@v.vt # vt assumed to be qnvec
    return vei[1:3]

# projection into internal space for class cls
def prjop_i(prj,v):
    vei=prj.mt@v.vt
    return vei[4:6]


if __name__ == '__main__':
    # test for qnnum projection operators

    def get_bmt(a,N):
        la=a.shape
        qnzero=qnn.Qnnum([0,0,1],N)     # qnnumber zero
        b=qnm.Qnmat(np.full(la,qnzero)) #qnnum zero matrix
        for i in range(6):
            for j in range(6):
                b.mt[i][j]=qnn.flt2qn(a.mt[i][j],N)
        return b

    a=Octa_num()
    la=a.shape
    print("la",la)
    N=2
    b=get_bmt(a,N)
    print("Octa.a")
    print(a.mt)
    qnm.printqnm("Octa.mt",b)

    a=Deca_num()
    la=a.shape
    print("la",la)
    N=5
    #b=get_bmt(a,N)
    print("Deca.a")
    print(a.mt)
    #qnm.printqnm("Deca.mt",b)

    a=Dode_num()
    la=a.shape
    print("la",la)
    N=3
    b=get_bmt(a,N)
    print("Dode.a")
    print(a.mt)
    qnm.printqnm("Dode.mt",b)
    
    
        
    #c=np.cos(np.pi/5)
    #s=np.sin(np.pi/5)
    #print("c",c)
    #print("s",s)
    #print("s/c",s/c)
    #qnc=qnn.flt2qn(c,N)
    #qns=qnn.flt2qn(s,N)
    #qnsc=qnn.flt2qn(s/c,N)
    #qnn.printqnn("qnc",qnc)
    #qnn.printqnn("qns",qns)
    #qnn.printqnn("qnsc",qnsc)
        
