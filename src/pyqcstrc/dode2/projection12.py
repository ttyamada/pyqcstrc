import numpy as np
from numpy.typing import NDArray
import random
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnmat.qnmat as qnm
import pyqcstrc.qnclass.qnmath

TAU=np.sqrt(3)/2.0
SQRT3=np.sqrt(3)
N=3

EPS=1e-6 # tolerance

#########
#  WIP  #
#########
def projection_numerical_par(vn: NDArray[np.float64]) -> NDArray[np.float64]:
    """This returns 6D vector which corresponds to a projection of vn onto Epar.
    
    Parameters
    ----------
    v: array
        6-dimensional vector

    Returns
    -------
    6d vectors projected onto Eperp.
    """
    m=np.array([\
            [ 0.5,          0.577350269,  0.0,         -0.288675135,  0.0,  0.0],\
            [ 0.288675135,  0.5,          0.288675135,  0.0,          0.0,  0.0],\
            [ 0.0,          0.288675135,  0.5,          0.288675135,  0.0,  0.0],\
            [-0.288675135,  0.0,          0.577350269,  0.5,          0.0,  0.0],\
            [ 0.0,          0.0,          0.0,          0.0,          1.0,  0.0],\
            [ 0.0,          0.0,          0.0,          0.0,          0.0,  0.0],\
        ])
    return m@vn

def projection(vt: NDArray[np.int64]) -> NDArray[np.int64]:
    """projection of a 6d vector onto Epar and Eperp in "SIN-style"
    NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    alpha = 2*a/np.sqrt(6)
    see Yamamoto ActaCrystal (1997)
    
    Parameters
    ----------
    vt: array
        6-dimensional vector in SIN-style
    
    Returns
    -------
    array containing two 3d vectors projected onto Epar and Eperp in SIN-style.
    """
    M0=np.array([ 0, 0, 1])
    M1=np.array([ 1, 0, 1])
    M2=np.array([-1, 0, 1])
    M3=np.array([ 1, 0, 2])
    M4=np.array([-1, 0, 2])
    M5=np.array([ 0, 1, 2])
    M6=np.array([ 0,-1, 2])
    #M5=np.array([ 0, 1, 1])
    #M6=np.array([ 0,-1, 1])
    v1e=mtrixcal(M5,M1,M0,M4,M0,M0,vt) # sin,1,0,-0.5,0,0
    v2e=mtrixcal(M4,M0,M1,M5,M0,M0,vt) # -0.5,0,1,sin,0,0
    v1i=mtrixcal(M6,M1,M0,M4,M0,M0,vt) # -sin,1,0,-0.5,0,0
    v2i=mtrixcal(M4,M0,M1,M6,M0,M0,vt) # -0.5,0,1,-sin,0,0
    v3e=mtrixcal(M0,M0,M0,M0,M1,M0,vt) # 0,0,0,0,1,0
    v3i=mtrixcal(M0,M0,M0,M0,M0,M1,vt) # 0,0,0,0,0,1
    return np.array([[v1e,v2e,v3e],[v1i,v2i,v3i]],dtype=np.int64)

def projection3(vt: NDArray[np.int64]) -> NDArray[np.int64]:
    """projection of a 6d vector onto Eperp in "SIN-style"
    NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    alpha = 2*a/np.sqrt(6)
    see Yamamoto ActaCrystal (1997)
    
    Parameters
    ----------
    vt: array
        6-dimensional vector in SQRT3-style
    
    Returns
    -------
    3d vectors projected onto Eperp in SQRT3-style.
    """
    M0=qnnum.Qnnum([ 0, 0, 1],3) #qnnum
    M1=qnnum.Qnnum([ 1, 0, 1]) #qnnum
    M2=np.array([-1, 0, 1]) #qnnum
    M3=np.array([ 1, 0, 2]) #qnnum
    M4=np.array([-1, 0, 2]) #qnnum
    M5=np.array([ 0, 1, 2]) #qnnum
    M6=np.array([ 0,-1, 2]) #qnnum
    #M5=np.array([ 0, 1, 1]) #qnnum
    #M6=np.array([ 0,-1, 1]) #qnnum
    #v1e=mtrixcal(M5,M1,M0,M4,M0,M0,vt) # sin,1,0,-0.5,0,0
    #v2e=mtrixcal(M4,M0,M1,M5,M0,M0,vt) # -0.5,0,1,sin,0,0
    #v3e=mtrixcal(M0,M0,M0,M0,M1,M0,vt) # 0,0,0,0,1,0
    v1i=mtrixcal(M6,M1,M0,M4,M0,M0,vt) # -sin,1,0,-0.5,0,0
    v2i=mtrixcal(M4,M0,M1,M6,M0,M0,vt) # -0.5,0,1,-sin,0,0
    v3i=mtrixcal(M0,M0,M0,M0,M0,M1,vt) # 0,0,0,0,0,1
 
    return np.array([v1i,v2i,v3i],dtype=np.int64)

def mtrixcal(m1: NDArray[np.int64],m2: NDArray[np.int64],m3: NDArray[np.int64],
             m4: NDArray[np.int64],m5: NDArray[np.int64],m6: NDArray[np.int64],
             v: NDArray[np.int64]) -> NDArray[np.int64]:
#def mtrixcal(m: np.matrix ,v: NDArray[np.int64]) -> NDArray[np.int64]:
    """function used in projection()
                        projection3()
                        projection_perp()
    
    Parameters
    ----------
    m1,m2,m3,m4,m5,m6:array for projection materix
    v: array
        6-dimensional vector in SQRT3-style

    Returns
    -------
    6d vectors projected onto Eperp in SQRT3-style.
    """
    a1=m1*v[0]  #mul(m1,v[0])
    a2=m2*v[1]  #mul(m2,v[1])
    a3=m3*v[2]  #mul(m3,v[2])
    a4=m4*v[3]  #mul(m4,v[3])
    a5=m5*v[4]  #mul(m5,v[4])
    a6=m6*v[5]  #mul(m6,v[5])
    a1=a1+a2    #add(a1,a2)
    a1=a1+a3    #add(a1,a3)
    a1=a1+a4    #add(a1,a4)
    a1=a1+a5    #add(a1,a5)
    a1=a1+a6  #add(a1,a6)
    return a1
