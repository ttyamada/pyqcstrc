import numpy as np
from numpy.typing import NDArray
import random

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
    #M5=np.array([ 0, 1, 1])
    #M6=np.array([ 0,-1, 1])
    M5=np.array([ 0, 1, 2])
    M6=np.array([ 0,-1, 2])
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
    M0=np.array([ 0, 0, 1])
    M1=np.array([ 1, 0, 1])
    M2=np.array([-1, 0, 1])
    M3=np.array([ 1, 0, 2])
    M4=np.array([-1, 0, 2])
    M5=np.array([ 0, 1, 2])
    M6=np.array([ 0,-1, 2])
    #M5=np.array([ 0, 1, 1])
    #M6=np.array([ 0,-1, 1])
    #v1e=mtrixcal(M5,M1,M0,M4,M0,M0,vt) # sin,1,0,-0.5,0,0
    #v2e=mtrixcal(M4,M0,M1,M5,M0,M0,vt) # -0.5,0,1,sin,0,0
    v1i=mtrixcal(M6,M1,M0,M4,M0,M0,vt) # -sin,1,0,-0.5,0,0
    v2i=mtrixcal(M4,M0,M1,M6,M0,M0,vt) # -0.5,0,1,-sin,0,0
    #v3e=mtrixcal(M0,M0,M0,M0,M1,M0,vt) # 0,0,0,0,1,0
    v3i=mtrixcal(M0,M0,M0,M0,M0,M1,vt) # 0,0,0,0,0,1
    return np.array([v1i,v2i,v3i],dtype=np.int64)

