#########
import numpy as np
from numpy.typing import NDArray
import random

TAU=np.sqrt(2)
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
            [ TAU,  0.0,  TAU,  0.0,  0.0,  0.0],\
            [ 1.0,  1.0, -1.0,  1.0,  0.0,  0.0],\
            [ 0.0,  TAU,  0.0, -TAU,  0.0,  0.0],\
            [-1.0,  1.0,  1.0,  1.0,  0.0,  0.0],\
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0],\
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0],\
        ])
    return m@vn

def projection(vt: NDArray[np.int64]) -> NDArray[np.int64]:
    """projection of a 6d vector onto Epar and Eperp in "SQRT2-style"
    NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    alpha = a/np.sqrt(2)
    see Yamamoto ActaCrystal (1997)
    
    Parameters
    ----------
    vt: array
        6-dimensional vector in SQRT2-style
    
    Returns
    -------
    array containing two 3d vectors projected onto Epar and Eperp in SQRT2-style.
    """
    M0=np.array([ 0, 0, 1]) #  0
    M1=np.array([ 1, 0, 1]) #  1
    M2=np.array([-1, 0, 1]) # -1
    M3=np.array([ 0, 1, 1]) #  sqrt(2)
    M4=np.array([ 0,-1, 1]) # -sqrt(2)
    v1e=mtrixcal(M2,M1,M0,M2,M0,M0,vt) #
    v2e=mtrixcal(M0,M1,M3,M1,M0,M0,vt) #
    v1i=mtrixcal(M3,M2,M0,M1,M0,M0,vt) #
    v2i=mtrixcal(M0,M1,M4,M1,M0,M0,vt) #
    v3e=mtrixcal(M0,M0,M0,M0,M0,M0,vt) # 0,0,0,0,0,0 
    v3i=mtrixcal(M0,M0,M0,M0,M0,M0,vt) # 0,0,0,0,0,0
    return np.array([[v1e,v2e,v3e],[v1i,v2i,v3i]],dtype=np.int64)

def projection3(vt: NDArray[np.int64]) -> NDArray[np.int64]:
    """projection of a 6d vector onto Eperp in "SQRT2-style"
    NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    alpha = a/np.sqrt(2)
    see Yamamoto ActaCrystal (1997)
    
    Parameters
    ----------
    vt: array
        6-dimensional vector in SQRT2-style
    
    Returns
    -------
    3d vectors projected onto Eperp in SQRT2-style.
    """
    M0=np.array([ 0, 0, 1]) #  0
    M1=np.array([ 1, 0, 1]) #  1
    M2=np.array([-1, 0, 1]) # -1
    M3=np.array([ 0, 1, 1]) #  sqrt(2)
    M4=np.array([ 0,-1, 1]) # -sqrt(2)
    #v1e=mtrixcal(M2,M1,M0,M2,M0,M0,vt) #
    #v2e=mtrixcal(M0,M1,M3,M1,M0,M0,vt) #
    v1i=mtrixcal(M3,M2,M0,M1,M0,M0,vt) # inner product of M3,,M0 and vt
    v2i=mtrixcal(M0,M1,M4,M1,M0,M0,vt) #
    #v3e=mtrixcal(M0,M0,M0,M0,M0,M0,vt) # 0,0,0,0,0,0 
    v3i=mtrixcal(M0,M0,M0,M0,M0,M0,vt) # 0,0,0,0,0,0
    return np.array([v1i,v2i,v3i],dtype=np.int64)
