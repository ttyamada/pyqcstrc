import numpy as np
import cython

from numpy.typing import NDArray
from octa2.math1 import (mul,add)

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
    v3e=mtrixcal(M0,M0,M0,M0,M0,M0,vt) # 0,0,0,0,0,0 
    v1i=mtrixcal(M3,M2,M0,M1,M0,M0,vt) #
    v2i=mtrixcal(M0,M1,M4,M1,M0,M0,vt) #
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
    #v3e=mtrixcal(M0,M0,M0,M0,M0,M0,vt) # 0,0,0,0,0,0 
    v1i=mtrixcal(M3,M2,M0,M1,M0,M0,vt) #
    v2i=mtrixcal(M0,M1,M4,M1,M0,M0,vt) #
    v3i=mtrixcal(M0,M0,M0,M0,M0,M0,vt) # 0,0,0,0,0,0
    return np.array([v1i,v2i,v3i],dtype=np.int64)

def mtrixcal(m1: NDArray[np.int64],m2: NDArray[np.int64],\
             m3: NDArray[np.int64],m4: NDArray[np.int64],\
             m5: NDArray[np.int64],m6: NDArray[np.int64],\
             v: NDArray[np.int64]) -> NDArray[np.int64]:
    """function used in projection()
                        projection3()
                        projection_perp()
    
    Parameters
    ----------
    m1,m2,m3,m4,m5,m6:array for projection materix
    v: array
        6-dimensional vector in SQRT2-style

    Returns
    -------
    6d vectors projected onto Eperp in SQRT2-style.
    """

    a1=mul(m1,v[0])
    a2=mul(m2,v[1])
    a3=mul(m3,v[2])
    a4=mul(m4,v[3])
    a5=mul(m5,v[4])
    a6=mul(m6,v[5])
    #a1:np.int64
    a1=add(a1,a2)
    a1=add(a1,a3)
    a1=add(a1,a4)
    a1=add(a1,a5)
    a1=add(a1,a6)
    return a1


