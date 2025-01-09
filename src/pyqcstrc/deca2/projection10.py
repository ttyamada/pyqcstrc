import numpy as np
from numpy.typing import NDArray
import random

EPS=1e-6 # tolerance
PI = np.pi
TAU = (np.sqrt(5)+1)/2.0
CONST1 =  1/np.sqrt(5)

C1 = np.cos(2*PI*1/5)
C2 = np.cos(2*PI*2/5)
C3 = np.cos(2*PI*3/5)
C4 = np.cos(2*PI*4/5)
C5 = np.cos(2*PI*5/5)
S1 = np.sin(2*PI*1/5)
S2 = np.sin(2*PI*2/5)
S3 = np.sin(2*PI*3/5)
S4 = np.sin(2*PI*4/5)
S5 = np.sin(2*PI*5/5)
C21 = np.cos(4*PI*1/5)
C22 = np.cos(4*PI*2/5)
C23 = np.cos(4*PI*3/5)
C24 = np.cos(4*PI*4/5)
C25 = np.cos(4*PI*5/5)
S21 = np.sin(4*PI*1/5)
S22 = np.sin(4*PI*2/5)
S23 = np.sin(4*PI*3/5)
S24 = np.sin(4*PI*4/5)
S25 = np.sin(4*PI*5/5)

def projection_5d(vn: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    parallel component of 5D direct lattice vector, x and y in 2*a/np.sqrt(5) unit, and z in c unit.
    input
    list:r, 5D reflection index
    """
    mat=np.array([
        [ C1-1, C2-1, C3-1, C4-1, 0],\
        [   S1,   S2,   S3,   S4, 0],\
        [    0,    0,    0,    0, 1],\
        [ C2-1, C4-1, C1-1, C3-1, 0],\
        [   S2,   S4,   S1,   S3, 0],\
    ])
    return mat@vn
    
def projection3_5d(vn: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    perpendicular component of 5D reciprocal lattice vector, in 2*a/np.sqrt(5) unit.
    input
    list:r, 5D reflection index
    float:a, lattice constant
    """
    mat=np.array([
        [ C2-1, C4-1, C1-1, C3-1, 0],\
        [   S2,   S4,   S1,   S3, 0],\
    ])
    return mat@vn

def projection_6d(vn: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    parallel component of 6D direct lattice vector, x and y in 2*a/np.sqrt(5) unit, and z in c unit.
    
    """
    mat=np.array([
        [ C1,  C2,  C3,  C4,  C5, 0],\
        [ S1,  S2,  S3,  S4,  S5 ,0],\
        [  0,   0,   0,   0,   0, 1],\
        [C21, C22, C23, C24, C25, 0],\
        [S21, S22, S23, S24, S25, 0],\
    ])
    return mat@vn
    
def projection3_6d(vn: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    perpendicular component of 6D direct lattice vector, x and y in 2*a/np.sqrt(5) unit.
    """
    mat=np.array([
        [C21, C22, C23, C24, C25, 0],\
        [S21, S22, S23, S24, S25, 0],\
    ])
    return mat@vn
    
def transform6to5(vn: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    this function transforms a 6-dim coordinates to its corresponding 5-dim coordinates.
    """
    mat = np.array([
        [ 4,-1,-1,-1,-1, 0],\
        [-1, 4,-1,-1,-1, 0],\
        [-1,-1, 4,-1,-1, 0],\
        [-1,-1,-1, 4,-1, 0],\
        [ 0, 0, 0, 0, 0, 5],\
    ])/5.0
    return mat@x

def projection(vt: NDArray[np.int64]) -> NDArray[np.int64]:
    """projection of a 6d vector onto Epar and Eperp in "SQRT5-style"
    NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    alpha = a/np.sqrt(2)
    see Yamamoto ActaCrystal (1997)
    
    Parameters
    ----------
    vt: array
        6-dimensional vector in SQRT5-style
    
    Returns
    -------
    array containing two 3d vectors projected onto Epar and Eperp in SQRT5-style.
    """
    M0=np.array([ 0, 0, 1]) #  0
    M1=np.array([ 1, 0, 1]) #  1
    M2=np.array([-1, 0, 1]) # -1
    M3=np.array([ 0, 1, 1]) #  sqrt(5)
    M4=np.array([ 0,-1, 1]) # -sqrt(5)
    v1e=mtrixcal(M2,M1,M0,M2,M0,M0,vt) #
    v2e=mtrixcal(M0,M1,M3,M1,M0,M0,vt) #
    v3e=mtrixcal(M0,M0,M0,M0,M0,M0,vt) # 0,0,0,0,0,0 
    v1i=mtrixcal(M3,M2,M0,M1,M0,M0,vt) #
    v2i=mtrixcal(M0,M1,M4,M1,M0,M0,vt) #
    v3i=mtrixcal(M0,M0,M0,M0,M0,M0,vt) # 0,0,0,0,0,0
    return np.array([[v1e,v2e,v3e],[v1i,v2i,v3i]],dtype=np.int64)

def projection3(vt: NDArray[np.int64]) -> NDArray[np.int64]:
    """projection of a 6d vector onto Eperp in "SQRT5-style"
    NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    alpha = a/np.sqrt(2)
    see Yamamoto ActaCrystal (1997)
    
    Parameters
    ----------
    vt: array
        6-dimensional vector in SQRT5-style
    
    Returns
    -------
    3d vectors projected onto Eperp in SQRT5-style.
    """
    M0=np.array([ 0, 0, 1]) #  0
    M1=np.array([ 1, 0, 1]) #  1
    M2=np.array([-1, 0, 1]) # -1
    M3=np.array([ 0, 1, 1]) #  sqrt(5)
    M4=np.array([ 0,-1, 1]) # -sqrt(5)
    #v1e=mtrixcal(M2,M1,M0,M2,M0,M0,vt) #
    #v2e=mtrixcal(M0,M1,M3,M1,M0,M0,vt) #
    v1i=mtrixcal(M3,M2,M0,M1,M0,M0,vt) #
    v2i=mtrixcal(M0,M1,M4,M1,M0,M0,vt) #
    #v3e=mtrixcal(M0,M0,M0,M0,M0,M0,vt) # 0,0,0,0,0,0 
    v3i=mtrixcal(M0,M0,M0,M0,M0,M0,vt) # 0,0,0,0,0,0
    return np.array([v1i,v2i,v3i],dtype=np.int64)

