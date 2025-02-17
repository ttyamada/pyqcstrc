import numpy as np
from numpy.typing import NDArray
from ico2.math1 import mul,add

def projection(vt: NDArray[np.int64]) -> NDArray[np.int64]:
    """projection of a 6d vector onto Epar and Eperp in "TAU-style"
    NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    alpha = a/np.sqrt(2.0+TAU)
    see Yamamoto ActaCrystal (1997)

    Parameters
    ----------
    vt: array
        6-dimensional vector in TAU-style

    Returns
    -------
    array containing two 3d vectors projected onto Epar and Eperp in TAU-style.
    """
    m0=np.array([ 0, 0, 1]) #  0 in 'TAU-style'
    m1=np.array([ 1, 0, 1]) #  1
    m2=np.array([-1, 0, 1]) # -1
    m3=np.array([ 0, 1, 1]) #  tau
    m4=np.array([ 0,-1, 1]) # -tau
    v1e=mtrixcal(m1,m3,m3,m0,m2,m0,vt) # 1,tau,tau,0,-1,0
    v2e=mtrixcal(m3,m0,m0,m1,m3,m1,vt) # tau,0,0,1,TAU,1
    v3e=mtrixcal(m0,m1,m2,m4,m0,m3,vt) # 0,1,-1,-tau,0,tau
    v1i=mtrixcal(m3,m2,m2,m0,m4,m0,vt) # tau,-1,-1,0,-tau,0
    v2i=mtrixcal(m2,m0,m0,m3,m2,m3,vt) # -1,0,0,tau,-1,tau
    v3i=mtrixcal(m0,m3,m4,m1,m0,m2,vt) # 0,tau,-tau,1,0,-1
    return np.array([[v1e,v2e,v3e],[v1i,v2i,v3i]],dtype=np.int64)

def projection3(vt: NDArray[np.int64]) -> NDArray[np.int64]:
    """projection of a 6d vector onto Eperp in "TAU-style"
    NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    alpha = a/np.sqrt(2.0+TAU)
    see Yamamoto ActaCrystal (1997)
    
    Parameters
    ----------
    vt: array
        6-dimensional vector in TAU-style

    Returns
    -------
    3d vectors projected onto Eperp in TAU-style.
    """

    m0=np.array([ 0, 0, 1]) #  0 in 'TAU-style'
    m1=np.array([ 1, 0, 1]) #  1
    m2=np.array([-1, 0, 1]) # -1
    m3=np.array([ 0, 1, 1]) #  tau
    m4=np.array([ 0,-1, 1]) # -tau
    #v1e=mtrixcal(m1,m3,m3,m0,m2,m0,h1,h2,h3,h4,h5,h6) # 1,tau,tau,0,-1,0
    #v2e=mtrixcal(m3,m0,m0,m1,m3,m1,h1,h2,h3,h4,h5,h6) # tau,0,0,1,TAU,1
    #v3e=mtrixcal(m0,m1,m2,m4,m0,m3,h1,h2,h3,h4,h5,h6) # 0,1,-1,-tau,0,tau
    v1i=mtrixcal(m3,m2,m2,m0,m4,m0,vt) # tau,-1,-1,0,-tau,0
    v2i=mtrixcal(m2,m0,m0,m3,m2,m3,vt) # -1,0,0,tau,-1,tau
    v3i=mtrixcal(m0,m3,m4,m1,m0,m2,vt) # 0,tau,-tau,1,0,-1
    return np.array([v1i,v2i,v3i],dtype=np.int64)

def projection_perp(vt: NDArray[np.int64]) -> NDArray[np.int64]:
    """This returns 6D indeces of a projection of 6D vector (v) onto Eperp
    
    Parameters
    ----------
    v: array
        6-dimensional vector in TAU-style

    Returns
    -------
    6d vectors projected onto Eperp in TAU-style.
    """
    m1=np.array([ 1, 0, 2]) #  (TAU+2)/2/(2+TAU)=1/2 in 'TAU-style'
    m2=np.array([-1, 2,10]) #  TAU/2/(2+TAU)
    m3=np.array([ 1,-2,10]) # -TAU/2/(2+TAU)
    h1=mtrixcal(m1,m3,m3,m3,m3,m3,vt) # (tau+2,-tau,-tau,-tau,-tau,-tau)/2
    h2=mtrixcal(m3,m1,m3,m2,m2,m3,vt) # (-tau,tau+2,-tau,tau,tau,-tau)/2
    h3=mtrixcal(m3,m3,m1,m3,m2,m2,vt) # (-tau,-tau,tau+2,-tau,tau,tau)/2
    h4=mtrixcal(m3,m2,m3,m1,m3,m2,vt) # (-tau,tau,-tau,tau+2,-tau,tau)/2
    h5=mtrixcal(m3,m2,m2,m3,m1,m3,vt) # (-tau,tau,tau,-tau,tau+2,-tau)/2
    h6=mtrixcal(m3,m3,m2,m2,m3,m1,vt) # (-tau,-tau,tau,tau,-tau,tau+2)/2
    return np.array([h1,h2,h3,h4,h5,h6],dtype=np.int64)
    #const=1/(2.0+TAU)
    #m1 =((TAU+2)*n1 -     TAU*n2 -     TAU*n3 -     TAU*n4 -     TAU*n5 -     TAU*n6)/2.0*const
    #m2 = ( - TAU*n1 + (TAU+2)*n2 -     TAU*n3 +     TAU*n4 +     TAU*n5 -     TAU*n6)/2.0*const
    #m3 = ( - TAU*n1 -     TAU*n2 + (TAU+2)*n3 -     TAU*n4 +     TAU*n5 +     TAU*n6)/2.0*const
    #m4 = ( - TAU*n1 +     TAU*n2 -     TAU*n3 + (TAU+2)*n4 -     TAU*n5 +     TAU*n6)/2.0*const
    #m5 = ( - TAU*n1 +     TAU*n2 +     TAU*n3 -     TAU*n4 + (TAU+2)*n5 -     TAU*n6)/2.0*const
    #m6 = ( - TAU*n1 -     TAU*n2 +     TAU*n3 +     TAU*n4 -     TAU*n5 + (TAU+2)*n6)/2.0*const
    #return m1,m2,m3,m4,m5,m6

def mtrixcal(
    m1: NDArray[np.int64],
    m2: NDArray[np.int64],
    m3: NDArray[np.int64],
    m4: NDArray[np.int64],
    m5: NDArray[np.int64],
    m6: NDArray[np.int64],
    v: NDArray[np.int64],
    ) -> NDArray[np.int64]:
    """function used in projection()
                        projection3()
                        projection_perp()
    
    Parameters
    ----------
    m1,m2,m3,m4,m5,m6:array for projection materix
    v: array
        6-dimensional vector in TAU-style

    Returns
    -------
    6d vectors projected onto Eperp in TAU-style.
    """
    a1=mul(m1,v[0])
    a2=mul(m2,v[1])
    a3=mul(m3,v[2])
    a4=mul(m4,v[3])
    a5=mul(m5,v[4])
    a6=mul(m6,v[5])
    a1=add(a1,a2)
    a1=add(a1,a3)
    a1=add(a1,a4)
    a1=add(a1,a5)
    a1=add(a1,a6)
    return a1
