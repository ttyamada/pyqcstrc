#
# PyQCdiff - Python library for Quasi-Crystal diffraction
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
cimport numpy as np
cimport cython

DTYPE_double = np.float64
DTYPE_int = int

cdef np.float64_t TAU=(1+np.sqrt(5))/2.0
cdef np.float64_t CONST1=1/np.sqrt(2*(2+TAU))
cdef np.float64_t CONST2=CONST1*np.sqrt(2) # CONST2=1/np.sqrt(2+TAU)
    
cpdef list projection_qpar_ico_elser(list v):
    """
    parallel component of a 6d Q vector in Elser's setting (ref: Yamamoto-1997)
    """
    cdef int h1,h2,h3,h4,h5,h6
    cdef double qx,qy,qz
    [h1,h2,h3,h4,h5,h6]=v
    qx=( h1-h5)+TAU*( h2+h3)
    qy=( h4+h6)+TAU*( h1+h5)
    qz=( h2-h3)+TAU*(-h4+h6)
    return [qx*CONST2,qy*CONST2,qz*CONST2]
    
cpdef list projection_qperp_ico_elser(list v):
    """
    perpendicular component of a 6d Q vector in Elser's setting (ref: Yamamoto-1997)
    """
    cdef int h1,h2,h3,h4,h5,h6
    cdef double qx,qy,qz
    [h1,h2,h3,h4,h5,h6]=v
    qx=(-h2-h3)+TAU*( h1-h5)
    qy=(-h1-h5)+TAU*( h4+h6)
    qz=( h4-h6)+TAU*( h2-h3)
    return [qx*CONST2,qy*CONST2,qz*CONST2]
    
cpdef list projection_qpar_ico_csg(list v):
    """
    parallel component of a 6d Q vector in Cahn Shechtman Gratias-1986 setting
    """
    cdef int h1,h2,h3,h4,h5,h6
    cdef double qx,qy,qz
    [h1,h2,h3,h4,h5,h6]=v
    qx=( h1-h4)+TAU*( h2+h5)
    qy=( h3-h6)+TAU*( h1+h4)
    qz=( h2-h5)+TAU*( h3+h6)
    return [qx*CONST1,qy*CONST1,qz*CONST1]
    
cpdef list projection_qperp_ico_csg(list v):
    """
    perpendicular component of a 6d Q vector in Cahn Shechtman Gratias-1986 setting
    """
    cdef int h1,h2,h3,h4,h5,h6
    cdef double qx,qy,qz
    [h1,h2,h3,h4,h5,h6]=v
    qx=( h2+h5)+TAU*(-h1+h4)
    qy=( h1+h4)+TAU*(-h3+h6)
    qz=( h3+h6)+TAU*( h2+h5)
    return [qx*CONST1,qy*CONST1,qz*CONST1]
    
cdef double norm(list v):
    return np.sqrt(v[0]**2+v[1]**2+v[2]**2)

cdef list lphason(np.ndarray[DTYPE_double, ndim=2] p, list qperp):
    cdef double dq1,dq2,dq3
    dq1=p[0][0]*qperp[0]+p[0][1]*qperp[1]+p[0][2]*qperp[2]
    dq2=p[1][0]*qperp[1]+p[0][1]*qperp[1]+p[1][2]*qperp[2]
    dq3=p[2][0]*qperp[2]+p[0][1]*qperp[1]+p[2][2]*qperp[2]
    return [dq1,dq2,dq3]
    
