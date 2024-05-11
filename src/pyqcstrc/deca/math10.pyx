#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
cimport numpy as np
cimport cython

DTYPE_double = np.float64
DTYPE_int = int

cdef np.float64_t SIN=np.sqrt(3)/2.0
cdef np.ndarray M0=np.array([ 0, 0, 1])
cdef np.ndarray M1=np.array([ 1, 0, 1])
cdef np.ndarray M2=np.array([-1, 0, 1])
cdef np.ndarray M3=np.array([ 1, 0, 2])
cdef np.ndarray M4=np.array([-1, 0, 2])
cdef np.ndarray M5=np.array([ 0, 1, 1])
cdef np.ndarray M6=np.array([ 0,-1, 1])

cpdef list add(DTYPE_int_t p1,DTYPE_int_t p2,DTYPE_int_t p3,DTYPE_int_t q1,DTYPE_int_t q2,DTYPE_int_t q3): # A+B
    cdef DTYPE_int_t c1,c2,c3,gcd
    cdef np.ndarray[DTYPE_int_t,ndim=1] x
    c1=p1*q3+q1*p3
    c2=p2*q3+q2*p3
    c3=p3*q3
    x=np.array([c1,c2,c3])
    gcd=np.gcd.reduce(x)
    if c3/gcd<0:
        return [int(-c1/gcd),int(-c2/gcd),int(-c3/gcd)]
    else:
        return [int(c1/gcd),int(c2/gcd),int(c3/gcd)]

cpdef list sub(DTYPE_int_t p1,DTYPE_int_t p2,DTYPE_int_t p3,DTYPE_int_t q1,DTYPE_int_t q2,DTYPE_int_t q3): # A-B
    cdef DTYPE_int_t c1,c2,c3,gcd
    cdef np.ndarray[DTYPE_int_t,ndim=1] x
    c1=p1*q3-q1*p3
    c2=p2*q3-q2*p3
    c3=p3*q3
    x=np.array([c1,c2,c3])
    gcd=np.gcd.reduce(x)
    if c3/gcd<0:
        return [int(-c1/gcd),int(-c2/gcd),int(-c3/gcd)]
    else:
        return [int(c1/gcd),int(c2/gcd),int(c3/gcd)]

cpdef list mul(DTYPE_int_t p1,DTYPE_int_t p2,DTYPE_int_t p3,DTYPE_int_t q1,DTYPE_int_t q2,DTYPE_int_t q3): # A*B
    cdef DTYPE_int_t c1,c2,c3,gcd
    cdef np.ndarray[DTYPE_int_t,ndim=1] x
    c1=4*p1*q1+3*p2*q2
    c2=4*(p1*q2+p2*q1)
    c3=4*p3*q3
    x=np.array([c1,c2,c3])
    gcd=np.gcd.reduce(x)
    #print('c1,c2,c3,gcd = %d %d %d %d'%(c1,c2,c3,gcd))
    if c3/gcd<0:
        return [int(-c1/gcd),int(-c2/gcd),int(-c3/gcd)]
    else:
        return [int(c1/gcd),int(c2/gcd),int(c3/gcd)]

cpdef list div(DTYPE_int_t p1,DTYPE_int_t p2,DTYPE_int_t p3,DTYPE_int_t q1,DTYPE_int_t q2,DTYPE_int_t q3): # A/B
    cdef DTYPE_int_t c1,c2,c3,gcd
    cdef np.ndarray[DTYPE_int_t,ndim=1] x
    if q1==0 and q2==0:
        print('ERROR_1:division error')
        return 1
    else:
        if p1==0 and p2==0:
            return [0,0,1]
        else:
            if q1!=0 and q2!=0:
                if 4*q1**2-3*q2**2!=0:
                    c1=q3*(4*p1*q1-3*p2*q2)
                    c2=-4*q3*(p1*q2-p2*q1)
                    c3=p3*(4*q1**2-3*q2**2)
                else:
                    c1=3*p2*q3
                    c2=4*p1*q3
                    c3=6*p3*q2
            elif q1==0 and q2!=0:
                c1=3*p2*q3
                c2=4*p1*q3
                c3=3*p3*q2
            else:
            #elif q1!=0 and q2==0:
                c1=p1*q3
                c2=p2*q3
                c3=p3*q1
            x=np.array([c1,c2,c3])
            gcd=np.gcd.reduce(x)
            if gcd!=0:
                if c3/gcd<0:
                    return [int(-c1/gcd),int(-c2/gcd),int(-c3/gcd)]
                else:
                    return [int(c1/gcd),int(c2/gcd),int(c3/gcd)]
            else:
                print('ERROR_2:division error',c1,c2,c3,p1,p2,p3,q1,q2,q3)
                return 1

cpdef int gcd(DTYPE_int_t p1,DTYPE_int_t p2,DTYPE_int_t p3):
    cdef np.ndarray[DTYPE_int_t,ndim=1] x
    x=np.array([p1,p2,p3])
    return np.gcd.reduce(x)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list projection(np.ndarray[DTYPE_int_t, ndim=1] h1,
                        np.ndarray[DTYPE_int_t, ndim=1] h2,
                        np.ndarray[DTYPE_int_t, ndim=1] h3,
                        np.ndarray[DTYPE_int_t, ndim=1] h4,
                        np.ndarray[DTYPE_int_t, ndim=1] h5,
                        np.ndarray[DTYPE_int_t, ndim=1] h6):
    # projection of a 6d vector onto Eperp, using "SIN-style"
    #
    # NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    # alpha = 2*a/np.sqrt(6)
    # see Yamamoto ActaCrystal (1997)
    cdef np.ndarray[DTYPE_int_t, ndim=1] v1i,v2i,v3i,v1e,v2e,v3e
    v1e=mtrixcal(M5,M1,M0,M4,M0,M0,h1,h2,h3,h4,h5,h6) # sin,1,0,-0.5,0,0
    v2e=mtrixcal(M4,M0,M1,M5,M0,M0,h1,h2,h3,h4,h5,h6) # -0.5,0,1,sin,0,0
    v1i=mtrixcal(M6,M1,M0,M4,M0,M0,h1,h2,h3,h4,h5,h6) # -sin,1,0,-0.5,0,0
    v2i=mtrixcal(M4,M0,M1,M6,M0,M0,h1,h2,h3,h4,h5,h6) # -0.5,0,1,-sin,0,0
    v3e=mtrixcal(M0,M0,M0,M0,M1,M0,h1,h2,h3,h4,h5,h6) # 0,0,0,0,1,0
    v3i=mtrixcal(M0,M0,M0,M0,M0,M1,h1,h2,h3,h4,h5,h6) # 0,0,0,0,0,1
    return [v1e,v2e,v3e,v1i,v2i,v3i]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list projection3(np.ndarray[DTYPE_int_t, ndim=1] h1,
                        np.ndarray[DTYPE_int_t, ndim=1] h2,
                        np.ndarray[DTYPE_int_t, ndim=1] h3,
                        np.ndarray[DTYPE_int_t, ndim=1] h4,
                        np.ndarray[DTYPE_int_t, ndim=1] h5,
                        np.ndarray[DTYPE_int_t, ndim=1] h6):
    # projection of a 6d vector onto Eperp, using "SIN-style"
    #
    # NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    # alpha = 2*a/np.sqrt(6)
    # see Yamamoto ActaCrystal (1997)
    cdef np.ndarray[DTYPE_int_t, ndim=1] v1i,v2i,v3i
    v1i=mtrixcal(M6,M1,M0,M4,M0,M0,h1,h2,h3,h4,h5,h6) # -sin,1,0,-0.5,0,0
    v2i=mtrixcal(M4,M0,M1,M6,M0,M0,h1,h2,h3,h4,h5,h6) # -0.5,0,1,-sin,0,0
    v3i=mtrixcal(M0,M0,M0,M0,M0,M1,h1,h2,h3,h4,h5,h6) # 0,0,0,0,0,1
    return [v1i,v2i,v3i]
