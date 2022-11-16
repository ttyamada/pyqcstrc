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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray mul_vectors(np.ndarray[DTYPE_int_t, ndim=3] vecs,
                        list a):
    cdef DTYPE_int_t num
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a,tmp2b
    num=vecs.shape[0]
    for i in range(num):
        tmp2a=mul_vector(vecs[i],a)
        if i==0:
            tmp2b=tmp2a
        else:
            tmp2b=np.vstack([tmp2b,tmp2a])
    #return tmp1b.reshape(num,6,3)
    return tmp2b

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray mul_vector(np.ndarray[DTYPE_int_t, ndim=2] vec,
                        list a):
    cdef DTYPE_int_t t1,t2,t3
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    for i in range(6):
        [t1,t2,t3]=mul(vec[i][0],vec[i][1],vec[i][2],a[0],a[1],a[2])
        if i==0:
            tmp1a=np.array([t1,t2,t3])
        else:
            tmp1a=np.append(tmp1a,[t1,t2,t3])
    return tmp1a.reshape(6,3)
    #return tmp1a

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray add_vectors(np.ndarray[DTYPE_int_t, ndim=2] vec1,
                            np.ndarray[DTYPE_int_t, ndim=2] vec2):
    cdef DTYPE_int_t t1,t2,t3
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    for i in range(6):
        [t1,t2,t3]=add(vec1[i][0],vec1[i][1],vec1[i][2],vec2[i][0],vec2[i][1],vec2[i][2])
        if i==0:
            tmp1a=np.array([t1,t2,t3])
        else:
            tmp1a=np.append(tmp1a,[t1,t2,t3])
    #return tmp1a.reshape(6,3)
    return tmp1a.reshape(6,3)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray sub_vectors(np.ndarray[DTYPE_int_t, ndim=2] vec1,
                            np.ndarray[DTYPE_int_t, ndim=2] vec2):
    cdef DTYPE_int_t t1,t2,t3
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    for i in range(6):
        [t1,t2,t3]=sub(vec1[i][0],vec1[i][1],vec1[i][2],vec2[i][0],vec2[i][1],vec2[i][2])
        if i==0:
            tmp1a=np.array([t1,t2,t3])
        else:
            tmp1a=np.append(tmp1a,[t1,t2,t3])
    #return tmp1a.reshape(6,3)
    return tmp1a.reshape(6,3)

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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray mtrixcal(np.ndarray[DTYPE_int_t, ndim=1] v1,
                        np.ndarray[DTYPE_int_t, ndim=1] v2,
                        np.ndarray[DTYPE_int_t, ndim=1] v3,
                        np.ndarray[DTYPE_int_t, ndim=1] v4,
                        np.ndarray[DTYPE_int_t, ndim=1] v5,
                        np.ndarray[DTYPE_int_t, ndim=1] v6,
                        np.ndarray[DTYPE_int_t, ndim=1] n1,
                        np.ndarray[DTYPE_int_t, ndim=1] n2,
                        np.ndarray[DTYPE_int_t, ndim=1] n3,
                        np.ndarray[DTYPE_int_t, ndim=1] n4,
                        np.ndarray[DTYPE_int_t, ndim=1] n5,
                        np.ndarray[DTYPE_int_t, ndim=1] n6):
    cdef DTYPE_int_t a1,a2,a3,a4,a5,a6
    cdef np.ndarray[DTYPE_int_t,ndim=1] val
    
    [a1,a2,a3]=mul(v1[0],v1[1],v1[2],n1[0],n1[1],n1[2])
    [a4,a5,a6]=mul(v2[0],v2[1],v2[2],n2[0],n2[1],n2[2])
    [a1,a2,a3]=add(a1,a2,a3,a4,a5,a6)
    [a4,a5,a6]=mul(v3[0],v3[1],v3[2],n3[0],n3[1],n3[2])
    [a1,a2,a3]=add(a1,a2,a3,a4,a5,a6)
    [a4,a5,a6]=mul(v4[0],v4[1],v4[2],n4[0],n4[1],n4[2])
    [a1,a2,a3]=add(a1,a2,a3,a4,a5,a6)
    [a4,a5,a6]=mul(v5[0],v5[1],v5[2],n5[0],n5[1],n5[2])
    [a1,a2,a3]=add(a1,a2,a3,a4,a5,a6)
    [a4,a5,a6]=mul(v6[0],v6[1],v6[2],n6[0],n6[1],n6[2])
    [a1,a2,a3]=add(a1,a2,a3,a4,a5,a6)
    val=np.array([a1,a2,a3])
    return val

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list dot_product(np.ndarray[DTYPE_int_t, ndim=1] a1,
                        np.ndarray[DTYPE_int_t, ndim=1] a2,
                        np.ndarray[DTYPE_int_t, ndim=1] a3,
                        np.ndarray[DTYPE_int_t, ndim=1] b1,
                        np.ndarray[DTYPE_int_t, ndim=1] b2,
                        np.ndarray[DTYPE_int_t, ndim=1] b3):
    # product of vectors A and B
    #
    # vector A
    # Ax=(a1[0]+a1[1]*sin)/a1[2]
    # Ay=(a2[0]+a2[1]*sin)/a2[2]
    # Az=(a3[0]+a3[1]*sin)/a3[2]
    #
    # vector B
    # Bx=(b1[0]+b1[1]*sin)/b1[2]
    # By=(b2[0]+b2[1]*sin)/b2[2]
    # Bz=(b3[0]+b3[1]*sin)/b3[2]
    #    
    # return:
    # A*B = Ax*Bx + Ay*By + Az*Bz
    #     = (t1+t2*SIN)/t3
    cdef DTYPE_int_t t1,t2,t3,t4,t5,t6
    [t1,t2,t3]=mul(a1[0],a1[1],a1[2],b1[0],b1[1],b1[2])
    [t4,t5,t6]=mul(a2[0],a2[1],a2[2],b2[0],b2[1],b2[2])
    [t1,t2,t3]=add(t1,t2,t3,t4,t5,t6)
    [t4,t5,t6]=mul(a3[0],a3[1],a3[2],b3[0],b3[1],b3[2])
    [t1,t2,t3]=add(t1,t2,t3,t4,t5,t6)
    return [t1,t2,t3]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray outer_product(np.ndarray[DTYPE_int_t, ndim=2] v1,
                                np.ndarray[DTYPE_int_t, ndim=2] v2):
    
    cdef DTYPE_int_t a1,a2,a3,b1,b2,b3,c1,c2,c3,c4,c5,c6,c7,c8,c9
    cdef np.ndarray[DTYPE_int_t, ndim=2] tmp2
    
    [a1,a2,a3]=mul(v1[1][0],v1[1][1],v1[1][2],v2[2][0],v2[2][1],v2[2][2])
    [b1,b2,b3]=mul(v1[2][0],v1[2][1],v1[2][2],v2[1][0],v2[1][1],v2[1][2])
    [c1,c2,c3]=sub(a1,a2,a3,b1,b2,b3)
    #
    [a1,a2,a3]=mul(v1[2][0],v1[2][1],v1[2][2],v2[0][0],v2[0][1],v2[0][2])
    [b1,b2,b3]=mul(v1[0][0],v1[0][1],v1[0][2],v2[2][0],v2[2][1],v2[2][2])
    [c4,c5,c6]=sub(a1,a2,a3,b1,b2,b3)
    #
    [a1,a2,a3]=mul(v1[0][0],v1[0][1],v1[0][2],v2[1][0],v2[1][1],v2[1][2])
    [b1,b2,b3]=mul(v1[1][0],v1[1][1],v1[1][2],v2[0][0],v2[0][1],v2[0][2])
    [c7,c8,c9]=sub(a1,a2,a3,b1,b2,b3)
    #
    tmp2=np.array([[c1,c2,c3],[c4,c5,c6],[c7,c8,c9]])
    return tmp2

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list det_matrix(np.ndarray[DTYPE_int_t, ndim=2] a,
                        np.ndarray[DTYPE_int_t, ndim=2] b,
                        np.ndarray[DTYPE_int_t, ndim=2] c):
    # determinant of 3x3 matrix in TAU style
    cdef DTYPE_int_t t1,t2,t3,t4,t5,t6,t7,t8,t9
    #
    [t1,t2,t3]=mul(a[0][0],a[0][1],a[0][2],b[1][0],b[1][1],b[1][2])
    [t1,t2,t3]=mul(t1,t2,t3,c[2][0],c[2][1],c[2][2])
    #        
    [t4,t5,t6]=mul(a[2][0],a[2][1],a[2][2],b[0][0],b[0][1],b[0][2])
    [t4,t5,t6]=mul(t4,t5,t6,c[1][0],c[1][1],c[1][2])
    #
    [t7,t8,t9]=add(t1,t2,t3,t4,t5,t6)
    #
    [t1,t2,t3]=mul(a[1][0],a[1][1],a[1][2],b[2][0],b[2][1],b[2][2])
    [t1,t2,t3]=mul(t1,t2,t3,c[0][0],c[0][1],c[0][2])
    #
    [t7,t8,t9]=add(t7,t8,t9,t1,t2,t3)
    #
    [t1,t2,t3]=mul(a[2][0],a[2][1],a[2][2],b[1][0],b[1][1],b[1][2])
    [t1,t2,t3]=mul(t1,t2,t3,c[0][0],c[0][1],c[0][2])
    #
    [t7,t8,t9]=sub(t7,t8,t9,t1,t2,t3)
    #
    [t1,t2,t3]=mul(a[1][0],a[1][1],a[1][2],b[0][0],b[0][1],b[0][2])
    [t1,t2,t3]=mul(t1,t2,t3,c[2][0],c[2][1],c[2][2])
    #
    [t7,t8,t9]=sub(t7,t8,t9,t1,t2,t3)
    #
    [t1,t2,t3]=mul(a[0][0],a[0][1],a[0][2],b[2][0],b[2][1],b[2][2])
    [t1,t2,t3]=mul(t1,t2,t3,c[1][0],c[1][1],c[1][2])

    [t7,t8,t9]=sub(t7,t8,t9,t1,t2,t3)
    #
    return [t7,t8,t9]
