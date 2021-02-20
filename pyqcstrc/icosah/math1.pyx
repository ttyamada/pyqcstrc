import sys
import numpy as np
cimport numpy as np
cimport cython

DTYPE_double = np.float64
DTYPE_int = int

cdef np.float64_t TAU=(1+np.sqrt(5))/2.0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray centroid(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron):
    # geometric center, centroid of tetrahedron
    cdef int i1,i2
    cdef DTYPE_int_t v1,v2,v3
    
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    tmp1a=np.array([0])
    for i2 in range(6):
        v1,v2,v3=0,0,1
        for i1 in range(4):
            v1,v2,v3=add(v1,v2,v3,tetrahedron[i1][i2][0],tetrahedron[i1][i2][1],tetrahedron[i1][i2][2])
        v1,v2,v3=mul(v1,v2,v3,1,0,4)
        if len(tmp1a)!=1:
            tmp1a=np.append(tmp1a,[v1,v2,v3])
        else:
            tmp1a=np.array([v1,v2,v3])
    return tmp1a.reshape(6,3)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray outer_product(np.ndarray[DTYPE_int_t, ndim=2] v1,np.ndarray[DTYPE_int_t, ndim=2] v2):
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
cpdef list inner_product(np.ndarray[DTYPE_int_t, ndim=2] v1,np.ndarray[DTYPE_int_t, ndim=2] v2):
    cdef DTYPE_int_t a1,a2,a3,b1,b2,b3,c1,c2,c3
    [a1,a2,a3]=mul(v1[0][0],v1[0][1],v1[0][2],v2[0][0],v2[0][1],v2[0][2])
    [b1,b2,b3]=mul(v1[1][0],v1[1][1],v1[1][2],v2[1][0],v2[1][1],v2[1][2])
    [c1,c2,c3]=mul(v1[2][0],v1[2][1],v1[2][2],v2[2][0],v2[2][1],v2[2][2])
    [a1,a2,a3]=add(a1,a2,a3,b1,b2,b3)
    [a1,a2,a3]=add(a1,a2,a3,c1,c2,c3)
    return [a1,a2,a3]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int coplanar_check(np.ndarray[DTYPE_int_t, ndim=3] point):
    # coplanar check
    cdef np.ndarray[DTYPE_int_t, ndim=1] x0e,y0e,z0e,x0i,y0i,z0i
    cdef np.ndarray[DTYPE_int_t, ndim=1] x1e,y1e,z1e,x1i,y1i,z1i
    cdef np.ndarray[DTYPE_int_t, ndim=1] x2e,y2e,z2e,x2i,y2i,z2i
    cdef np.ndarray[DTYPE_int_t, ndim=1] x3e,y3e,z3e,x3i,y3i,z3i
    cdef DTYPE_int_t a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3
    cdef int i1,flag
    cdef np.ndarray[DTYPE_int_t, ndim=2] v1,v2,v3,v4
    if len(point)>3:
        x0e,y0e,z0e,x0i,y0i,z0i=projection(point[0][0],point[0][1],point[0][2],point[0][3],point[0][4],point[0][5])
        x1e,y1e,z1e,x1i,y1i,z1i=projection(point[1][0],point[1][1],point[1][2],point[1][3],point[1][4],point[1][5])
        x2e,y2e,z2e,x2i,y2i,z2i=projection(point[2][0],point[2][1],point[2][2],point[2][3],point[2][4],point[2][5])
        [a1,a2,a3]=sub(x1i[0],x1i[1],x1i[2],x0i[0],x0i[1],x0i[2]) # e1
        [b1,b2,b3]=sub(y1i[0],y1i[1],y1i[2],y0i[0],y0i[1],y0i[2])
        [c1,c2,c3]=sub(z1i[0],z1i[1],z1i[2],z0i[0],z0i[1],z0i[2])
        v1=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
        [a1,a2,a3]=sub(x2i[0],x2i[1],x2i[2],x0i[0],x0i[1],x0i[2]) # e2
        [b1,b2,b3]=sub(y2i[0],y2i[1],y2i[2],y0i[0],y0i[1],y0i[2])
        [c1,c2,c3]=sub(z2i[0],z2i[1],z2i[2],z0i[0],z0i[1],z0i[2])
        v2=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
        v3=outer_product(v1,v2)
        flag=0
        for i1 in range(3,len(point)):
            x3e,y3e,z3e,x3i,y3i,z3i=projection(point[i1][0],point[i1][1],point[i1][2],point[i1][3],point[i1][4],point[i1][5])
            [a1,a2,a3]=sub(x3i[0],x3i[1],x3i[2],x0i[0],x0i[1],x0i[2])
            [b1,b2,b3]=sub(y3i[0],y3i[1],y3i[2],y0i[0],y0i[1],y0i[2])
            [c1,c2,c3]=sub(z3i[0],z3i[1],z3i[2],z0i[0],z0i[1],z0i[2])
            v4=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
            [d1,d2,d3]=inner_product(v3,v4)
            if d1==0 and d2==0:
                flag+=0
            else:
                flag+=1
                break
        if flag==0:
            return 1 # coplanar
        else:
            return 0
    else:
        return 1 # coplanar

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list projection(np.ndarray[DTYPE_int_t, ndim=1] h1,
                    np.ndarray[DTYPE_int_t, ndim=1] h2,
                    np.ndarray[DTYPE_int_t, ndim=1] h3,
                    np.ndarray[DTYPE_int_t, ndim=1] h4,
                    np.ndarray[DTYPE_int_t, ndim=1] h5,
                    np.ndarray[DTYPE_int_t, ndim=1] h6):
    # projection of a 6d vector onto Epar and Eperp, using "TAU-style"
    #
    # NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    # alpha = a/np.sqrt(2.0+TAU)
    # see Yamamoto ActaCrystal (1997)
    cdef np.ndarray[DTYPE_int_t, ndim=1] m1,m2,m3,m4
    cdef np.ndarray[DTYPE_int_t, ndim=1] v1e,v2e,v3e,v1i,v2i,v3i
    m0=np.array([ 0, 0, 1]) #  0 in 'TAU-style'
    m1=np.array([ 1, 0, 1]) #  1
    m2=np.array([-1, 0, 1]) # -1
    m3=np.array([ 0, 1, 1]) #  tau
    m4=np.array([ 0,-1, 1]) # -tau
    v1e=mtrixcal(m1,m3,m3,m0,m2,m0,h1,h2,h3,h4,h5,h6) # 1,tau,tau,0,-1,0
    v2e=mtrixcal(m3,m0,m0,m1,m3,m1,h1,h2,h3,h4,h5,h6) # tau,0,0,1,TAU,1
    v3e=mtrixcal(m0,m1,m2,m4,m0,m3,h1,h2,h3,h4,h5,h6) # 0,1,-1,-tau,0,tau
    v1i=mtrixcal(m3,m2,m2,m0,m4,m0,h1,h2,h3,h4,h5,h6) # tau,-1,-1,0,-tau,0
    v2i=mtrixcal(m2,m0,m0,m3,m2,m3,h1,h2,h3,h4,h5,h6) # -1,0,0,tau,-1,tau
    v3i=mtrixcal(m0,m3,m4,m1,m0,m2,h1,h2,h3,h4,h5,h6) # 0,tau,-tau,1,0,-1
    return [v1e,v2e,v3e,v1i,v2i,v3i]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list projection_perp(np.ndarray[DTYPE_int_t, ndim=1] n1,
                            np.ndarray[DTYPE_int_t, ndim=1] n2,
                            np.ndarray[DTYPE_int_t, ndim=1] n3,
                            np.ndarray[DTYPE_int_t, ndim=1] n4,
                            np.ndarray[DTYPE_int_t, ndim=1] n5,
                            np.ndarray[DTYPE_int_t, ndim=1] n6):
    # This returns 6D indeces of a projection of 6D vector (n1,n2,n3,n4,n5,n6) onto Eperp
    # Direct lattice vector
    cdef np.ndarray[DTYPE_int_t, ndim=1] m1,m2,m3
    cdef np.ndarray[DTYPE_int_t, ndim=1] h1,h2,h3,h4,h5,h6
    m1=np.array([ 1, 0, 2]) #  (TAU+2)/2/(2+TAU)=1/2 in 'TAU-style'
    m2=np.array([-1, 2,10]) #  TAU/2/(2+TAU)
    m3=np.array([ 1,-2,10]) # -TAU/2/(2+TAU)
    h1=mtrixcal(m1,m3,m3,m3,m3,m3,n1,n2,n3,n4,n5,n6) # (tau+2,-tau,-tau,-tau,-tau,-tau)/2
    h2=mtrixcal(m3,m1,m3,m2,m2,m3,n1,n2,n3,n4,n5,n6) # (-tau,tau+2,-tau,tau,tau,-tau)/2
    h3=mtrixcal(m3,m3,m1,m3,m2,m2,n1,n2,n3,n4,n5,n6) # (-tau,-tau,tau+2,-tau,tau,tau)/2
    h4=mtrixcal(m3,m2,m3,m1,m3,m2,n1,n2,n3,n4,n5,n6) # (-tau,tau,-tau,tau+2,-tau,tau)/2
    h5=mtrixcal(m3,m2,m2,m3,m1,m3,n1,n2,n3,n4,n5,n6) # (-tau,tau,tau,-tau,tau+2,-tau)/2
    h6=mtrixcal(m3,m3,m2,m2,m3,m1,n1,n2,n3,n4,n5,n6) # (-tau,-tau,tau,tau,-tau,tau+2)/2
    return [h1,h2,h3,h4,h5,h6]
    #const=1/(2.0+TAU)
    #m1 =((TAU+2)*n1 -     TAU*n2 -     TAU*n3 -     TAU*n4 -     TAU*n5 -     TAU*n6)/2.0*const
    #m2 = ( - TAU*n1 + (TAU+2)*n2 -     TAU*n3 +     TAU*n4 +     TAU*n5 -     TAU*n6)/2.0*const
    #m3 = ( - TAU*n1 -     TAU*n2 + (TAU+2)*n3 -     TAU*n4 +     TAU*n5 +     TAU*n6)/2.0*const
    #m4 = ( - TAU*n1 +     TAU*n2 -     TAU*n3 + (TAU+2)*n4 -     TAU*n5 +     TAU*n6)/2.0*const
    #m5 = ( - TAU*n1 +     TAU*n2 +     TAU*n3 -     TAU*n4 + (TAU+2)*n5 -     TAU*n6)/2.0*const
    #m6 = ( - TAU*n1 -     TAU*n2 +     TAU*n3 +     TAU*n4 -     TAU*n5 + (TAU+2)*n6)/2.0*const
    #return m1,m2,m3,m4,m5,m6

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
    # Ax=(a1[0]+a1[1]*tau)/a1[2]
    # Ay=(a2[0]+a2[1]*tau)/a2[2]
    # Az=(a3[0]+a3[1]*tau)/a3[2]
    #
    # vector B
    # Bx=(b1[0]+b1[1]*tau)/b1[2]
    # By=(b2[0]+b2[1]*tau)/b2[2]
    # Bz=(b3[0]+b3[1]*tau)/b3[2]
    #    
    # return:
    # A*B = Ax*Bx + Ay*By + Az*Bz
    #     = (t1+t2*TAU)/t3
    cdef DTYPE_int_t t1,t2,t3,t4,t5,t6
    [t1,t2,t3]=mul(a1[0],a1[1],a1[2],b1[0],b1[1],b1[2])
    [t4,t5,t6]=mul(a2[0],a2[1],a2[2],b2[0],b2[1],b2[2])
    [t1,t2,t3]=add(t1,t2,t3,t4,t5,t6)
    [t4,t5,t6]=mul(a3[0],a3[1],a3[2],b3[0],b3[1],b3[2])
    [t1,t2,t3]=add(t1,t2,t3,t4,t5,t6)
    return [t1,t2,t3]

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

@cython.boundscheck(False)
@cython.wraparound(False)
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list sub(DTYPE_int_t p1,DTYPE_int_t p2,DTYPE_int_t p3,DTYPE_int_t q1,DTYPE_int_t q2,DTYPE_int_t q3): # A-B
    cdef DTYPE_int_t c1,c2,c3,d1,d2,d3,gcd
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list mul(DTYPE_int_t p1,DTYPE_int_t p2,DTYPE_int_t p3,DTYPE_int_t q1,DTYPE_int_t q2,DTYPE_int_t q3): # A*B
    cdef DTYPE_int_t c1,c2,c3,d1,d2,d3,gcd
    cdef np.ndarray[DTYPE_int_t,ndim=1] x
    c1=p1*q1+p2*q2
    c2=p1*q2+p2*q1+p2*q2
    c3=p3*q3
    x=np.array([c1,c2,c3])
    gcd=np.gcd.reduce(x)
    if c3/gcd<0:
        return [int(-c1/gcd),int(-c2/gcd),int(-c3/gcd)]
    else:
        return [int(c1/gcd),int(c2/gcd),int(c3/gcd)]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list div(DTYPE_int_t p1,DTYPE_int_t p2,DTYPE_int_t p3,DTYPE_int_t q1,DTYPE_int_t q2,DTYPE_int_t q3): # A/B
    cdef DTYPE_int_t gcd,c1,c2,c3
    cdef np.ndarray[DTYPE_int_t,ndim=1] x
    if q1==0 and q2==0:
        print('ERROR_1:division error')
        return 1
    else:
        if p1==0 and p2==0:
            return [0,0,1]
        else:
            if q2!=0:
                if q1!=0:
                    c1=(p1*q1 + p1*q2 - p2*q2)*q3
                    c2=(p2*q1 - p1*q2)*q3
                    c3=(q1*q1 - q2*q2 + q1*q2)*p3
                else:
                    c1=(-p1+p2)*q3
                    c2=p1*q3
                    c3=q2*p3
            elif q1!=0:
                c1=p1*q3
                c2=p2*q3
                c3=q1*p3
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double triangle_area(np.ndarray[DTYPE_int_t, ndim=2] v1,
                            np.ndarray[DTYPE_int_t, ndim=2] v2,
                            np.ndarray[DTYPE_int_t, ndim=2] v3):
    cdef double vx0,vx1,vx2,vy0,vy1,vy2,vz0,vz1,vz2,area
    #cdef int a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3
    cdef np.ndarray[np.float64_t, ndim=1] vec1,vec2,vec3
    cdef np.ndarray[DTYPE_int_t, ndim=1] x0,x1,x2,y0,y1,y2,z0,z1,z2
    x0=v1[0]
    y0=v1[1]
    z0=v1[2]
    #
    x1=v2[0]
    y1=v2[1]
    z1=v2[2]
    #
    x2=v3[0]
    y2=v3[1]
    z2=v3[2]
    #
    vx0=(x0[0]+x0[1]*TAU)/float(x0[2])
    vx1=(x1[0]+x1[1]*TAU)/float(x1[2])
    vx2=(x2[0]+x2[1]*TAU)/float(x2[2])
    vy0=(y0[0]+y0[1]*TAU)/float(y0[2])
    vy1=(y1[0]+y1[1]*TAU)/float(y1[2])
    vy2=(y2[0]+y2[1]*TAU)/float(y2[2])
    vz0=(z0[0]+z0[1]*TAU)/float(z0[2])
    vz1=(z1[0]+z1[1]*TAU)/float(z1[2])
    vz2=(z2[0]+z2[1]*TAU)/float(z2[2])
    vec1=np.array([vx1-vx0,vy1-vy0,vz1-vz0])
    vec2=np.array([vx2-vx0,vy2-vy0,vz2-vz0])
    
    vec3=np.cross(vec2,vec1) # cross product
    area=np.sqrt(vec3[0]**2+vec3[1]**2+vec3[2]**2)/2.0
    #area=np.cross(vec2,vec1)/2.0  # cross product
    area=abs(area)
    return area
