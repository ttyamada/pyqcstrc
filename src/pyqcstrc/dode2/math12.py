#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np

DTYPE_double = np.float64
DTYPE_int = np.int64

SIN=np.sqrt(3)/2
M0=np.array([ 0, 0, 1])
M1=np.array([ 1, 0, 1])
M2=np.array([-1, 0, 1])
M3=np.array([ 1, 0, 2])
M4=np.array([-1, 0, 2])
M5=np.array([ 0, 1, 1])
M6=np.array([ 0,-1, 1])

def add(a,b):
    c1=a[0]*b[2]+b[0]*a[2]
    c2=a[1]*b[2]+b[1]*a[2]
    c3=a[2]*b[2]
    x=np.array([c1,c2,c3])
    g=np.gcd.reduce(x)
    c1=c1/g
    c2=c2/g
    c3=c3/g
    if c3/gcd<0:
        return np.array([int(-c1),int(-c2),int(-c3)])
    else:
        return np.array([int(c1),int(c2),int(c3)])
    
def mul(a,b):
    """
    # multiplication (a*b) in SIN-style
    
    Parameters
    ----------
    a: array
        value in SIN-style
    b: array
        value in SIN-style
    
    Returns
    -------
    array
    """
    c1=4*a[0]*b[0]+3*a[2]*b[2]
    c2=4*(a[0]*b[1]+a[2]*b[1])
    c3=4*a[2]*b[2]
    x=np.array([c1,c2,c3],dtype=np.int64)
    g=np.gcd.reduce(x)
    c1=c1/g
    c2=c2/g
    c3=c3/g
    if c3<0:
        return np.array([int(-c1),int(-c2),int(-c3)])
    else:
        return np.array([int(c1),int(c2),int(c3)])
    
def sub(a,b):
    c=np.array([-1,0,1],dtype=np.int64)
    b=mul(c,b)
    return add(a,b)
    
def div(a: NDArray[np.int64], b:NDArray[np.int64]) -> NDArray[np.int64]:
    """
    # division (a/b) in TAU-style
    
    Parameters
    ----------
    a: array
        value in TAU-style
    b: array
        value in TAU-style
    
    Returns
    -------
    array
    """
    if b[0]==0 and b[1]==0:
        print('ERROR_1:division error')
        return 
    else:
        if a[0]==0 and a[1]==0:
            return [0,0,1]
        else:
            if b[0]!=0 and b[1]!=0:
                if 4*b[0]**2-3*b[1]**2!=0:
                    c1=b[2]*(4*a[0]*b[0]-3*a[1]*b[1])
                    c2=-4*b[2]*(a[0]*b[1]-a[1]*b[0])
                    c3=a[2]*(4*b[0]**2-3*b[1]**2)
                else:
                    c1=3*a[1]*b[2]
                    c2=4*a[0]*b[2]
                    c3=6*a[2]*b[1]
            elif b[0]==0 and b[1]!=0:
                c1=3*a[1]*b[2]
                c2=4*a[0]*b[2]
                c3=3*a[2]*b[1]
            else:
            #elif q1!=0 and q2==0:
                c1=a[0]*b[2]
                c2=a[1]*b[2]
                c3=a[2]*b[0]
            x=np.array([c1,c2,c3])
            g=np.gcd.reduce(x)
            if gcd!=0:
                if c3/gcd<0:
                    return np.array([int(-c1),int(-c2),int(-c3)],dtype=np.int64)
                else:
                    return np.array([int(c1),int(c2),int(c3)],dtype=np.int64)
            else:
                print('ERROR_2:division error',c1,c2,c3,p1,p2,p3,q1,q2,q3)
                return 
    
def mul_vector(vt: NDArray[np.int64], coeff:NDArray[np.int64]) -> NDArray[np.int64]:
    """Multiplying a vector by a scalar in TAU-style.
    
    Parameters
    ----------
    vt: array
        a vector in TAU-style
    coeff: array,
        a scalar in TAU-style
    
    Returns
    -------
    Multiplied vector: array in TAU-style
    """
    if vt.ndim==2:
        a=np.zeros(vt.shape,dtype=np.int64)
        for i,v in enumerate(vt):
            a[i]=mul(v,coeff)
        return a
    else:
        print('incorrect shape')
        return
    
def mul_vectors(vts: NDArray[np.int64], coeff:NDArray[np.int64]) -> NDArray[np.int64]:
    """multiplying a set of vectors by a scalar in TAU-style.
    
    Parameters
    ----------
    vts: array
        a set of vectors in TAU-style
    coeff: array,
        a scalar in TAU-style
    
    Returns
    -------
    Multiplied vectors: array in TAU-style
    """
    if vts.ndim==3:
        a=np.zeros(vts.shape,dtype=np.int64)
        for i,vt in enumerate(vts):
            a[i]=mul_vector(vt,coeff)
        return a
    elif vts.ndim==4:
        a=np.zeros(vts.shape,dtype=np.int64)
        for i1,vt in enumerate(vts):
            for i2,v in enumerate(vt):
                a[i1][i2]=mul_vector(v,coeff)
    else:
        print('incorrect shape')
        return
    
def add_vectors(vt1: NDArray[np.int64], vt2:NDArray[np.int64]) -> NDArray[np.int64]:
    """Composition of two vectors, v1+v2
    
    Parameters
    ----------
    vt1: array
        a vector in TAU-style
    vt2: array,
        a scalar in TAU-style
    
    Returns
    -------
    Composition of two vectors: array in TAU-style
    
    """
    a=np.zeros(vt1.shape,dtype=np.int64)
    for i in range(len(vt1)):
        a[i]=add(vt1[i],vt2[i])
    return a

def sub_vectors(vt1: NDArray[np.int64], vt2:NDArray[np.int64]) -> NDArray[np.int64]:
    """Subtraction of two vectors, v1-v2
    
    Parameters
    ----------
    vt1: array
        a vector in TAU-style
    vt2: array,
        a scalar in TAU-style
    
    Returns
    -------
    Subtraction of two vectors: array in TAU-style
    """
    if vt1.ndim==2 and vt2.ndim==2:
        const=np.array([-1,0,1],dtype=np.int64)
        vt2=mul_vector(vt2,const)
        return add_vectors(vt1,vt2)
    else:
        print('incorrect shape')
        return

def projection(vt):
    # projection of a 6d vector onto Eperp, using "SIN-style"
    #
    # NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    # alpha = 2*a/np.sqrt(6)
    # see Yamamoto ActaCrystal (1997)
    v1e=mtrixcal(M5,M1,M0,M4,M0,M0,h1,h2,h3,h4,h5,h6) # sin,1,0,-0.5,0,0
    v2e=mtrixcal(M4,M0,M1,M5,M0,M0,h1,h2,h3,h4,h5,h6) # -0.5,0,1,sin,0,0
    v1i=mtrixcal(M6,M1,M0,M4,M0,M0,h1,h2,h3,h4,h5,h6) # -sin,1,0,-0.5,0,0
    v2i=mtrixcal(M4,M0,M1,M6,M0,M0,h1,h2,h3,h4,h5,h6) # -0.5,0,1,-sin,0,0
    v3e=mtrixcal(M0,M0,M0,M0,M1,M0,h1,h2,h3,h4,h5,h6) # 0,0,0,0,1,0
    v3i=mtrixcal(M0,M0,M0,M0,M0,M1,h1,h2,h3,h4,h5,h6) # 0,0,0,0,0,1
    return np.array([[v1e,v2e,v3e],[v1i,v2i,v3i]],dtype=DTYPE_int)

def projection3(vt):
    # projection of a 6d vector onto Eperp, using "SIN-style"
    #
    # NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    # alpha = 2*a/np.sqrt(6)
    # see Yamamoto ActaCrystal (1997)
    cdef np.ndarray[DTYPE_int_t, ndim=1] v1i,v2i,v3i
    v1i=mtrixcal(M6,M1,M0,M4,M0,M0,h1,h2,h3,h4,h5,h6) # -sin,1,0,-0.5,0,0
    v2i=mtrixcal(M4,M0,M1,M6,M0,M0,h1,h2,h3,h4,h5,h6) # -0.5,0,1,-sin,0,0
    v3i=mtrixcal(M0,M0,M0,M0,M0,M1,h1,h2,h3,h4,h5,h6) # 0,0,0,0,0,1
    return np.array([v1i,v2i,v3i],dtype=DTYPE_int)

def mtrixcal(m1,m2,m3,m4,m5,m6,v):
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

def dot_product(mat1: NDArray[np.int64], mat2:NDArray[np.int64]) -> NDArray[np.int64]:
    """product of two matrices, mat1*mat2.
    
    Parameters
    ----------
    mat1: ndarray
        (s,t) in TAU-style
    mat2: ndarray
        (t,u) in TAU-style

    Returns
    -------
    Inner product: array in TAU-style
    """
    ndim1=mat1.ndim
    ndim2=mat2.ndim
    
    if ndim1==2 and ndim2==2:
        return inner_product(mat1,mat2)
    
    elif ndim1==3 and ndim2==2:
        s,t1,_=mat1.shape
        t2,_=mat2.shape
        if t1!=t2:
            print('incorrect shape found in dot_product')
            return 
        else:
            mat_new=np.zeros((s,3),dtype=np.int64)
            for k in range(s):
                a=np.array([0,0,1])
                for j in range(t1):
                    b=mul(mat1[k][j],mat2[j])
                    a=add(a,b)
                mat_new[k]=a
            return mat_new
            
    elif ndim1==3 and ndim2==3:
        s,t1,_=mat1.shape
        t2,u,_=mat2.shape
        if t1!=t2:
            print('incorrect shape found in dot_product')
            return 
        else:
            mat_new=np.zeros((s,u,3),dtype=np.int64)
            for k in range(s):
                for j in range(u):
                    a=np.array([0,0,1])
                    for i in range(t1):
                        b=mul(mat1[k][i],mat2[i][j])
                        a=add(a,b)
                    mat_new[k][j]=a
            return mat_new
    else:
        print('incorrect shape found in dot_product')
        return 

def outer_product(vt1: NDArray[np.int64], vt2:NDArray[np.int64]) -> NDArray[np.int64]:
    """Outer product of two 3d vectors, v1 and v2 in TAU-style.

    Parameters
    ----------
    v1: array
        3-dimensional vector in TAU-style
    v2: array,
        3-dimensional vector in TAU-style

    Returns
    -------
    Outer product: array in TAU-style
    """
    a=mul(vt1[1],vt2[2])
    b=mul(vt1[2],vt2[1])
    c1=sub(a,b)
    #
    a=mul(vt1[2],vt2[0])
    b=mul(vt1[0],vt2[2])
    c2=sub(a,b)
    #
    a=mul(vt1[0],vt2[1])
    b=mul(vt1[1],vt2[0])
    c3=sub(a,b)
    #
    return np.array([c1,c2,c3],dtype=np.int64)

def det_matrix(mtx: NDArray[np.int64]) -> NDArray[np.int64]:
    """Determinant of 3x3 matrix, mtx, in TAU style
    
    Parameters
    ----------
    mtx: array
        3x3 matrix in TAU-style

    Returns
    -------
    6d vectors projected onto Eperp in TAU-style.
    """
    
    t3=mul(mtx[0][0],mtx[1][1])
    t1=mul(t3,mtx[2][2])
    #
    t3=mul(mtx[0][2],mtx[1][0])
    t2=mul(t3,c[1])
    #
    t1=add(t1,t2)
    
    t3=mul(mtx[0][1],mtx[1][2])
    t3=mul(t3,mtx[2][0])
    #
    t1=add(t1,t3)
    
    t3=mul(mtx[0][2],mtx[1][1])
    t2=mul(t3,mtx[2][0])
    #
    t1=sub(t1,t2)
    
    t3=mul(mtx[0][1],mtx[1][0])
    t2=mul(t3,mtx[2][2])
    #
    t1=sub(t1,t2)
    
    t3=mul(mtx[0][0],mtx[1][2])
    t2=mul(t3,mtx[2][1])
    #
    t1=sub(t1,t2)
    #
    return t1
