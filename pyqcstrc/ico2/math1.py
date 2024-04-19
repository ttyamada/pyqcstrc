#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
sys.path.append('..')
import numericalc
import numpy as np

TAU=(1+np.sqrt(5))/2.0

def add(a,b):
    """
    # summation (a+b) in TAU-style
    
    Parameters
    ----------
    a: array
        value in TAU-style
    b: array
        value in TAU-style
    
    Returns
    -------
    float
    """
    c1=a[0]*b[2]+b[0]*a[2]
    c2=a[1]*b[2]+b[1]*a[2]
    c3=a[2]*b[2]
    x=np.array([c1,c2,c3],dtype=np.int64)
    g=np.gcd.reduce(x)
    c1=c1/g
    c2=c2/g
    c3=c3/g
    if c3<0:
        return np.array([int(-c1),int(-c2),int(-c3)])
    else:
        return np.array([int(c1),int(c2),int(c3)])

def mul(a,b): 
    """
    # multiplication (a*b) in TAU-style
    
    Parameters
    ----------
    a: array
        value in TAU-style
    b: array
        value in TAU-style
    
    Returns
    -------
    float
    """
    c1=a[0]*b[0]+a[1]*b[1]
    c2=a[0]*b[1]+a[1]*b[0]+a[1]*b[1]
    c3=a[2]*b[2]
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
    """
    # subtraction (a/b) in TAU-style
    
    Parameters
    ----------
    a: array
        value in TAU-style
    b: array
        value in TAU-style
    
    Returns
    -------
    float
    """
    c=np.array([-1,0,1],dtype=np.int64)
    b=mul(c,b)
    return add(a,b)

def div(a,b):
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
    float
    """
    if np.all(b[:2]==0):
        print('ERROR_1:division error')
        return 
    else:
        if np.all(a[:2]==0):
            return np.array([0,0,1],dtype=np.int64)
        else:
            if b[1]!=0:
                if b[0]!=0:
                    c1=(a[0]*b[0] + a[0]*b[1] - a[1]*b[1])*b[2]
                    c2=(a[1]*b[0] - a[0]*b[1]            )*b[2]
                    c3=(b[0]*b[0] - b[1]*b[1] + b[0]*b[1])*a[2]
                else:
                    c1=(-a[0]+a[1])*b[2]
                    c2=        a[0]*b[2]
                    c3=        b[1]*a[2]
            else:
                c1=a[0]*b[2]
                c2=a[1]*b[2]
                c3=b[0]*a[2]
            x=np.array([c1,c2,c3],dtype=np.int64)
            g=np.gcd.reduce(x)
            if g!=0:
                c1=c1/g
                c2=c2/g
                c3=c3/g
                if c3<0:
                    return np.array([int(-c1),int(-c2),int(-c3)],dtype=np.int64)
                else:
                    return np.array([int(c1),int(c2),int(c3)],dtype=np.int64)
            else:
                print('ERROR_2:division error')
                return 

numericalc.numericalc.numeric_value(def add_vectors(v1,v2):
    """Composition of two vectors, v1+v2
    
    Parameters
    ----------
    v1: array
        a vector in TAU-style
    v2: array,
        a scalar in TAU-style
    
    Returns
    -------
    Composition of two vectors: array in TAU-style
    
    """
    for i in range(len(v1)):
        v1[i]=add(v1[i],v2[i])
    return v1

def sub_vectors(v1,v2):
    """Subtraction of two vectors, v1-v2
    
    Parameters
    ----------
    v1: array
        a vector in TAU-style
    v2: array,
        a scalar in TAU-style
    
    Returns
    -------
    Subtraction of two vectors: array in TAU-style
    """
    const=np.array([-1,0,1],dtype=np.int64)
    v2=mul_vector(v2,const)
    return add_vectors(v1,v2)

def mul_vector(v,a):
    """Multiplying a vector by a scalar in TAU-style.

    Parameters
    ----------
    vecs: array
        a vector in TAU-style
    a: array,
        a scalar in TAU-style
    
    Returns
    -------
    Multiplied vector: array in TAU-style
    """
    for i in range(len(v)):
        v[i]=mul(v[i],a)
    return v

def mul_vectors(vecs,a):
    """multiplying a set of vectors by a scalar in TAU-style.

    Parameters
    ----------
    vecs: array
        a set of vectors in TAU-style
    a: array,
        a scalar in TAU-style
    
    Returns
    -------
    Multiplied vectors: array in TAU-style
    """
    for i in range(len(vecs)):
        vecs[i]=mul_vector(vecs[i],a)
    return vecs

def outer_product(v1,v2):
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
    a=mul(v1[1],v2[2])
    b=mul(v1[2],v2[1])
    c1=sub(a,b)
    #
    a=mul(v1[2],v2[0])
    b=mul(v1[0],v2[2])
    c2=sub(a,b)
    #
    a=mul(v1[0],v2[1])
    b=mul(v1[1],v2[0])
    c3=sub(a,b)
    #
    return np.array([c1,c2,c3],dtype=np.int64)

def inner_product(v1,v2):
    """Inner product of two 3d vectors, v1 and v2 in TAU-style.

    Parameters
    ----------
    v1: array
        3-dimensional vector in TAU-style
    v2: array,
        3-dimensional vector in TAU-style

    Returns
    -------
    Inner product: array in TAU-style
    """
    a=mul(v1[0],v2[0])
    b=mul(v1[1],v2[1])
    c=mul(v1[2],v2[2])
    a=add(a,b)
    return add(a,c)

def dot_product(v1,v2):
    """product of two 3d vectors, v1 and v2. (same as inner_product???)
    
    Parameters
    ----------
    v1: array
        3-dimensional vector in TAU-style
    v2: array,
        3-dimensional vector in TAU-style

    Returns
    -------
    Inner product: array in TAU-style
    """
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
    #a=mul(1[0],v2[0])
    #b=mul(1[1],v2[1])
    #c=mul(1[2],v2[2])
    #a=add(t1,t2)
    #return add(a,c)
    return inner_product(v1,v2)






def projection(v):
    """projection of a 6d vector onto Epar and Eperp in "TAU-style"
    NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    alpha = a/np.sqrt(2.0+TAU)
    see Yamamoto ActaCrystal (1997)

    Parameters
    ----------
    v: array
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
    v1e=mtrixcal(m1,m3,m3,m0,m2,m0,v) # 1,tau,tau,0,-1,0
    v2e=mtrixcal(m3,m0,m0,m1,m3,m1,v) # tau,0,0,1,TAU,1
    v3e=mtrixcal(m0,m1,m2,m4,m0,m3,v) # 0,1,-1,-tau,0,tau
    v1i=mtrixcal(m3,m2,m2,m0,m4,m0,v) # tau,-1,-1,0,-tau,0
    v2i=mtrixcal(m2,m0,m0,m3,m2,m3,v) # -1,0,0,tau,-1,tau
    v3i=mtrixcal(m0,m3,m4,m1,m0,m2,v) # 0,tau,-tau,1,0,-1
    return np.array([[v1e,v2e,v3e],[v1i,v2i,v3i]],dtype=np.int64)

def projection3(v):
    """projection of a 6d vector onto Eperp in "TAU-style"
    NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    alpha = a/np.sqrt(2.0+TAU)
    see Yamamoto ActaCrystal (1997)
    
    Parameters
    ----------
    v: array
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
    v1i=mtrixcal(m3,m2,m2,m0,m4,m0,v) # tau,-1,-1,0,-tau,0
    v2i=mtrixcal(m2,m0,m0,m3,m2,m3,v) # -1,0,0,tau,-1,tau
    v3i=mtrixcal(m0,m3,m4,m1,m0,m2,v) # 0,tau,-tau,1,0,-1
    return np.array([v1i,v2i,v3i],dtype=np.int64)

def projection_perp(v):
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
    h1=mtrixcal(m1,m3,m3,m3,m3,m3,v) # (tau+2,-tau,-tau,-tau,-tau,-tau)/2
    h2=mtrixcal(m3,m1,m3,m2,m2,m3,v) # (-tau,tau+2,-tau,tau,tau,-tau)/2
    h3=mtrixcal(m3,m3,m1,m3,m2,m2,v) # (-tau,-tau,tau+2,-tau,tau,tau)/2
    h4=mtrixcal(m3,m2,m3,m1,m3,m2,v) # (-tau,tau,-tau,tau+2,-tau,tau)/2
    h5=mtrixcal(m3,m2,m2,m3,m1,m3,v) # (-tau,tau,tau,-tau,tau+2,-tau)/2
    h6=mtrixcal(m3,m3,m2,m2,m3,m1,v) # (-tau,-tau,tau,tau,-tau,tau+2)/2
    return np.array([h1,h2,h3,h4,h5,h6],dtype=np.int64)
    #const=1/(2.0+TAU)
    #m1 =((TAU+2)*n1 -     TAU*n2 -     TAU*n3 -     TAU*n4 -     TAU*n5 -     TAU*n6)/2.0*const
    #m2 = ( - TAU*n1 + (TAU+2)*n2 -     TAU*n3 +     TAU*n4 +     TAU*n5 -     TAU*n6)/2.0*const
    #m3 = ( - TAU*n1 -     TAU*n2 + (TAU+2)*n3 -     TAU*n4 +     TAU*n5 +     TAU*n6)/2.0*const
    #m4 = ( - TAU*n1 +     TAU*n2 -     TAU*n3 + (TAU+2)*n4 -     TAU*n5 +     TAU*n6)/2.0*const
    #m5 = ( - TAU*n1 +     TAU*n2 +     TAU*n3 -     TAU*n4 + (TAU+2)*n5 -     TAU*n6)/2.0*const
    #m6 = ( - TAU*n1 -     TAU*n2 +     TAU*n3 +     TAU*n4 -     TAU*n5 + (TAU+2)*n6)/2.0*const
    #return m1,m2,m3,m4,m5,m6

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

def centroid(tetrahedron):
    """geometric center, centroid of tetrahedron, in TAU-style.

    Parameters
    ----------
    tetrahedron: array
        6-dimensional vector in TAU-style
    
    Returns
    -------
    centroid: array in TAU-style
    """
    
    v0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    for i2 in range(6):
        v2=v0[i2]
        for i1 in range(4):
            v1=add(v1,tetrahedron[i1][i2])
        v2[i2]=mul(v1,np.array([1,0,4]))
    return v2

def coplanar_check(p):
    """Check whether a given set of points (in TAU-style) is coplanar or not.
    
    Parameters
    ----------
    p: array
        a set of pointsin TAU-style.

    Returns
    -------
    int
    #bool
    """
    if len(p)>3:
        xyz0i=projection3(p[0])
        xyz1i=projection3(p[1])
        xyz2i=projection3(p[2])
        v1=sub_vectors(xyz1i,xyz0i)
        v2=sub_vectors(xyz2i,xyz0i)
        v3=outer_product(v1,v2)
        flag=0
        for i1 in range(3,len(p)):
            xyz3i=projection3(p[i1])
            v4=sub_vectors(xyz3i,xyz0i)
            
            d=inner_product(v3,v4)
            if np.all(d[:2])==0:
                flag+=0
            else:
                flag+=1
                break
        if flag==0:
            return 1 # coplanar
            #return True # coplanar
        else:
            return 0
            #return False
    else:
        return 1 # coplanar
        #return True # coplanar

# これは必要か？　行列式？
def det_matrix(mtx):
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







def triangle_area(a):
    """Numerial calcuration of area of given triangle, a.
    The coordinates of the tree vertecies of the triangle are given in TAU-style.
    
    Parameters
    ----------
    a: array containing 3-dimensional coordinates of tree vertecies of a triangle (a) in TAU-style
    
    Returns
    -------
    area of given triangle: float
    """
    
    vx1=numericalc.numeric_value(a[1][0])-numericalc.numeric_value(a[0][0])
    vy1=numericalc.numeric_value(a[1][1])-numericalc.numeric_value(a[0][1])
    vz1=numericalc.numeric_value(a[1][2])-numericalc.numeric_value(a[0][2])
    
    vx2=numericalc.numeric_value(a[2][0])-numericalc.numeric_value(a[0][0])
    vy2=numericalc.numeric_value(a[2][1])-numericalc.numeric_value(a[0][1])
    vz2=numericalc.numeric_value(a[2][2])-numericalc.numeric_value(a[0][2])
    
    v1=np.array([vx1,vy1,vz1])
    v2=np.array([vx2,vy2,vz2])
    
    v3=np.cross(v2,v1) # cross product
    return np.sqrt(np.sum(np.abs(v3**2)))/2.0

if __name__ == '__main__':
    
    import random
    
    ncycle=20
    eps=1e-3
    
    def math_check(a,b):
        """checking basic arithmetic operations in TAU-style.
        """
        flg=0
        a1=numericalc.numeric_value(a)
        b1=numericalc.numeric_value(b)
        
        c=add(a,b)
        c1=numericalc.numeric_value(c)
        c2=a1+b1
        if abs(c1-c2)<eps:
            flg+=1
        else:
            print('+')
            
        c=sub(a,b)
        c1=numericalc.numeric_value(c)
        c2=a1-b1
        if abs(c1-c2)<eps:
            flg+=1
        else:
            print('-')
            
        c=mul(a,b)
        c1=numericalc.numeric_value(c)
        c2=a1*b1
        if abs(c1-c2)<eps:
            flg+=1
        else:
            print('*')
            
        c=div(a,b)
        c1=numericalc.numeric_value(c)
        c2=a1/b1
        if abs(c1-c2)<eps:
            flg+=1
        else:
            print('/')
            
        if flg==4:
            return 0
        else:
            print(a,b)
            return 1
    
    def generate_random_value():
        """ generate value in TAU-style
        """
        nmax=10
        v=np.zeros((3),dtype=np.int64)
        for i1 in range(2):
            v[i1]=random.randrange(-nmax,nmax) # a and b in (a+b*TAU)/c.
        v[2]=random.randrange(1,nmax) # c in (a+b*TAU)/c.
        return v
        
    def generate_random_vector(ndim=6):
        """ generate ndim vector in TAU-style
        ndim: dimension of vectors
        """
        nmax=10
        v=np.zeros((ndim,3), dtype=np.int64)
        for i1 in range(ndim):
            v[i1]=generate_random_value()
        return v
        
    def generate_random_vectors(n,ndim=6):
        """
        num: number of generated vectors.
        ndim: dimension of vectors
        """
        v=np.zeros((n,ndim,3), dtype=np.int64)
        for i1 in range(n):
            v[i1]=generate_random_vector(ndim)
        return v
    
    #-----------------------------------
    # check basic arithmetic operations
    #-----------------------------------
    flg=0
    for _ in range(ncycle):
        a=generate_random_value()
        b=generate_random_value()
        flg+=math_check(a,b)
    if flg==0:
        print('math_check: Correct!')
    else:
        print('math_check: Wrong')
    
    #-----------------------------------
    # check operations on vectors
    #-----------------------------------
    
    # 積：定数xベクトル
    flg=0
    const=np.array([1,1,2])
    nconst=numericalc.numeric_value(const)
    for _ in range(ncycle):
        v1=generate_random_vector()
        nv1=numericalc.numerical_vector(v1)
        a=nv1*nconst
        v=mul_vector(v1,const)
        b=numericalc.numerical_vector(v)
        if np.allclose(a,b):
            pass
        else:
            flg+=0
    if flg==0:
        print('mul_vector: Correct!')
    else:
        print('mul_vector: Wrong')
        
    # 積：定数xベクトルのセット
    nset=5
    flg=0
    const=np.array([1,1,2])
    nconst=numericalc.numeric_value(const)
    for _ in range(ncycle):
        vs=generate_random_vectors(nset)
        mvs=mul_vectors(vs,const)
        for i in range(len(vs)):
            nv1=numericalc.numerical_vector(vs[i])
            a=nv1*nconst
            b=numericalc.numerical_vector(mvs[i])
            if np.allclose(a,b):
                pass
            else:
                flg+=0
    if flg==0:
        print('mul_vectors: Correct!')
    else:
        print('mul_vectors: Wrong')
    
    # ベクトル合成
    flg=0
    for _ in range(ncycle):
        v1=generate_random_vector()
        v2=generate_random_vector()
        #
        n1=numericalc.numerical_vector(v1)
        n2=numericalc.numerical_vector(v2)
        a=n1+n2
        #
        v=add_vectors(v1,v2)
        b=numericalc.numerical_vector(v)
        #print(b)
        if np.allclose(a,b):
            pass
        else:
            flg+=0
    if flg==0:
        print('add_vectors: Correct!')
    else:
        print('add_vectors: Wrong')
    
    # ベクトルの差
    flg=0
    for _ in range(ncycle):
        v1=generate_random_vector()
        v2=generate_random_vector()
        #
        n1=numericalc.numerical_vector(v1)
        n2=numericalc.numerical_vector(v2)
        a=n1-n2
        #
        v=sub_vectors(v1,v2)
        b=numericalc.numerical_vector(v)
        #print(b)
        if np.allclose(a,b):
            pass
        else:
            flg+=0
    if flg==0:
        print('sub_vectors: Correct!')
    else:
        print('sub_vectors: Wrong')
    
    # 外積
    flg=0
    for _ in range(ncycle):
        v1=generate_random_vector(3)
        v2=generate_random_vector(3)
        #
        n1=numericalc.numerical_vector(v1)
        n2=numericalc.numerical_vector(v2)
        a=np.cross(n1,n2)
        #
        v=outer_product(v1,v2)
        b=numericalc.numerical_vector(v)
        #print(b)
        if np.allclose(a,b):
            pass
        else:
            flg+=0
    if flg==0:
        print('outer_product: Correct!')
    else:
        print('outer_product: Wrong')

    # 内積
    flg=0
    for _ in range(ncycle):
        v1=generate_random_vector(3)
        v2=generate_random_vector(3)
        #
        n1=numericalc.numerical_vector(v1)
        n2=numericalc.numerical_vector(v2)
        a=np.dot(n1,n2)
        #
        v=inner_product(v1,v2)
        b=numericalc.numeric_value(v)
        #print(b)
        if abs(a-b)<eps:
            pass
        else:
            flg+=0
    if flg==0:
        print('inner_product: Correct!')
    else:
        print('inner_product: Wrong')
    
    
    #-----------------------------------
    # check: projection
    #-----------------------------------
    
    v=generate_random_vector()
    v=projection(v)
    ve=v[0]
    vi=v[1]
    