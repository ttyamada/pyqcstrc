#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
sys.path.append('.')
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

def add_vectors(vt1,vt2):
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

def sub_vectors(vt1,vt2):
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

def mul_vector(vt,coeff):
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
        for i in range(len(vt)):
            a[i]=mul(vt[i],coeff)
        return a
    else:
        print('incorrect shape')
        return

def mul_vectors(vts,coeff):
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
        for i in range(len(vts)):
            a[i]=mul_vector(vts[i],coeff)
        return a
    elif vts.ndim==4:
        a=np.zeros(vts.shape,dtype=np.int64)
        for i1 in range(len(vts[i1])):
            for i2 in range(len(vts[i1][i2])):
                a[i1][i2]=mul_vector(vts[i1][i2],coeff)
    else:
        print('incorrect shape')
        return

def outer_product(vt1,vt2):
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

def inner_product(vt1,vt2):
    """Inner product of two vectors, v1 and v2 in TAU-style.

    Parameters
    ----------
    vt1: array
        vector in TAU-style
    vt2: array,
         vector in TAU-style

    Returns
    -------
    Inner product: array in TAU-style
    """
    s1,_=vt1.shape
    s2,_=vt2.shape
    if s1!=s2:
        print('matrices have not a proper shape.')
        return 
    else:
        a=np.array([0,0,1])
        for i in range(s1):
            b=mul(vt1[i],vt2[i])
            a=add(a,b)
        return a

def dot_product(mat1,mat2):
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

def dot_product_1(mat1,mat2):
    """product of two matrices, mat1*mat2.
    
    Parameters
    ----------
    mat1: ndarray
        (s,t) in "NOT" TAU-style
    mat2: ndarray
        (t,u) in TAU-style

    Returns
    -------
    Inner product: array in TAU-style
    """
    ndim1=mat1.ndim
    ndim2=mat2.ndim
    
    if ndim1==2 and ndim2==2:
        s,t1,=mat1.shape
        t2,_=mat2.shape
        if t1!=t2:
            print('incorrect shape found in dot_product')
            return 
        else:
            mat_new=np.zeros((s,3),dtype=np.int64)
            for k in range(s):
                a=np.array([0,0,1])
                for j in range(t1):
                    val=np.array([mat1[k][j],0,1])
                    b=mul(val,mat2[j])
                    a=add(a,b)
                mat_new[k]=a
            return mat_new
            
    elif ndim1==2 and ndim2==3:
        s,t1,=mat1.shape
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
                        val=np.array([mat1[k][j],0,1])
                        b=mul(val,mat2[i][j])
                        a=add(a,b)
                    mat_new[k][j]=a
            return mat_new
    else:
        print('incorrect shape found in dot_product')
        return 








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

def centroid(obj):
    """geometric center, centroid of tetrahedron, triangle or edge, in TAU-style.

    Parameters
    ----------
    obj: array
        6-dimensional vector in TAU-style
    
    Returns
    -------
    centroid: array in TAU-style
    """
    
    num=len(obj)
    v0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
    i2=0
    for i2 in range(6):
        v2=v0[i2]
        i1=0
        for i1 in range(num):
            v2=add(v2,obj[i1][i2])
            i1+=1
        v0[i2]=mul(v2,np.array([1,0,num]))
        i2+=1
    return v0

def centroid_obj(obj):
    """geometric center, centroid of tetrahedron, in TAU-style.

    Parameters
    ----------
    tetrahedron: array
        6-dimensional vector in TAU-style
    
    Returns
    -------
    centroid: array in TAU-style
    """
    #print('centroid_obj')
    
    #  geometric center, centroid of OBJ
    tmp=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
    for tetrahedron in obj:
        p=centroid(tetrahedron)
        tmp=add_vectors(tmp,p)
    return mul_vector(tmp,np.array([1,0,len(obj)]))

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
            #return 1 # coplanar
            return True # coplanar
        else:
            #return 0
            return False
    else:
        #return 1 # coplanar
        return True # coplanar

def matrixpow(ma,n):
    """
    """
    (mx,my)=ma.shape
    if mx==my:
        if n==0:
            return np.identity(mx)
        elif n<0:
            tmp=np.identity(mx)
            inva = np.linalg.inv(ma)
            for i in range(-n):
                tmp=np.dot(tmp,inva)
            return tmp
        else:
            tmp=np.identity(mx)
            for i in range(n):
                tmp=np.dot(tmp,ma)
            return tmp
    else:
        print('matrix has not regular shape')
        return 

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

if __name__ == '__main__':
    
    # test
    
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
    