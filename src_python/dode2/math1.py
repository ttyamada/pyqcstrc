#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
from numpy.typing import NDArray
import cython
#sys.path.append('.')
#from numericalc import coplanar_check_numeric_tau
from dode2.numericalc import coplanar_check_numeric_tau

SIN=np.sqrt(3)/2
DTYPE_int = cython.long
#DTYPE_int = DTYPE_int

def add(a: NDArray[DTYPE_int], b:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """
    # summation (a+b) in SIN-style
    
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
    c1=a[0]*b[2]+b[0]*a[2]
    c2=a[1]*b[2]+b[1]*a[2]
    c3=a[2]*b[2]
    x=np.array([c1,c2,c3],dtype=DTYPE_int)
    g=np.gcd.reduce(x)
    c1=int(c1/g)
    c2=int(c2/g)
    c3=int(c3/g)
    if c3<0:
        return np.array([-c1,-c2,-c3])
    else:
        return np.array([c1,c2,c3])

def mul(a: NDArray[DTYPE_int], b:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
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
    c1=4*a[0]*b[0]+3*a[1]*b[1]
    c2=4*(a[0]*b[1]+a[1]*b[0])
    c3=4*a[2]*b[2]
    x=np.array([c1,c2,c3],dtype=DTYPE_int)
    g=np.gcd.reduce(x)
    c1=int(c1/g)
    c2=int(c2/g)
    c3=int(c3/g)
    if c3<0:
        return np.array([-c1,-c2,-c3])
    else:
        return np.array([c1,c2,c3])

def sub(a: NDArray[DTYPE_int], b:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """
    # subtraction (a/b) in SIN-style
    
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
    c=np.array([-1,0,1],dtype=DTYPE_int)
    b=mul(c,b)
    return add(a,b)

def div(a: NDArray[DTYPE_int], b:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """
    # division (a/b) in SIN-style
    
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
    if np.all(b[:2]==0):
        print('ERROR_1:division error')
        return 
    else:
        if np.all(a[:2]==0):
            return np.array([0,0,1],dtype=DTYPE_int)
        else:
            if b[1]!=0:
                if b[0]!=0:
                    if 4*b[0]**2-3*b[1]**2!=0:
                        c1=b[2]*(4*a[0]*b[0]-3*a[1]*b[1])
                        c2=-4*b[2]*(a[0]*b[1]-a[1]*b[0])
                        c3=a[2]*(4*b[0]**2-3*b[1]**2)
                    else:
                        c1=3*a[1]*b[2]
                        c2=4*a[0]*b[2]
                        c3=6*a[2]*b[1]
                else:
                    c1=3*a[1]*b[2]
                    c2=4*a[0]*b[2]
                    c3=3*a[2]*b[1]
            else:
                c1=a[0]*b[2]
                c2=a[1]*b[2]
                c3=b[0]*a[2]
            x=np.array([c1,c2,c3],dtype=DTYPE_int)
            g=np.gcd.reduce(x)
            if g!=0:
                c1=int(c1/g)
                c2=int(c2/g)
                c3=int(c3/g)
                if c3<0:
                    return np.array([-c1,-c2,-c3],dtype=DTYPE_int)
                else:
                    return np.array([c1,c2,c3],dtype=DTYPE_int)
            else:
                print('ERROR_2:division error')
                return 

def add_vectors(vt1: NDArray[DTYPE_int], vt2:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Composition of two vectors, v1+v2
    
    Parameters
    ----------
    vt1: array
        a vector in SIN-style
    vt2: array,
        a scalar in SIN-style
    
    Returns
    -------
    Composition of two vectors: array in SIN-style
    
    """
    a=np.zeros(vt1.shape,dtype=DTYPE_int)
    for i in range(len(vt1)):
        a[i]=add(vt1[i],vt2[i])
    return a

def sub_vectors(vt1: NDArray[DTYPE_int], vt2:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Subtraction of two vectors, v1-v2
    
    Parameters
    ----------
    vt1: array
        a vector in SIN-style
    vt2: array,
        a scalar in SIN-style
    
    Returns
    -------
    Subtraction of two vectors: array in SIN-style
    """
    if vt1.ndim==2 and vt2.ndim==2:
        const=np.array([-1,0,1],dtype=DTYPE_int)
        vt2=mul_vector(vt2,const)
        return add_vectors(vt1,vt2)
    else:
        print('incorrect shape')
        return

def mul_vector(vt: NDArray[DTYPE_int], coeff:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Multiplying a vector by a scalar in SIN-style.
    
    Parameters
    ----------
    vt: array
        a vector in SIN-style
    coeff: array,
        a scalar in SIN-style
    
    Returns
    -------
    Multiplied vector: array in SIN-style
    """
    if vt.ndim==2:
        a=np.zeros(vt.shape,dtype=DTYPE_int)
        for i,v in enumerate(vt):
            a[i]=mul(v,coeff)
        return a
    else:
        print('incorrect shape')
        return

def mul_vectors(vts: NDArray[DTYPE_int], coeff:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """multiplying a set of vectors by a scalar in SIN-style.
    
    Parameters
    ----------
    vts: array
        a set of vectors in SIN-style
    coeff: array,
        a scalar in SIN-style
    
    Returns
    -------
    Multiplied vectors: array in SIN-style
    """
    if vts.ndim==3:
        a=np.zeros(vts.shape,dtype=DTYPE_int)
        for i,vt in enumerate(vts):
            a[i]=mul_vector(vt,coeff)
        return a
    elif vts.ndim==4:
        a=np.zeros(vts.shape,dtype=DTYPE_int)
        for i1,vt in enumerate(vts):
            for i2,v in enumerate(vt):
                a[i1][i2]=mul_vector(v,coeff)
    else:
        print('incorrect shape')
        return

def shift_vectors(vts: NDArray[DTYPE_int], vt: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Shift a set of vectors by adding a vector in SIN-style.
    
    Parameters
    ----------
    vts: array
        a set of vectors in SIN-style
    coeff: array,
        a scalar in SIN-style
    
    Returns
    -------
    Multiplied vectors: array in SIN-style
    """
    if vts.ndim==3:
        a=np.zeros(vts.shape,dtype=DTYPE_int)
        for i,vt1 in enumerate(vts):
            a[i]=add_vectors(vt1,vt)
        return a
    elif vts.ndim==4:
        a=np.zeros(vts.shape,dtype=DTYPE_int)
        for i1,vt1 in enumerate(vts):
            for i2,vt2 in enumerate(vt1):
                a[i1][i2]=add_vectors(vt2,vt)
    else:
        print('incorrect shape')
        return
    
    
def outer_product(vt1: NDArray[DTYPE_int], vt2:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Outer product of two 3d vectors, v1 and v2 in SIN-style.

    Parameters
    ----------
    v1: array
        3-dimensional vector in SIN-style
    v2: array,
        3-dimensional vector in SIN-style

    Returns
    -------
    Outer product: array in SIN-style
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
    return np.array([c1,c2,c3],dtype=DTYPE_int)

def inner_product(vt1: NDArray[DTYPE_int], vt2:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Inner product of two vectors, v1 and v2 in SIN-style.

    Parameters
    ----------
    vt1: array
        vector in SIN-style
    vt2: array,
         vector in SIN-style

    Returns
    -------
    Inner product: array in SIN-style
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

def dot_product(mat1: NDArray[DTYPE_int], mat2:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """product of two matrices, mat1*mat2.
    
    Parameters
    ----------
    mat1: ndarray
        (s,t) in SIN-style
    mat2: ndarray
        (t,u) in SIN-style

    Returns
    -------
    Inner product: array in SIN-style
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
            mat_new=np.zeros((s,3),dtype=DTYPE_int)
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
            mat_new=np.zeros((s,u,3),dtype=DTYPE_int)
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

def dot_product_1(mat1: NDArray[DTYPE_int], mat2:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """product of two matrices, mat1*mat2.
    
    Parameters
    ----------
    mat1: ndarray
        (s,t) in "NOT" SIN-style
    mat2: ndarray
        (t,u) in SIN-style

    Returns
    -------
    Inner product: array in SIN-style
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
            mat_new=np.zeros((s,3),dtype=DTYPE_int)
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
            mat_new=np.zeros((s,u,3),dtype=DTYPE_int)
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

def centroid(obj: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
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
    v0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=DTYPE_int)
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

def centroid_obj(obj: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
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
    tmp=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=DTYPE_int)
    for tetrahedron in obj:
        p=centroid(tetrahedron)
        tmp=add_vectors(tmp,p)
    return mul_vector(tmp,np.array([1,0,len(obj)]))

def coplanar_check(p: NDArray[DTYPE_int],num_iteration: int=5) -> bool:
    """Check whether a given set of points (in TAU-style) is coplanar or not.
    
    メモ：xyz1とxyz2の選び方次第で、outer_product(v1,v2)が小さくなりcoplanarと間違って判定する場合がある。
    これを避けるために適切なxyz1とxyz2の選び方が必要。以下では、ランダムにxyz1とxyz2の選ぶ。
    
    Parameters
    ----------
    p: array
        a set of pointsin TAU-style.

    Returns
    -------
    int
    #bool
    """
    
    """
    num=len(p)
    if num>3:
        flag=0
        lst0=[i for i in range(num)]
        for _ in range(num_iteration):
            lst3=random.sample(lst0, 3)
            xyz0i=projection3(p[lst3[0]])
            xyz1i=projection3(p[lst3[1]])
            xyz2i=projection3(p[lst3[2]])
            v1=sub_vectors(xyz1i,xyz0i)
            v2=sub_vectors(xyz2i,xyz0i)
            v3=outer_product(v1,v2)
            flag=0
            if np.all(d[:2])==0):
                pass
            else:
                flag=1
                break
        if flag==1:
            counter=0
            lst=list(filter(lambda x: x not in lst3, lst0))
            for i in lst:
                xyz3i=projection3(p[i])
                v4=sub_vectors(xyz3i,xyz0i)
                d=inner_product(v3,v4)
                if np.all(d[:2])==0:
                    pass
                else:
                    counter=1
                    break
            if counter==0:
                return True # coplanar
            else:
                return False
        else:
            'error in coplanar_check_numeric. increase num_iteration.'
            return 
    else:
        return True # coplanar
    """
    return coplanar_check_numeric_tau(p,num_iteration)

def matrixpow(ma: NDArray[DTYPE_int], n: int) -> NDArray[DTYPE_int]:
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

def det_matrix(mtx: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
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
    t2=mul(t3,t1) #t2=mul(t3,c[1])
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

