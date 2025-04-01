#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
import cython
from numpy.typing import NDArray

#sys.path.append('.')
#from numericalc import coplanar_check_numeric_tau
from octa2.numericalc import coplanar_check_numeric_tau

#from libcpp import bool
#from cpython import bool as bool_t
#from cython.cimports.cython.view import array as cvarray
#from cython.cimports.cython import cython

if cython.compiled:
    DTYPE_int = cython.long
else:
    DTYPE_int = np.int64  #long

SQRT2=np.sqrt(2)


def qnreduce(x:cython.long[:]) -> cython.long[:]:
    g: cython.long
    g=np.gcd.reduce(x)
    x[0]=(DTYPE_int)(x[0]/g)
    x[1]=(DTYPE_int)(x[1]/g)
    x[2]=(DTYPE_int)(x[2]/g)
    if x[2]<0:
        x[0]=-x[0]; x[1]=-x[1]; x[2]=-x[2]
    return x

#def add(a: NDArray[DTYPE_int], b: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
def add(a: cython.long[:], b: cython.long[:]) -> cython.long[:]:
    """
    # summation (a+b) in SQRT2-style
    
    Parameters
    ----------
    a: array
        value in SQRT2-style
    b: array
        value in SQRT2-style
    
    Returns
    -------
    array
    """    
    x0=np.arange(3,dtype=np.dtype("l")) # ndim=1
    x=cython.declare(cython.long[:],x0)
    
    x[0]=a[0]*b[2]+b[0]*a[2]
    x[1]=a[1]*b[2]+b[1]*a[2]
    x[2]=a[2]*b[2]
    #x: DTYPE_int=np.array([c[0],c[1],c[2]],dtype=DTYPE_int)
    x=qnreduce(x)
    return x  #np.array([c[0],c[1],c[2]])

#def sub(a: NDArray[DTYPE_int], b: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
def sub(a: cython.long[:], b: cython.long[:]) -> cython.long[:]:
    """
    # summation (a+b) in SQRT2-style
    
    Parameters
    ----------
    a: array
        value in SQRT2-style
    b: array
        value in SQRT2-style
    
    Returns
    -------
    array
    """    

    x0=np.arange(3,dtype=np.dtype("l"))
    x=cython.declare(cython.long[:],x0)
    
    x[0]=a[0]*b[2]-b[0]*a[2]
    x[1]=a[1]*b[2]-b[1]*a[2]
    x[2]=a[2]*b[2]
    #x: DTYPE_int=np.array([c[0],c[1],c[2]],dtype=DTYPE_int)
    x=qnreduce(x)
    return x  #np.array([c[0],c[1],c[2]])


#@cython.cfunc
#@cython.inline
#@cython.exceptval(-1.0)


#def mul(a: NDArray[DTYPE_int], b: NDArray[DTYPE_int])  -> NDArray[DTYPE_int]:
def mul(a: cython.long[:], b: cython.long[:]) -> cython.long[:]:
    """
    # multiplication (a*b) in SQRT2-style
    
    Parameters
    ----------
    a: array
        value in SQRT2-style
    b: array
        value in SQRT2-style
    
    Returns
    -------
    array
    """

    x0=np.arange(3,dtype=np.dtype("l"))
    x=cython.declare(cython.long[:],x0)
    # following three parallelizable
    x[0]=a[0]*b[0]+2*a[1]*b[1]
    x[1]=a[0]*b[1]+a[1]*b[0]
    x[2]=a[2]*b[2]
    x=qnreduce(x)
    return x  #np.array([c[0],c[1],c[2]])

#def div(a: NDArray[DTYPE_int], b:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
def div(a: cython.long[:], b: cython.long[:]) -> cython.long[:] :
    """
    # division (a/b) in SQRT2-style
    
    Parameters
    ----------
    a: array
        value in SQRT2-style
    b: array
        value in SQRT2-style
    
    Returns
    -------
    array
    """

    #cdef np.ndarray[DTYPE_int,ndim=1] x
    x0=np.arange(3,dtype=np.dtype("l"))
    x=cython.declare(cython.long[:],x0)
    
    x[0]=(a[0]*b[0]-2*a[1]*b[1])*b[2]
    x[1]=(a[1]*b[0]-a[0]*b[1])*b[2]
    x[2]=(b[0]**2-2*b[1]**2)*a[2]
    
    x=qnreduce(x)
    return x  #np.array([c[0],c[1],c[2]])

def add_vectors(vt1: cython.long[:,:], vt2: cython.long[:,:]) -> cython.long[:,:]:
    """Composition of two vectors, v1+v2
    
    Parameters
    ----------
    vt1: array
        a vector in SQRT2-style
    vt2: array,
        a scalar in SQRT2-style
    
    Returns
    -------
    Composition of two vectors: array in SQRT2-style
    
    """
    #a=np.zeros(vt1.shape,dtype=DTYPE_int)
    ln: cython.long = len(vt1)
    a0=np.arange(ln*3,dtype=np.dtype("l")).reshape((ln,3))
    a=cython.declare(cython.long[:,:],a0)
    
    for i in range(ln):
        a[i]=add(vt1[i],vt2[i])
    return a

def sub_vectors(vt1: cython.long[:,:], vt2: cython.long[:,:]) -> cython.long[:,:]:
    """Subtraction of two vectors, v1-v2
    
    Parameters
    ----------
    vt1: array
        a vector in SQRT2-style
    vt2: array,
        a scalar in SQRT2-style
    
    Returns
    -------
    Composition of two vectors: array in SQRT2-style
    """
    
    #a=np.zeros(vt1.shape,dtype=DTYPE_int)
    ln: cython.long = len(vt1)
    a0=np.arange(ln*3,dtype=np.dtype("l")).reshape((ln,3))
    a=cython.declare(cython.long[:,:],a0)
    #print("type(vt1)",type(vt1)) # for check
    for i in range(ln):
        #print("type(vt1[i])",type(vt1[i]),"type(vt1[i])",type(vt2[i])) # for test
        a[i]=sub(vt1[i],vt2[i])
    return a


def mul_vector(vt: cython.long[:,:], coeff: cython.long[:]) -> cython.long[:,:]:
    """Multiplying a vector by a scalar in SQRT2-style.
    
    Parameters
    ----------
    vt: array
        a vector in SQRT2-style
    coeff: array,
        a scalar in SQRT2-style
    
    Returns
    -------
    Multiplied vector: array in SQRT2-style
    """

    #a=np.zeros(vt.shape,dtype=DTYPE_int)
    shape=vt.shape
    ln: cython.long = shape[0]
    a0=np.arange(ln*3,dtype=np.dtype("l")).reshape((ln,3))
    a=cython.declare(cython.long[:,:],a0)
    
    #for i,v in enumerate(vt):
    for i in range(ln):
        v=vt[i]
        a[i]=mul(v,coeff)
    return a
 

def mul_vectors(vts: cython.long[:,:,:], coeff: cython.long[:]) -> cython.long[:,:,:]:
    """multiplying a set of vectors by a scalar in SQRT2-style.
    
    Parameters
    ----------
    vts: array
        a set of vectors in SQRT2-style
    coeff: array,
        a scalar in SQRT2-style
    
    Returns
    -------
    Multiplied vectors: array in SQRT2-style
    """
    shape=vts.shape
    ln0: cython.long=shape[0]
    ln1: cython.long=shape[1]
    #if vts.ndim==3:
        #a=np.zeros(vts.shape,dtype=DTYPE_int)
    a0=np.arange(ln0*ln1*3,dtype=np.dtype("l")).reshape((ln0,ln1,3))
    a=cython.declare(cython.long[:,:,:],a0)
    for i in range(ln0):
        vt=vts[i]
        a[i]=mul_vector(vt,coeff)
    return a
    #elif vts.ndim==4:
    #    a=np.zeros(vts.shape,dtype=DTYPE_int)
    #    for i1,vt in enumerate(vts):
    #        for i2,v in enumerate(vt):
    #            a[i1][i2]=mul_vector(v,coeff)
    #else:
    #    print('incorrect shape')
    #    return 999999

def shift_vectors(vts: cython.long[:,:,:], vt: cython.long[:,:]) -> cython.long[:,:,:]:
    """Shift a set of vectors by adding a vector in SQRT2-style.
    
    Parameters
    ----------
    vts: array
        a set of vectors in SQRT2-style
    coeff: array,
        a scalar in SQRT2-style
    
    Returns
    -------
    Multiplied vectors: array in SQRT2-style
    """
    #if vts.ndim==3:
    #a=np.zeros(vts.shape,dtype=DTYPE_int)
    shape=vts.shape
    ln0: cython.long=shape[0]
    ln1: cython.long=shape[1]
    #if vts.ndim==3:
        #a=np.zeros(vts.shape,dtype=DTYPE_int)
    a0=np.arange(ln0*ln1*3,dtype=np.dtype("l")).reshape((ln0,ln1,3))
    a=cython.declare(cython.long[:,:,:],a0)
    for i in range(ln0):
        vt1=vts[i]
        a[i]=add_vectors(vt1,vt)
    return a
    #elif vts.ndim==4:
    #    a=np.zeros(vts.shape,dtype=DTYPE_int)
    #    for i1,vt1 in enumerate(vts):
    #        for i2,vt2 in enumerate(vt1):
    #            a[i1][i2]=add_vectors(vt2,vt)
    #else:
    #    print('incorrect shape')
    #    return 999999
    
    
def outer_product(vt1: cython.long[:,:], vt2: cython.long[:,:]) -> cython.long[:,:]:
    """Outer product of two 3d vectors, v1 and v2 in SQRT2-style.

    Parameters
    ----------
    v1: array
        3-dimensional vector in SQRT2-style
    v2: array,
        3-dimensional vector in SQRT2-style

    Returns
    -------
    Outer product: array in SQRT2-style
    """
    shape=vt1.shape
    ln: cython.long=shape[0]
    c0=np.arange(ln*3,dtype=np.dtype("l")).reshape((ln,3))
    c=cython.declare(cython.long[:,:],c0)
    a=mul(vt1[1],vt2[2])
    b=mul(vt1[2],vt2[1])
    c[0]=sub(a,b)
    #
    a=mul(vt1[2],vt2[0])
    b=mul(vt1[0],vt2[2])
    c[1]=sub(a,b)
    #
    a=mul(vt1[0],vt2[1])
    b=mul(vt1[1],vt2[0])
    c[2]=sub(a,b)
    #
    return c

def inner_product(vt1: cython.long[:,:], vt2: cython.long[:,:]) -> cython.long[:]:
    """Inner product of two vectors, v1 and v2 in SQRT2-style.

    Parameters
    ----------
    vt1: array
        vector in SQRT2-style
    vt2: array,
         vector in SQRT2-style

    Returns
    -------
    Inner product: array in SQRT2-style
    """
    ln: cython.long=len(vt1)
    a=np.array([0,0,1])
    for i in range(ln):
        b=mul(vt1[i],vt2[i])
        a=add(a,b)
    return a

def dot_product(mat1: NDArray[DTYPE_int], mat2:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """product of two matrices, mat1*mat2.
    
    Parameters
    ----------
    mat1: ndarray
        (s,t) in SQRT2-style
    mat2: ndarray
        (t,u) in SQRT2-style

    Returns
    -------
    Inner product: array in SQRT2-style
    """
    ndim1=mat1.ndim
    ndim2=mat2.ndim
    
    if ndim1==2 and ndim2==2:
        return inner_product(mat1,mat2)
    
    elif ndim1==3 and ndim2==2:
        s:cython.long =mat1.shape[0]
        t1:cython.long =mat1.shape[1]
        t2:cython.long=mat2.shape[0]
        if t1!=t2:
            print('incorrect shape found in dot_product')
            return 999999
        else:
            mat_new=np.zeros((s,3),dtype=DTYPE_int)  #DTYPE_int)
            for k in range(s):
                a=np.array([0,0,1])
                for j in range(t1):
                    b=mul(mat1[k][j],mat2[j])
                    a=add(a,b)
                mat_new[k]=a
            return mat_new
            
    elif ndim1==3 and ndim2==3:
        s:cython.long=mat1.shape[0]
        t1:cython.long=mat1.shape[1]
        t2:cython.long=mat2.shape[0]
        u:cython.long=mat2.shape[1]
        if t1!=t2:
            print('incorrect shape found in dot_product')
            return 999999
        else:
            mat_new=np.zeros((s,u,3),dtype=DTYPE_int)  #DTYPE_int)
            for k in range(s):
                for j in range(u):
                    a=np.array([0,0,1])
                    for i in range(t1):
                        b=mul(mat1[k][i],mat2[i][j])
                        a=add(a,b)
                    mat_new[k][j]=a
            return mat_new

def dot_product_1(mat1: NDArray[DTYPE_int], mat2:NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """product of two matrices, mat1*mat2.
    
    Parameters
    ----------
    mat1: ndarray
        (s,t) in "NOT" SQRT2-style
    mat2: ndarray
        (t,u) in SQRT2-style

    Returns
    -------
    Inner product: array in SQRT2-style
    """
    ndim1=mat1.ndim
    ndim2=mat2.ndim
    
    if ndim1==2 and ndim2==2:
        s:cython.long=mat1.shape[0]
        t1:cython.long=mat1.shape[1]
        t2:cython.long=mat2.shape[0]
        if t1!=t2:
            print('incorrect shape found in dot_product')
            return 999999
        else:
            mat_new=np.zeros((s,3),dtype=DTYPE_int)  #DTYPE_int)
            for k in range(s):
                a=np.array([0,0,1])
                for j in range(t1):
                    val=np.array([mat1[k][j],0,1])
                    b=mul(val,mat2[j])
                    a=add(a,b)
                mat_new[k]=a
            return mat_new
            
    elif ndim1==2 and ndim2==3:
        s:cython.long = mat1.shape[0]
        t1:cython.long = mat1.shape[1]
        t2:cython.long = mat2.shape[0]
        u:cython.long = mat2.shape[1]
        if t1!=t2:
            print('incorrect shape found in dot_product')
            return 999999
        else:
            mat_new=np.zeros((s,u,3),dtype=DTYPE_int)  #DTYPE_int)
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
        return 999999

def centroid(obj: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """geometric center, centroid of tetrahedron, triangle or edge, in SQRT2-style.

    Parameters
    ----------
    obj: array
        6-dimensional vector in SQRT2-style
    
    Returns
    -------
    centroid: array in SQRT2-style
    """
    
    num:cython.long = len(obj)
    v0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=DTYPE_int)  #DTYPE_int)
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

# needless???
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
    tmp=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=DTYPE_int)  #DTYPE_int)
    ln:cython.long=obj.shape[0]
    for i in range(ln):
        tet=obj[i]
        p=centroid(tet)
        tmp=add_vectors(tmp,p)
    return mul_vector(tmp,np.array([1,0,len(obj)]))

"""
def coplanar_check(p: NDArray[DTYPE_int],num_iteration: int=5) -> bool:
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
   return coplanar_check_numeric_tau(p,num_iteration)
"""

def matrixpow(ma: NDArray[DTYPE_int], n: int) -> NDArray[DTYPE_int]:
    """
    """
    (mx,my)=ma.shape
    n:cython.long
    if mx==my:
        if n==0:
            return np.identity(mx)
        elif n<0:
            tmp=np.identity(mx)
            inva = np.linalg.inv(ma)
            for i in range(-n):
                #tmp=np.dot(tmp,inva)
                tmp=tmp@inva
            return tmp
        else:
            tmp=np.identity(mx)
            for i in range(n):
                #tmp=np.dot(tmp,ma)
                tmp=tmp@ma
            return tmp
    else:
        print('matrix has not regular shape')
        return 999999

def det_matrix(mtx: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Determinant of 3x3 matrix, mtx, in SQRT2 style
    
    Parameters
    ----------
    mtx: array
        3x3 matrix in SQRT2-style

    Returns
    -------
    6d vectors projected onto Eperp in SQRT2-style.
    """      
    
    t3=mul(mtx[0][0],mtx[1][1])
    t1=mul(t3,mtx[2][2])
    #
    t3=mul(mtx[0][2],mtx[1][0])
    t2=mul(t3,mtx[2][1])
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
