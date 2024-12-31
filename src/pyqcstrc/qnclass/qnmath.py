import sys
import numpy as np
from numpy.typing import NDArray
#sys.path.append('.')
#from numericalc import coplanar_check_numeric_tau
from pyqcstrc.dode2.numericalc import coplanar_check_numeric_tau
    
class Qnmath:
    #!/usr/bin/env python
    #
    # PyQCstrc - Python library for Quasi-Crystal structure
    # Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
    #
    #SQRTN=np.sqrt(5)
    N=5
    
    def __init__(self, N):
        self.N = N
        
    def __add__(a: NDArray[np.int64], b: NDArray[np.int64]):
        return add(a,b)
    
    def __sub__(a: NDArray[np.int64], b: NDArray[np.int64]):
        return sub(a,b)
    
    def __mul__(a: NDArray[np.int64], b: NDArray[np.int64]):
        return mul(a,b)
    
    def __truediv(a: NDArray[np.int64], b: NDArray[np.int64]):
        return div(a,b)
    
    def add(a: NDArray[np.int64], b:NDArray[np.int64]) -> NDArray[np.int64]:
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
	    #N=self.N
	    c1=a[0]*b[2]+b[0]*a[2]
	    c2=a[1]*b[2]+b[1]*a[2]
	    c3=a[2]*b[2]
	    x=np.array([c1,c2,c3],dtype=np.int64)
	    g=np.gcd.reduce(x)
	    c1=int(c1/g)
	    c2=int(c2/g)
	    c3=int(c3/g)
	    if c3<0:
	        return np.array([-c1,-c2,-c3])
	    else:
	        return np.array([c1,c2,c3])
    
    def sub(a: NDArray[np.int64], b:NDArray[np.int64]) -> NDArray[np.int64]:
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
	    #N=self.N
	    c1=a[0]*b[2]-b[0]*a[2]
	    c2=a[1]*b[2]-b[1]*a[2]
	    c3=a[2]*b[2]
	    x=np.array([c1,c2,c3],dtype=np.int64)
	    g=np.gcd.reduce(x)
	    c1=int(c1/g)
	    c2=int(c2/g)
	    c3=int(c3/g)
	    if c3<0:
	        return np.array([-c1,-c2,-c3])
	    else:
	        return np.array([c1,c2,c3])

    def mul(a: NDArray[np.int64], b:NDArray[np.int64]) -> NDArray[np.int64]:
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
	    #N=self.N
	    c1=a[0]*b[0]+N*a[1]*b[1]
	    c2=a[0]*b[1]+a[1]*b[0]
	    c3=a[2]*b[2]
	    x=np.array([c1,c2,c3],dtype=np.int64)
	    g=np.gcd.reduce(x)
	    c1=int(c1/g)
	    c2=int(c2/g)
	    c3=int(c3/g)
	    if c3<0:
	        return np.array([-c1,-c2,-c3])
	    else:
	        return np.array([c1,c2,c3])
    
    def div(a: NDArray[np.int64], b:NDArray[np.int64]) -> NDArray[np.int64]:
	    """
	    # division (a/b) in SIN-style
	    
	    Parameters
	    ----------
	    a: array
	        value in SQRTN
	    b: array
	        value in SQRTN
	    c: array
	        inverse of b
	    Returns
	    -------
	    array
	    """
	    #N=self.N
	    c1=b[0]*b[2]
	    c2=-b[1]*b[2]
	    c3=b[0]*b[0]-N*b[1]*b[1]
	    #c=[c1,c2,c3]
	    c=np.array([c1,c2,c3],dtype=np.int64)
	    if c3==0:
	        print('ERROR_1:division error')
	        return
	    return mul(a,c)

    def add_vectors(vt1: NDArray[np.int64], vt2:NDArray[np.int64]) -> NDArray[np.int64]:
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
	    a=np.zeros(vt1.shape,dtype=np.int64)
	    for i in range(len(vt1)):
	        a[i]=vt1[i]+vt2[i]  #add(vt1[i],vt2[i])
	    return a

    def sub_vectors(vt1: NDArray[np.int64], vt2:NDArray[np.int64]) -> NDArray[np.int64]:
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
	        #const=np.array([-1,0,1],dtype=np.int64)
	        #vt2=mul_vector(vt2,const)
	        return vt1-vt2  #add_vectors(vt1,vt2)
	    else:
	        print('incorrect shape')
	        return

    def mul_vector(vt: NDArray[np.int64], coeff:NDArray[np.int64]) -> NDArray[np.int64]:
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
	        a=np.zeros(vt.shape,dtype=np.int64)
	        for i,v in enumerate(vt):
	            a[i]=v+coeff  #mul(v,coeff)
	        return a
	    else:
	        print('incorrect shape')
	        return

    def mul_vectors(vts: NDArray[np.int64], coeff:NDArray[np.int64]) -> NDArray[np.int64]:
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

    def shift_vectors(vts: NDArray[np.int64], vt: NDArray[np.int64]) -> NDArray[np.int64]:
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
	        a=np.zeros(vts.shape,dtype=np.int64)
	        for i,vt1 in enumerate(vts):
	            a[i]=add_vectors(vt1,vt)
	        return a
	    elif vts.ndim==4:
	        a=np.zeros(vts.shape,dtype=np.int64)
	        for i1,vt1 in enumerate(vts):
	            for i2,vt2 in enumerate(vt1):
	                a[i1][i2]=add_vectors(vt2,vt)
	    else:
	        print('incorrect shape')
	        return

    def outer_product(vt1: NDArray[np.int64], vt2:NDArray[np.int64]) -> NDArray[np.int64]:
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
	    a=vt1[1]*vt2[2] #mul(vt1[1],vt2[2])
	    b=vt1[2]*vt2[1] #mul(vt1[2],vt2[1])
	    c1=a-b          #sub(a,b)
	    #
	    a=vt1[2]*vt2[0] #mul(vt1[2],vt2[0])
	    b=vt1[0]*vt2[2] #mul(vt1[0],vt2[2])
	    c2=a-b          #sub(a,b)
	    #
	    a=vt1[0]*vt2[1] #mul(vt1[0],vt2[1])
	    b=vt1[1]*vt2[0] #mul(vt1[1],vt2[0])
	    c3=a-b          #sub(a,b)
	    #
	    return np.array([c1,c2,c3],dtype=np.int64)

    def inner_product(vt1: NDArray[np.int64], vt2:NDArray[np.int64]) -> NDArray[np.int64]:
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
	            b=vt1[i]*vt2[i]  #mul(vt1[i],vt2[i])
	            a=a+b            #add(a,b)
	        return a

    def dot_product(mat1: NDArray[np.int64], mat2:NDArray[np.int64]) -> NDArray[np.int64]:
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
	            mat_new=np.zeros((s,3),dtype=np.int64)
	            for k in range(s):
	                a=np.array([0,0,1])
	                for j in range(t1):
	                    b=mat1[k][j]*mat2[j]  #mul(mat1[k][j],mat2[j])
	                    a=a+b                 #add(a,b)
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
	                        b=mat1[k][i]*mat2[i][j]  #mul(mat1[k][i],mat2[i][j])
	                        a=a+b                    #add(a,b)
	                    mat_new[k][j]=a
	            return mat_new
	    else:
	        print('incorrect shape found in dot_product')
	        return 

    def dot_product_1(mat1: NDArray[np.int64], mat2:NDArray[np.int64]) -> NDArray[np.int64]:
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
	            mat_new=np.zeros((s,3),dtype=np.int64)
	            for k in range(s):
	                a=np.array([0,0,1])
	                for j in range(t1):
	                    val=np.array([mat1[k][j],0,1])
	                    b=val*mat2[j]  #mul(val,mat2[j])
	                    a=a+b          #add(a,b)
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
	                        b=val*mat2[i][j]  #mul(val,mat2[i][j])
	                        a=a+b             #add(a,b)
	                    mat_new[k][j]=a
	            return mat_new
	    else:
	        print('incorrect shape found in dot_product')
	        return 

    