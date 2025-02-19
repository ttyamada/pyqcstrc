#if __name__ == '__main__':

# test
import sys
import numpy as np
from numpy.typing import NDArray
#sys.path.append('.')
from ico2.numericalc import coplanar_check_numeric_tau

import random
from pyqcstrc.ico2.numericalc import (numeric_value,
                                    numerical_vector,
                                    numerical_vectors,
                                    get_internal_component_numerical,
                                    get_internal_component_sets_numerical,
                                    point_on_segment,
                                    coplanar_check_numeric_tau,
                                    )
from math1 import (add,sub,mul,
                   div,add_vectors,sub_vectors,mul_vectors,
                   inner_product,outer_product
                   )

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
    