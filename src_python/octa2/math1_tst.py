import sys
import numpy as np
from numpy.typing import NDArray
from octa2.math1 import (add,sub,mul,div)

from octa2.numericalc import (numeric_value,
                            numerical_vector,
                            numerical_vectors,
                            get_internal_component_numerical,
                            get_internal_component_sets_numerical,
                            point_on_segment,
                            coplanar_check_numeric_tau,
                            )

# if __name__ == '__main__':
   
   # test

import random
sys.path.append('.')
from numericalc import (
    numeric_value,
    )

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

def math_check(a,b):
    """checking basic arithmetic operations in SQRT2-style.
    """
    flg=0
    a1=numeric_value(a)
    b1=numeric_value(b)
    print("a",a)
    print("b",b)
    print("a1",a1)
    print("b1",b1)
    
    c=add(a,b)
    c1=numeric_value(c)
    print("c",c)
    print("c1",c1)
    c2=a1+b1
    print("c2",c2)
    if abs(c1-c2)<eps:
        flg+=1
    else:
        print('+')
        
    c=sub(a,b)
    c1=numeric_value(c)
    c2=a1-b1
    print("c",c)
    print("c1",c1)
    print("a1-a2",c2)
    if abs(c1-c2)<eps:
        flg+=1
    else:
        print('-')
        
    c=mul(a,b)
    c1=numeric_value(c)
    c2=a1*b1
    print("a*b",c)
    print("a1*b1")
    
    if abs(c1-c2)<eps:
        flg+=1
    else:
        print('*')
        
    c=div(a,b)
    c1=numeric_value(c)
    c2=a1/b1
    print("a/b",c)
    print("a1/b1",c2)
    if abs(c1-c2)<eps:
        flg+=1
    else:
        print('/')
        
    if flg==4:
        return 0
    else:
        print(a,b)
        return 1


"""

"""
ncycle=20
eps=1e-3

a=[1,0,1]
b=[0,1,1]

flg=math_check(a,b)
print("flg",flg)

