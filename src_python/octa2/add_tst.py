import numpy as np
import cython
import time

def qnreduce0(x):
    g=np.gcd.reduce(x)
    x[0]=(x[0]/g)
    x[1]=(x[1]/g)
    x[2]=(x[2]/g)
    if x[2]<0:
        x[0]=-x[0]; x[1]=-x[1]; x[2]=-x[2]
    return x

def add0(a, b):
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

    x=np.array([0,0,0])
    x[0]=a[0]*b[2]+b[0]*a[2]
    x[1]=a[1]*b[2]+b[1]*a[2]
    x[2]=a[2]*b[2]
    #x: DTYPE_int=np.array([c[0],c[1],c[2]],dtype=DTYPE_int)
    x=qnreduce0(x)
    return x  #np.array([c[0],c[1],c[2]])

def qnreduce(x:cython.long[:]) -> cython.long[:]:
    #g: np.int64
    g=np.gcd.reduce(x)
    x[0]=(x[0]/g)
    x[1]=(x[1]/g)
    x[2]=(x[2]/g)
    if x[2]<0:
        x[0]=-x[0]; x[1]=-x[1]; x[2]=-x[2]
    return x

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
    #x0 = np.array([3],dtype=np.int64)
    #x=cython.declare(cython.long[:],x0)
    x=np.array([0,0,0],dtype=np.int64)
    x[0]=a[0]*b[2]+b[0]*a[2]
    x[1]=a[1]*b[2]+b[1]*a[2]
    x[2]=a[2]*b[2]
    #x: DTYPE_int=np.array([c[0],c[1],c[2]],dtype=DTYPE_int)
    x=qnreduce(x)
    return x  #np.array([c[0],c[1],c[2]])

def add_tst0(a,b,n):
    print("n",n)
    start_time: cython.double = time.time()
    i: cython.int
    for i in range(n):
        c=add0(a,b)

    end_time: cython.double = time.time()
    elapsed_time: cython.double = end_time - start_time
    
    print("elapsed time", elapsed_time, "sec")
    print("time per cycle", elapsed_time / n)
    
def add_tst1(a: cython.long[:],b: cython.long[:],n: cython.int):
    start_time: cython.double = time.time()
    for i in range(n):
        c=add(a,b)

    end_time: cython.double = time.time()
    elapsed_time: cython.double = end_time - start_time

    print("elapsed time", elapsed_time, "sec")
    print("time per cycle", elapsed_time / n)
  