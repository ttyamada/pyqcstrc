import sys
import numpy as np
import time
import cython 
from add_tst import (add_tst0,add_tst1)
from add_tst00 import (add_tst00)

a0=np.array([2,0,1],dtype=np.int64)
b0=np.array([0,1,1],dtype=np.int64)
c0=np.array([0,0,1],dtype=np.int64)
print("a0",a0); print("b0",b0); print("c0",c0)

a=cython.declare(cython.long[:],a0)
b=cython.declare(cython.long[:],b0)
c=cython.declare(cython.long[:],c0)
print("a",a); print("b",b); print("c",c)

n: cython.int= 100000
add_tst00(a,b,n)

add_tst0(a,b,n)

add_tst1(a,b,n)

