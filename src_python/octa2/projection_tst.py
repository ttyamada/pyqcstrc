#if __name__ == '__main__':
    
# test
import sys
import itertools
import cython
import numpy as np
import random

from numpy.typing import NDArray
from projection import (projection3)

DTYPE_int = int
#DTYPE_int = np.int64

vt0=[0,0,1]
vt1=[1,0,1]

vt =([vt1,vt1,vt0,vt0,vt0,vt0])
print("vt",vt)
vti = projection3(vt)
print("vti",vti)




