import numpy as np
from numpy.typing import NDArray
import cython
from typing import Self

import crsys
import qnnum as qnn
#import qnvec as qnv

# super class for edges,triangles and tetrahedra
# which are composed of 2 3 and 4 points
# they are represented by a[2][n] a[3][n] and a[4][n]
# in nD space
# as a special case, a point is represented by a[n]

#add,sub,iadd,isub should be implemented in subclass

class QnNdarray(NDArray):
    def __new__(cls,shape):
        return super().__new__(cls,shape,dtype=qnn.Qnnum)

    def __init__(self,shape):
        self.shape=shape

def qnndarray_init():
    global n,N,isys
    isys=crsys.isys
    n=crsys.n
    N=crsys.N
     
def copy(qna1: QnNdarray) -> QnNdarray:
    return np.copy(qna1)

def zeros(shape) -> QnNdarray:
    qna=QnNdarray(shape)
    qn0=qnn.zero()
    it = np.nditer(qna, flags=['multi_index','refs_ok'], op_flags=['readwrite'])
    while not it.finished:  # loop up to last index
        it[0] = qn0
        idx = it.multi_index
        #print('idx=', idx ,', self[idx]=', self[idx], ', it[0]=', it[0]) # for test
        it.iternext()   #it : next index
    return qna
    
# any kind of 3D array assumed
def anya(vec:NDArray[qnn.Qnnum],shape)->QnNdarray:
    #print("shape",shape) # for test
    #print("len(shape)",len(shape))  # for test
    qnva=QnNdarray(shape)
    ndim=len(shape)
    #print("ndim",ndim) # for test
    if ndim==1:
        for i in range(shape[0]):
            qnva[i]=vec[i]
    elif ndim==2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                qnva[i][j]=vec[i][j]
    elif ndim==3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    qnva[i][j][k]=vec[i][j][k]
    return qnva

# only ndim=1,2,3
def printqndm(str:str, qnm:QnNdarray):
    ndim=qnm.ndim
    #print("qnm.ndim",qnm.ndim) # for test
    #print("qnm.shape",qnm.shape) # for test
    print(str)
    if ndim==1: # for a vector
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            print(qnn.qn2npa(qnm[i]),end=" ")
        print("]")
    elif ndim==2: # for a matrix
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            for j in range(qnm.shape[1]):
                print(qnn.qn2npa(qnm[i][j]),end=" ")
            print("]")
        print("")
    elif ndim==3: # for a matrix array
        for i in range(qnm.shape[0]):
            print("")
            for j in range(qnm.shape[1]):
                print("[",end=" ")
                for k in range(qnm.shape[2]):
                    print(qnn.qn2npa(qnm[i][j][k]),end=" ")
                print("]")
        print("")
    else:
        print("ord in printqnm should be 1, 2 or 3 but",ord); exit()
        
# insert, append will be necessary for qnnpyDelaunay
# insert a point in the ponts array 
def insert(index,point):
    return np.insert(index,point)

def append(point):
    return np.append(point)

