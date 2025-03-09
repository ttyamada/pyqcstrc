import numpy as np
from numpy.typing import NDArray
import cython

import crsys
import qnnum as qnn
import qnvec as qnv

# super class for edges,triangles and tetrahedra
# which are composed of 2 3 and 4 points
# they are represented by a[2][n] a[3][n] and a[4][n]
# in nD space
# as a special case, a point is represented by a[n]

class QnNdarray(np.ndarray):
    
    def __new__(cls,shape):
        return super().__new__(cls,shape,dtype=qnn.Qnnum)

    def __init__(self, shape):
        self.shape=shape
            
    def __add__(self,b):
        # shap should be (n)
        return add_vectors(self,b)
    
    def __sub__(self,b):
        # shap should be (n)
        return sub_vectors(self,b)
   
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
def anya(vec:NDArray[qnn.Qnnum], shape)->QnNdarray:
    qnva=QnNdarray(shape)
    ndim=vec.ndim
    if ndim==1:
        for i in range(shape[0]):
            qnva[i][j]=vec[i][j]
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
        
def add_vectors(vt1: QnNdarray, vt2:QnNdarray) -> QnNdarray:
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
    a=np.zeros(vt1.shape,dtype=qnn.Qnnum)
    for i in range(len(vt1)):
        a[i]=qnv.add(vt1[i],vt2[i])
    return a

def sub_vectors(vt1: QnNdarray, vt2:QnNdarray) -> QnNdarray:
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
    if vt1.ndim==1 and vt2.ndim==1:
        #const=np.array([-1,0,1],dtype=qnn.Qnnum)
        #vt2=mul_vector(vt2,const)
        #return add_vectors(vt1,vt2)
        return qnv.sub(vt1,vt2)
    else:
        print('incorrect shape in sub_vectors')
        return
    
def shift_vectors(vs:QnNdarray, v:QnNdarray) -> QnNdarray:
    if vs.ndim==1:  #3:
        a=np.zeros(vs.shape,dtype=qnn.Qnnum)
        la=vs.shape
        for i,v1 in enumerate(vs):  #range(la[0]):
            a[i]=add_vectors(v1,v)
        return a
    elif vs.ndim==2:  #4:
        a=np.zeros(vs.shape,dtype=qnn.Qnnum)
        for i1,v1 in enumerate(vs):
            for i2,v2 in enumerate(v1):
                a[i1][i2]=add_vectors(v2,v)
    else:
        print('incorrect shape in shift_vectors')
        return
        
