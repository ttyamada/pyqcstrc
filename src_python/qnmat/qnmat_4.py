import sys
import numpy as np
import cython
from typing import Self
from numpy.typing import NDArray

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnndarray as qna
from numpy.typing import(NDArray)

def qnm2npa(a) -> np.ndarray:
    # Qnmatrix to np.array converter
    shape=a.shape
    b=np.zeros(shape,dtype=qnn.Qnnum)
    for i in range(shape[0]):
        for j in range(shape[1]):
            b[i][j]=[a[i][j].n[0],a[i][j].n[1],a[i][j].n[2]]
    return b

def qnm2flt(a) -> np.ndarray:
    shape=a.shape
    b=np.zeros(shape,dtype=qnn.Qnnum)
    for i in range(shape[0]):
        for j in range(shape[1]):
            b[i][j]=(a[i][j].n[0]+a[i][j].n[1]*np.sqrt(N))/a[i][j].n[2]
    return b

# get qnmat from int matrix
def intm2qnm(a:np.array,n_:np.int64) -> Qnmat:
    shape=a.shape
    b=Qnmat(shape) #qnnum zero vector
    for i in range(shape[0]):
        for j in range(shape[1]):
            b[i][j]=qnn.int2qn(a[i][j],N)
    return b

#def printqnm(str:str,qnm:Qnmat):
#def printqnm(str:str,qna:QnNdarray):
def printqnm(str:str,qnm:qna.QnNdarray):
    #ndim=2  # 
    ndim=len(qnm.shape)
    shape=qnm.shape
    #if shape[1]==0:
    #    ndim=1
    #elif shape[2]==0:
    #    ndim=2
    #else:
    #    ndim=3

    print(str)
    if ndim==1:
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            print(qnn.qn2npa(qnm[i]),end=" ")
        print("]")
    elif ndim==2:
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            for j in range(qnm.shape[1]):
                print(qnn.qn2npa(qnm[i][j]),end=" ")
            print("]")
        print("")
    elif ndim==3: # for several matrices
        for i in range(qnm.shape[0]):
            print("")
            for j in range(qnm.shape[1]):
                print("[",end=" ")
                for k in range(qnm.shape[2]):
                    print(qnn.qn2npa(qnm[i][j][k]),end=" ")
                print("]")
        print("")
    else:
        print("ord in printqnm should be 1 2 or 3 but",ord); exit()


def printfm(str:str,fm:NDArray[np.float64]):
    #ndim=2  # 
    ndim=len(fm.shape)
    shape=fm.shape
    #if shape[1]==0:
    #    ndim=1
    #elif shape[2]==0:
    #    ndim=2
    #else:
    #    ndim=3

    print(str)
    if ndim==1:
        for i in range(fm.shape[0]):
            print("[",end=" ")
            print(fm[i],end=" ")
        print("]")
    elif ndim==2:
        for i in range(fm.shape[0]):
            print("[",end=" ")
            for j in range(fm.shape[1]):
                print(fm[i][j],end=" ")
            print("]")
        print("")
    elif ndim==3: # for several matrices
        for i in range(fm.shape[0]):
            print("")
            for j in range(fm.shape[1]):
                print("[",end=" ")
                for k in range(fm.shape[2]):
                    print(fm[i][j][k],end=" ")
                print("]")
        print("")
    else:
        print("ord in printfm should be 1 2 or 3 but",ord); exit()
