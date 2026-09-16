import sys
import numpy as np
from numpy.typing import NDArray
from typing import Self

import crsys as crs
import qnnum as qnn
import qnndarray as qna


def printqnv(str:str,qnv:qna.QnNdarray):
    shape=qnv.shape
    ndim=len(shape)

    if ndim==1 :  # for qnvector
        print(str,"[",end=" ")
        for i in range(shape[0]):
            print(qnn.qn2npa(qnv[i]),end=" ")
        print("]")
    elif ndim==2: #  for triangle/tetrahedron or qnmatrix
        for i in range(shape[0]):
            print(str,"[",end=" ")
            for j in range(shape[1]):
                print(qnn.qn2npa(qnv[i][j]),end=" ")
            print("]")
        #print("")
    elif ndim==3: # for triangles/tetrahedra
        for i in range(shape[0]):
            print(str,"[",end=" ")
            for j in range(shape[1]):
                for k in range(shape[0]):
                    print(qnn.qn2npa(qnv[i][j][k]),end=" ")
                print("]")
            #print("")
        print("")
    else:
        print("ndim should be 1 2 or 3 but",ndim)
        exit()

def printqnv2(str:str,qnv1:Qnvec,qnv2:Qnvec):
    print(str,"[",end=" ")
    n1=qnv1.shape
    for i in range(n1):
        j=qnv1[i]
        print(qnn.qn2npa(j),end=" ")
    print("] [",end="")
    n2=qnv2.shape
    for i in range(n2):
        j=qnv1[i]
        print(qnn.qn2npa(j),end="]")
        
def printqnvs(str:str,qnv1:Qnvec):
    shape=qnv1.shape
    ndim=len(shape)
    #print("shape",shape,"ndim",ndim)  # for test
    n=shape[0]
    print(str)
    for j in range(n):
        printqnv("",qnv1[j])
     
def eq(qnv1:Qnvec, qnv2:Qnvec):
    if len(qnv1.shape) != len(qnv2.shape):
        return False
    n_=qnv1.n
    for i in range(n_):
        if qnv1[i]!=qnv2[i]:
            return False
    return True

def not_eq(qnv1:Qnvec, qnv2:Qnvec):
    n_=qnv1.n
    for i in range(n_):
        if qnv1[i]!=qnv2[i]:
            return True
    return False
    
    
    
