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

def sub(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    #a=np.empty(mat1.shape, dtype=qnn.Qnnum)
    shape=ma1.shape
    n1=len(ma1.shape)
    n2=len(ma2.shape)
    #N=ma1[0][0].N
    a=Qnmat(shape)
    if(n1==1 and n2==1): # vectors
        for i in range(shape[0]):
            a[i]=ma1[i]-ma2[i]  #add(v1[i],v2[i])
        return a
    elif(n1==2 and n2==2): # matrices
        for i in range(shape[0]):
            for j in range(shape[1]):
                a[i][j]=ma1[i][j]-ma2[i][j]  #add(v1[i],v2[i])
        return a 
    
def iadd(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    ma1=add(ma1,ma2)
    return ma1

def isub(ma1:Qnmat, ma2:Qnmat) -> Qnmat:
    ma1=sub(ma1,ma2)
    return ma1

def mul_scl(ma1: Qnmat, scl: qnn.Qnnum) -> Qnmat:
    ndim=len(ma1.shape)
    if ndim==1:
        for i in range(ma1.shape[0]):
            ma1[i]=ma1[i]*scl
    if ndim==2:
        for i in range(ma1.shape[0]):
            for j in range(ma1.shape[1]):
                ma1[i][j]=ma1[i][j]*scl
    return ma1

    
# for ma1@ma2 (ma1 and ma2 should be qnvec or qnmat)
def mul(ma1: Qnmat, ma2: Qnmat) -> Qnmat: 
    ndm1=len(ma1.shape)
    ndm2=len(ma2.shape)
    #print("ndm1=",ndm1,"ndm2=",ndm2)  # for test
    if ndm1==1 and ndm2==1:  # dot product
        if ma1.shape[0]==ma2.shape[0]:
            ma3=qnn.zero()
            for i in range(ma1.shape[0]):
                ma3+=ma1[i]*ma2[i]
        else:
            print("ma1.shape[0] should be equal to ma2.shape[0] for ndim1==ndim2"); exit()
    elif ndm1==1 and ndm2==2:  # vec*matrix
        if ma1.shape[0]==ma2.shape[0]:
            ma3=qnv.zerov(ma2.shape[1])
            for j in range(ma2.shape[1]):
                for i in range(ma1.shape[0]):
                    ma3[j]+=ma1[i]*ma2[i][j]
        else:
            print("ma1.shape[0] should be equal to ma2.shape[0] for ndim1=1 and ndim2=2"); exit()
    elif ndm1==2 and ndm2==1:  # matrix*vec
        if ma1.shape[1]==ma2.shape[0]:
            ma3=qnv.zerov(ma1.shape[0])
            for j in range(ma2.shape[0]):
                for i in range(ma1.shape[0]):
                    ma3[j]+=ma1[j][i]*ma2[i]
        else:
            print("ma1.shape[1] should be equal to ma2.shape[0] for ndim1=2 and ndim2=1"); exit()
    elif ndm1==2 and ndm2==2:
        if ma1.shape[0]==ma1.shape[1] and ma2.shape[0]==ma2.shape[1] and ma1.shape[0]==ma2.shape[0]:
            ma3=qnv.zerov(ma1.shape)
            for k in range(ma2.shape[0]):
                for j in range(ma1.shape[0]):
                    for i in range(ma1.shape[0]):
                        ma3[k][j]+=ma1[k][i]*ma2[i][j]
        else:
             print("square matrix is assumed for ndm1=ndm2=2"); exit()

    #ma3=np.matmul(ma1,ma2,dtype=qnn.Qnnum)
    return ma3

# for similarity transformation
# not confirmed yet
def pow(ma: Qnmat, n_: int) -> Qnmat:
    """
    """
    (mx,my)=ma.shape
    #N=ma[0][0].N
    if mx==my:
        if n_==0:
            return np.identity(mx)
#        elif n<0:
#            tmp=unitm(n,N)
#            mai = copy(ma) # copy for qnmatinv
#            qmt.qnmatinv(mai,n) #???
#            for i in range(-n):
#                tmp=np.dot(tmp,inva)
#            return tmp
        else:
            tmp=np.unitm(n_,N)
            for i in range(n_):
                tmp=np.dot(tmp,ma)
            return tmp
    else:
        print('matrix has not regular shape')
        exit()
