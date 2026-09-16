import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnndarray as qna
from numpy.typing import(NDArray)

# fpr check float version     
# this should be a function for @ operator  
def matinv_f(a_i:np.matrix,n:np.int64): # qnmatrix inversion
    # return inversion matrix of a_i
    # n is the order of a (nxn qnnumber matrix)
    a=np.copy(a_i)
    pivot=np.ndarray(n,dtype=float)
    ipivot=np.ndarray(n,dtype=np.int64) 
    index=np.ndarray((n,2),dtype=np.int64)
    #N=a[0][0].N
    qn0=0.0 # for float version
    qn1=1.0
    
    det=qn1  #1.0 
    for  j in range(n):
        ipivot[j]=-1  #ipivot[j]=0
    
    for i in range(n): 
        t=qn0
        for j in range(n):
            if ipivot[j]==0: #if ipivot[j]==1:
                continue
            for k in range(n):
                if ipivot[k]<0: #if ipivot[k]-1<0:
                    if np.abs(t)>=np.abs(a[j][k]):
                        continue
                    ir=j
                    ic=k
                    t=np.copy(a[j][k])
                elif ipivot[k]>0: #elif ipivot[k]-1>0:
                    return a
    
        ipivot[ic]=ipivot[ic]+1
        if ir!=ic:
            det=-det
            for l in range(n):
                swap=np.copy(a[ir][l])
                a[ir][l]=np.copy(a[ic][l])
                a[ic][l]=swap

        index[i][0]=ir
        index[i][1]=ic
        pivot[i]=np.copy(a[ic][ic])
        det=det*pivot[i]
        a[ic][ic]=qn1  #1.0
        for l in range(n):
            a[ic][l]=a[ic][l]/pivot[i]

        for l1 in range(n):
            if l1==ic:
                continue
            t=a[l1][ic]
            a[l1][ic]=qn0  #0.0
            for l in range(n):
                a[l1][l]=a[l1][l]-a[ic][l]*t
    for i in range(n):
        l=n-1-i  #l=n+1-i
        if index[l][0]==index[l][1]:
            continue
        ir=index[l][0]
        ic=index[l][1]
        for k in range(n):
            t=a[k][ir]
            a[k][ir]=np.copy(a[k][ic])
            a[k][ic]=t
    return a
       
