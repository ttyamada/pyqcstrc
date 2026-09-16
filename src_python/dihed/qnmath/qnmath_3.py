import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnndarray as qna
from numpy.typing import(NDArray)

def qsort(x:qnv.Qnvec,ip:np.array,nx: np.int64) -> qnv.Qnvec:
    #     quick sort (ascending order of x)
    #     nx: the number of data x
    #     ip: the initial order
    #     st: a work array

    def setlrs(s:np.int64,st):
        #label .l1
        l=st[s][0] 
        r=st[s][1] 
        s=s-1 
        return l,r,s

    def setijxt(l:np.int64,r:np.int64,x):
        #label .l2
        i=l 
        j=r 
        lr=(int)((l+r)/2)
        xt=x[lr]
        return i,j,xt
    
    xc=np.copy(x)
    qnv.printqnv("qnvs",xc) # for test
    st=np.ndarray((nx,2),dtype=np.int64)
    if nx==0: return 

    print("nx",nx) # for test
    for i in range(nx): 
        ip[i]=i

    s=0  # s=1 
    st[0][0]=0     # st[1][0]=1 
    st[0][1]=nx-1  # st[1][1]=nx 

    l,r,s=setlrs(s,st)
    i,j,xt=setijxt(l,r,x)
    
    while True:
        #label .l3
        while True:
            if i<nx-1: #if i<nx: 
                if x[i]<xt:
                    i+=1
                    continue
                else:
                    break
            else:
                break
        while True:
            if j>0: #if j>1:
                if xt<x[j]:
                    j-=1
                    continue
                else:
                    break
            else:
                break

        if i<=j:
            temp=qnn.copy(x[j])  #temp=xc[j]
            x[j]=qnn.copy(x[i])  #xc[j]=qnn.copy(xc[i])
            x[i]=qnn.copy(temp)  #xc[i]=temp
            itemp=ip[j]
            ip[j]=np.copy(ip[i])
            ip[i]=np.copy(itemp)
            if i<=nx-1 and j>=0:  #if i<=nx and j>=1:
                i+=1
                j-=1
                continue
            else:
                break
            
        if j-l>r-i:
            if l<j:
                s=s+1
                st[s][0]=l
                st[s][1]=j
            l=i 
        else:
            if i<r:
                s=s+1
                st[s][0]=i
                st[s][1]=r
            r=j
        if l<r: 
            i,j,xt=setijxt(l,r,x)  #i,j,xt=setijxt(l,r,xc)
            continue
        if s!=-1:  #if s!=0: 
            l,r,s=setlrs(s,st)
            i,j,xt=setijxt(l,r,x) #i,j,xt=setijxt(l,r,xc)
            continue
        else:
            break
    return x  #return xc
