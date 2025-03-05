import numpy as np
import cython
import qnvec.qnvec as qnv
import qnnum.qnnum as qnn

def qsort(x:qnv.Qnvec,ip:np.array,nx: np.int64):
    #     quick sort (ascending order of x)
    #     nx: the number of data x
    #     ip: the initial order
    #     st: a work array

    st=np.ndarray((nx,2),dtype=np.int63)
    if nx==0: return 
    for i in range(nx): 
        ip[i]=i
    
    s=1 
    st[1][0]=1 
    st[1][1]=nx 
    #1   continue
    label .l1
    l=st[s][0] 
    r=st[s][1] 
    s=s-1 
#2   continue
    label .l2
    i=l 
    j=r 
    xt=x[(l+r)/2]
    #3  contine
    label .l3
    if i<nx: 
        if x[i]<xt:
            i=i+1
            goto .l3  #go to 3
    #4  continue
    label .l4
    if j>1:
        if xt<x[j]:
            j=j-1
            goto .l4  #go to 4

    if i<=j:
        temp=x[j]
        x[j]=x[i]
        x[i]=temp
        itemp=ip[j]
        ip[j]=ip[i]
        ip[i]=itemp
        if i<=nx and j>=1:
            i=i+1
            j=j-1
            goto .l3  #go to 3
    if j-l>r-i:
        if i<r:
            s=s+1
            st[s][0]=i
            st[s][1]=r
        r=j
    else:  
        label .l5
        if l<j:
            s=s+1
            st[s][0]=l
            st[s][1]=j
        l=i 
    #6  continue
    label .l6
    if l<r: 
        goto .l2 #go to 2 
    if s!=0: 
        goto .l1 # go to 1
    return 
