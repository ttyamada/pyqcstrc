cimport sys
cimport numpy as np
cimport pyqcstrc.qnnum.qnnum as qnn
cimport pyqcstrc.qnvec.qnvec as qnv
cimport pyqcstrc.qnmat.qnmat as qnm
#from goto cimport with_goto

cdef abs(a:qnn.Qnnum):
    qnzero=qnn.Qnnum([0,0,1],a.N)
    if(a<qnn.qnzero):
        return -a
    if(a>=qnn.qnzero):
        return a

cdef qnmatinv(a:qnm.Qnmat,n:np.int64):
    #subroutine matinv(a,nm,n,b,m,determ,ipivot,index,pivot) 
    #
    #    n*n matrix a is replaced by its inverse matrix a**-1
    #    if m is nonzero vector b is replaced by a**-1*b
    #
    #      dimension a(nm,nm),b(nm),ipivot(nm),index(nm,2),pivot(nm)
    a=np.ndarray((n,n),dtype=qnn.Qnnum) 
    b=np.ndarray(n,dtype=qnn.Qnnum)
    ipivot=np.ndarray(n,dtype=qnn.Qnnum) 
    index=np.ndarray((n,2),dtype=qnn.Qnnum)

    det=1.0 
    for  j in range(n):
        ipivot[j]=0
    
    for i in range(n): 
        t=qnn.Qnnum([0,0,1],a.N)
        for j in range(n):
            if ipivot[j]==1:
                continue
            for k in range(n):
                if ipivot[k]-1<0:
                    if abs(t)>=abs(a[j][k]):
                        continue
                    ir=j
                    ic=k
                    t=a[j][k]
                elif ipivot[k]-1>0:
                    return
    
        ipivot[ic]=ipivot[ic]+1
        if ir!=ic:
            det=-det
            for l in range(n):
                t=a[ir][l]
                a[ir][l]=a[ic][l]
                a[ic][l]=t

        index[i][1]=ir
        index[i][2]=ic
        pivot[i]=a[ic][ic]
        det=det*pivot[i]
        a[ic][ic]=1.0
        for l in range(n):
            a[ic][l]=a[ic][l]/pivot[i]

        for l1 in range(n):
            if l1==ic:
                continue
            t=a[l1][ic]
            a[l1][ic]=0.0
            for l in range(n):
                a[l1][l]=a[l1][l]-a[ic][l]*t
    for i in range(n):
        l=n+1-i
        if index[l][1]==index[l][2]:
            continue
        ir=index[l][1]
        ic=index[l][2]
        for k in range(n):
            t=a[k][ir]
            a[k][ir]=a[k][ic]
            a[k][ic]=t
       
#@with_goto     
cdef qsort(x:qnv.Qnvec,ip:np.array,nx: np.int64):
    #     quick sort (ascending order of x)
    #     nx: the number of data x
    #     ip: the initial order
    #     st: a work array
    
    cdef setlrs():
        #label .l1
        l=st[s][0] 
        r=st[s][1] 
        s=s-1 
#2   continue
    cdef setijxt():
        #label .l2
        i=l 
        j=r 
        xt=x[(l+r)/2]

    st=np.ndarray((nx,2),dtype=np.int63)
    if nx==0: return 
    
    for i in range(nx): 
        ip[i]=i

    s=1 
    st[1][0]=1 
    st[1][1]=nx 

    setlrs()
    setijxt()
    
    while True:
        while True:
            #label .l3
            if i<nx: 
                if x[i]<xt:
                    i=i+1
                    continue
                else:
                    break
            else:
                break
        while True:
            if j>1:
                if xt<x[j]:
                    j=j-1
                    continue
                else:
                    break
            else:
                break

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
            setijxt()
            continue
        if s!=0: 
            setlrs()
            setijxt()
            continue
        else:
            break
    return 

#END subroutine qsortr


#END subroutine

#if __name__ == '__main__':
    # test
