import sys
import numpy as np
import cython

import crsys
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnndarray as qna

def qnmath_init():
    global n,N
    n=crsys.n
    N=crsys.N

def abs(a:qnn.Qnnum):
    #N=a.N
    qn0=qnn.Qnnum([0,0,1],N)
    if(a<qn0):
        return -a
    if(a>=qn0):
        return a

#def qnmatinv(a:qnm.Qnmat,n:np.int64):
#    return np.linalg.inv(a)
    
# this should be a function for @ operator
def qnmatinv(a_i:qnm.Qnmat,n:np.int64): # qnmatrix inversion
    # return inversion matrix of a_i
    # n is the order of a_i (nxn qnnumber matrix)
    a=qnm.copy(a_i)
    #a=np.copy(a_i)
    pivot=np.ndarray(n,dtype=qnn.Qnnum)
    ipivot=np.ndarray(n,dtype=np.int64) 
    index=np.ndarray((n,2),dtype=np.int64)
    #N=a[0][0].N
    qn0=qnn.Qnnum([0,0,1],N)  # 0
    qn1=qnn.Qnnum([1,0,1],N)  # 1
    
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
                    if abs(t)>=abs(a[j][k]):
                        continue
                    ir=j
                    ic=k
                    t=qnn.copy(a[j][k])
                elif ipivot[k]>0: #elif ipivot[k]-1>0:
                    return a
    
        ipivot[ic]=ipivot[ic]+1
        if ir!=ic:
            det=-det
            for l in range(n):
                swap=qnn.copy(a[ir][l])
                a[ir][l]=qnn.copy(a[ic][l])
                a[ic][l]=swap

        index[i][0]=ir
        index[i][1]=ic
        pivot[i]=qnn.copy(a[ic][ic])
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
        l=n-1-i  # l=n+1-i
        if index[l][0]==index[l][1]:
            continue
        ir=index[l][0]
        ic=index[l][1]
        for k in range(n):
            t=a[k][ir]
            a[k][ir]=qnn.copy(a[k][ic])
            a[k][ic]=t
    #qnm.printqnm("a in qnmatinv",a)  # for test
    return a

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
#2   continue
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
    
    for i in range(nx): 
        ip[i]=i

    s=0  # s=1 
    st[0][0]=0     # st[1][0]=1 
    st[0][1]=nx-1  # st[1][1]=nx 

    l,r,s=setlrs(s,st)
    i,j,xt=setijxt(l,r,x)
    
    while True:
        while True:
            #label .l3
            if i<nx-1: #if i<nx: 
                if x[i]<xt:
                    i=i+1
                    continue
                else:
                    break
            else:
                break
        while True:
            if j>0: #if j>1:
                if xt<x[j]:
                    j=j-1
                    continue
                else:
                    break
            else:
                break

        if i<=j:
            temp=xc[j]
            xc[j]=qnn.copy(xc[i])
            xc[i]=temp
            itemp=ip[j]
            ip[j]=np.copy(ip[i])
            ip[i]=itemp
            if i<=nx-1 and j>=0:  #if i<=nx and j>=1:
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
            i,j,xt=setijxt(l,r,xc)
            continue
        if s!=-1:  #if s!=0: 
            l,r,s=setlrs(s,st)
            i,j,xt=setijxt(l,r,xc)
            continue
        else:
            break
    return xc

# for test float version of qsort   
def qsort_f(x:np.array,ip:np.array,nx: np.int64):
    #     quick sort (ascending order of x)
    #     nx: the number of data x
    #     ip: the initial order
    #     st: a work array
    def setlrs(s:np.int64, st:np.array):
        #label .l1
        l=st[s][0] 
        r=st[s][1] 
        s=s-1 
        return l,r,s
#2   continue
    def setijxt(l:np.int64,r:np.int64,x:np.array):
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
    
    for i in range(nx): 
        ip[i]=i

    s=0  #1 
    st[0][0]=0  #st[1][0]=1 
    st[0][1]=nx-1  #st[1][1]=nx 

    l,r,s=setlrs(s,st)
    i,j,xt=setijxt(l,r,xc)
    
    while True:
        while True:
            #label .l3
            if i<nx-1:  #if i<nx: 
                if xc[i]<xt:
                    i=i+1
                    continue
                else:
                    break
            else:
                break
        while True:
            if j>0:  #if j>1:
                if xt<xc[j]:
                    j=j-1
                    continue
                else:
                    break
            else:
                break

        if i<=j:
            temp=xc[j]
            xc[j]=np.copy(xc[i])
            xc[i]=temp
            itemp=ip[j]
            ip[j]=np.copy(ip[i])
            ip[i]=itemp
            if i<=nx-1 and j>=0: #if i<=nx and j>=1:
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
            i,j,xt=setijxt(l,r,xc)
            continue
        if s!=-1:  #if s!=0: 
            l,r,s=setlrs(s,st)
            i,j,xt=setijxt(l,r,xc)
            continue
        else:
            break
    return xc


def centroid(obj: qnv.Qnvec) -> qnv.Qnvec:
    """geometric center, centroid of tetrahedron, triangle or edge, in qnvec.

    Parameters
    ----------
    obj: array
        6-dimensional vector in qnvec
    
    Returns
    -------
    centroid in qnvec
    """
    #N=obj[0].N
    n_=obj.shape[0]
    num=len(obj) # length of obj
    v2=qnv.Qnvec(n_,N) # nD zero qnvector
    qnnum=qnn.Qnnum([1,0,num],N) # 1/num
    for i1 in range(num):
        v2=v2+obj[i1]
    v0=v2*qnnum
    return v0

# needless???
def centroid_obj(obj: qnv.Qnvec) -> qnv.Qnvec:
    """geometric center, centroid of tetrahedron, in qnvec.

    Parameters
    ----------
    tetrahedron: array
        6-dimensional vector in qnvec
    
    Returns
    -------
    centroid in qnvec
    """
    #print('centroid_obj')
    
    #  geometric center, centroid of OBJ
    #N=obj[0].N
    shape=qnv.shape
    n=shape[1]
    len=shape[0]  # 1/len(obj)
    tmp=qnv.Qnvec(n,N) # zero vector
    for thd in obj:
        tmp=tmp+thd
    tmp=tmp*len
    return tmp

def det_matrix(mtx: qnm.Qnmat, n_:np.int64) -> qnn.Qnnum:
    if n_==2:
        return det_matrix_2d(mtx)
    elif n_==3:
        return det_matrix_3d(mtx)

def det_matrix_3d(mtx: qnm.Qnmat) -> qnn.Qnnum:
    """Determinant of 3x3 matrix, mtx, in qnnumber
    
    Parameters
    ----------
    mtx: array
        3x3 matrix in qnnumer

    Returns
    -------
    determinant in qnnumber
    """
    #N=mtx.N
    shape=mtx.shape
    if shape[0]!=3:
        print("shape of mtx in det_matrix_3d should be (3,3) but",shape); exit(0)
    det=qnn.Qnmtrx([0,0,1],N) # zero qnnumber
    det=det+mtx[0][0]*mtx[1][1]*mtx[2][2]
    det=det+mtx[0][1]*mtx[1][2]*mtx[2][0]    
    det=det+mtx[0][2]*mtx[1][0]*mtx[2][1]
    det=det-mtx[0][2]*mtx[1][1]*mtx[2][0]
    det=det-mtx[0][1]*mtx[1][0]*mtx[2][2]    
    det=det-mtx[0][0]*mtx[1][2]*mtx[2][1]

    return det

def det_matrix_2d(mtx: qnm.Qnmat) -> qnn.Qnnum:
    """Determinant of 3x3 matrix, mtx, in qnnumber
    
    Parameters
    ----------
    mtx: array
        2x2 matrix in qnnumer

    Returns
    -------
    determinant in qnnumber
    """
    #N=mtx.N
    shape=mtx.shape
    if shape[0]!=2:
        print("shape of mtx in det_matrix_2d should be (3,3) but",shape); exit(0)
    det=qnn.Qnmtrx([0,0,1],N) # zero qnnumber
    det=det+mtx[0][0]*mtx[1][1]
    det=det-mtx[0][1]*mtx[1][0]
    return det


# this should be a function
def matrixtr(mtx: qnm.Qnmat) -> qnm.Qnmat:
    """ return transposed matrix of mtx """
    #N=mtx[0][0].N
    n_=mtx.shape[0]
    mtxt=qnm.Qnmat(n_,N)
    for i in range(n_):
        for j in range(n_):
            mtxt[i][j]=qnn.copy(mtx[j][i])
    return mtxt
        

