import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnmat.qnmat as qnm

def abs(a:qnn.Qnnum):
    N=a.N
    qn0=qnn.Qnnum([0,0,1],N)
    if(a<qn0):
        return -a
    if(a>=qn0):
        return a

def qnmatinv(a:qnm.Qnmat,n:np.int64): # qnmatrix inversion
    # a is replaced by its inversion matrix
    # n is the order of a (nxn qnnumber matrix)
    ipivot=np.ndarray(n,dtype=qnn.Qnnum) 
    index=np.ndarray((n,2),dtype=np.int64)
    N=a.N
    qn0=qnn.Qnnum([0,0,1],N)
    qn1=qnn.Qnnum([1,0,1],N)
    det=qn1  #1.0 
    for  j in range(n):
        ipivot[j]=0
    
    for i in range(n): 
        t=qn0
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

        index[i][0]=ir
        index[i][1]=ic
        ipivot[i]=a[ic][ic]
        det=det*ipivot[i]
        a[ic][ic]=1.0
        for l in range(n):
            a[ic][l]=a[ic][l]/ipivot[i]

        for l1 in range(n):
            if l1==ic:
                continue
            t=a[l1][ic]
            a[l1][ic]=0.0
            for l in range(n):
                a[l1][l]=a[l1][l]-a[ic][l]*t
    for i in range(n):
        l=n+1-i
        if index[l][0]==index[l][1]:
            continue
        ir=index[l][0]
        ic=index[l][1]
        for k in range(n):
            t=a[k][ir]
            a[k][ir]=a[k][ic]
            a[k][ic]=t
       

def qsort(x:qnv.Qnvec,ip:np.array,nx: np.int64):
    #     quick sort (ascending order of x)
    #     nx: the number of data x
    #     ip: the initial order
    #     st: a work array
    
    def setlr(s,st):
        #label .l1
        l=st[s][0] 
        r=st[s][1] 
        s=s-1 
        return l,r,s
#2   continue
    def setijxt(l,r,x):
        #label .l2
        i=l 
        j=r 
        xt=x[(l+r)/2]
        return i,j,xt

    st=np.ndarray((nx,2),dtype=np.int64)
    if nx==0: return 
    
    for i in range(nx): 
        ip[i]=i

    s=1 
    st[1][0]=1 
    st[1][1]=nx 

    l,r,s=setlr(s,st)
    i,j,xt=setijxt(l,r,x)
    
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
            i,j,xt=setijxt(l,r,x)
            continue
        if s!=0: 
            l,r,s=setlr(s,st)
            i,j,xt=setijxt(l,r,x)
            continue
        else:
            break


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
    N=obj[0].N
    n=obj.shape[0]
    num=len(obj) # length of obj
    v2=qnv.Qnvec(n,N) # nD zero qnvector
    qnnum=qnn.Qnnum(1,0,num) # 1/num
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
    N=obj[0].N
    n=qnv.shape[0]
    len=qnn.Qnnum([1,0,len(obj)],N)  # 1/len(obj)
    tmp=qnv.Qnvec(n,N) # zero vector
    for thd in obj:
        tmp=tmp+thd
    tmp=tmp*len
    return tmp

def det_matrix(mtx: qnm.Qnmat) -> qnn.Qnnum:
    """Determinant of 3x3 matrix, mtx, in qnnumber
    
    Parameters
    ----------
    mtx: array
        3x3 matrix in qnnumer

    Returns
    -------
    determinant in qnnumber
    """
    N=mtx[0][0].N
    t3=qnn.Qnmtrx([0,0,1],N) # zero qnnumber
    t3=t3+mtx[0][0]*mtx[1][1]*mtx[2][2]
    t3=t3+mtx[0][1]*mtx[1][2]*mtx[2][0]    
    t3=t3+mtx[0][2]*mtx[1][0]*mtx[2][1]
    t3=t3-mtx[0][2]*mtx[1][1]*mtx[2][0]
    t3=t3-mtx[0][1]*mtx[1][0]*mtx[2][2]    
    t3=t3-mtx[0][0]*mtx[1][2]*mtx[2][1]

    return t3

def matrixtr(mtx: qnm.Qnmat):
    """ replace mtx with its transposed matrix"""
    n=mtx.n
    N=mtx.N
    mt=[qn0]*mtx.shape
    mtt=qnm.Qnmat(mt,n,N)
    for i in range(n):
        for j in range(n):
            mtt[j][i]=mtx[i][j]
    for i in range(n):
        for j in range(n):
            mtx[i][j]=mtt[i][j]
        

#if __name__ == '__main__':
    # test
