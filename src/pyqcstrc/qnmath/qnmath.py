import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnmat.qnmat as qnm
#from goto import with_goto

def abs(a:qnn.Qnnum):
    qnzero=qnn.Qnnum([0,0,1],a.N)
    if(a<qnn.qnzero):
        return -a
    if(a>=qnn.qnzero):
        return a

def qnmatinv(a:qnm.Qnmat,n:np.int64):
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
    qn0=qnn.Qnnum([0,0,1],a.N)
    qn1=qnn.Qnnum([1,0,1],a.N)
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
def qsort(x:qnv.Qnvec,ip:np.array,nx: np.int64):
    #     quick sort (ascending order of x)
    #     nx: the number of data x
    #     ip: the initial order
    #     st: a work array
    
    def setlrs():
        #label .l1
        l=st[s][0] 
        r=st[s][1] 
        s=s-1 
#2   continue
    def setijxt():
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

def centroid(obj: qnv.Qnvec) -> qnv.Qnvec:
    """geometric center, centroid of tetrahedron, triangle or edge, in SQRT2-style.

    Parameters
    ----------
    obj: array
        6-dimensional vector in SQRT2-style
    
    Returns
    -------
    centroid: array in SQRT2-style
    """
    N=obj[0].N
    num=len(obj) # length of obj
    v2=qnv.Qnvec(6,N) # 6D zero qnvector
    qnnum=qnn.Qnnum(1,0,num) # 1/num
    for i1 in range(num):
        v2=v2+obj[i1]
    v0=v2*qnnum
    return v0

# needless???
def centroid_obj(obj: qnv.Qnvec) -> qnv.Qnvec:
    """geometric center, centroid of tetrahedron, in TAU-style.

    Parameters
    ----------
    tetrahedron: array
        6-dimensional vector in TAU-style
    
    Returns
    -------
    centroid: array in TAU-style
    """
    #print('centroid_obj')
    
    #  geometric center, centroid of OBJ
    N=obj[0].N
    len=qnn.Qnnum([1,0,len(obj)],N)  # 1/len(obj)
    tmp=qnv.Qnvec(6,N) # zero vector
    for thd in obj:
        tmp=tmp+thd
    tmp=tmp*len
    return tmp

def coplanar_check(p: NDArray[np.int64],num_iteration: int=5) -> bool:
    """Check whether a given set of points (in TAU-style) is coplanar or not.
    
    メモ：xyz1とxyz2の選び方次第で、outer_product(v1,v2)が小さくなりcoplanarと間違って判定する場合がある。
    これを避けるために適切なxyz1とxyz2の選び方が必要。以下では、ランダムにxyz1とxyz2の選ぶ。
    
    Parameters
    ----------
    p: array
        a set of pointsin TAU-style.

    Returns
    -------
    int
    #bool
    """
    
    """
    num=len(p)
    if num>3:
        flag=0
        lst0=[i for i in range(num)]
        for _ in range(num_iteration):
            lst3=random.sample(lst0, 3)
            xyz0i=projection3(p[lst3[0]])
            xyz1i=projection3(p[lst3[1]])
            xyz2i=projection3(p[lst3[2]])
            v1=sub_vectors(xyz1i,xyz0i)
            v2=sub_vectors(xyz2i,xyz0i)
            v3=outer_product(v1,v2)
            flag=0
            if np.all(d[:2])==0):
                pass
            else:
                flag=1
                break
        if flag==1:
            counter=0
            lst=list(filter(lambda x: x not in lst3, lst0))
            for i in lst:
                xyz3i=projection3(p[i])
                v4=sub_vectors(xyz3i,xyz0i)
                d=inner_product(v3,v4)
                if np.all(d[:2])==0:
                    pass
                else:
                    counter=1
                    break
            if counter==0:
                return True # coplanar
            else:
                return False
        else:
            'error in coplanar_check_numeric. increase num_iteration.'
            return 
    else:
        return True # coplanar
    """
    return coplanar_check_numeric_tau(p,num_iteration)

#def matrixpow(ma: NDArray[np.int64], n: int) -> NDArray[np.int64]:
#    """
#    """
#    (mx,my)=ma.shape
#    if mx==my:
#        if n==0:
#            return np.identity(mx)
#        elif n<0:
#            tmp=np.identity(mx)
#            inva = np.linalg.inv(ma)
#            for i in range(-n):
#                #tmp=np.dot(tmp,inva)
#                tmp=tmp@inva
#            return tmp
#        else:
#            tmp=np.identity(mx)
#            for i in range(n):
#                #tmp=np.dot(tmp,ma)
#                tmp=tmp@ma
#            return tmp
#    else:
#        print('matrix has not regular shape')
#        return 

def det_matrix(mtx: qnm.Qnmat) -> qnn.Qnnum:
    """Determinant of 3x3 matrix, mtx, in SQRT2 style
    
    Parameters
    ----------
    mtx: array
        3x3 matrix in SQRT2-style

    Returns
    -------
    6d vectors projected onto Eperp in SQRT2-style.
    """
    
    t3=   mtx.m[0][0]*mtx.m[1][1]*mtx.m[2][2]
    t3=t3+mtx.m[0][1]*mtx.m[1][2]*mtx.m[2][0]    
    t3=t3+mtx.m[0][2]*mtx.m[1][0]*mtx.m[2][1]
    t3=t3-mtx.m[0][2]*mtx.m[1][1]*mtx.m[2][0]
    t3=t3-mtx.m[0][1]*mtx.m[1][0]*mtx.m[2][2]    
    t3=t3-mtx.m[0][0]*mtx.m[1][2]*mtx.m[2][1]

    return t3

def matrixtr(mtx: qnm.Qnmat) -> qnn.Qnnum:
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
