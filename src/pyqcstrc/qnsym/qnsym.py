import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnmat.qnmat as qnm
import pyqcstrc.prjop.prjop as prj
import pyqcstrc.qnmath.qnmath as qmt
import pyqcstrc.qnndarray.qnndarray as qna
#import pyqcstrc.qnclass.qnmath

class Qnsym_Octa(qna.QnNdarray):
    def __new__(cls):
        nr=16
        n=5
        N=2
        shape=(nr,n,n)
        #print("shape in __new__",shape) # for test
        return super().__new__(cls,shape,N)
        
    def __init__(self):
        # generators R8 and M
        nr=16
        n=5
        N=2
        # two generating elements
        r8= np.zeros((n, n),dtype=np.int64)  # nxn integer rotation matrix
        rm = np.zeros((n, n),dtype=np.int64) # nxn integer rotation matrix
        r8[0][1]=1; r8[1][2]=1; r8[2][3]=1; r8[3][0]=-1 # R8 
        rm[1][2]=1; rm[2][1]=1; rm[0][3]=1; rm[3][0]=1 # M
        shape=(nr,n,n) # for nr nxn rotation matrices
        #print("shape in __init__",shape) # for test
        r=np.zeros(shape,dtype=np.int64) # nD int array
        #print("r.shape",r.shape) # fpr test
        set_r(r8,8,rm,2,r) # set all integer symmetry operators r
        print_r(r)  #; exit() # for test
        
        prj0=prj.Qnprj_Octa()
        prji=prj.Qnprj_Octa()
        qna.printqndm("prji",prji) # for test
        
        qmt.qnmatinv(prji,n) # get inversion matrix
        
        qnr=get_qnr(prj0,prji,self)  # block diagonakl symmetry operator for ext and int comp.
        #self.qnmrq=qnmrq
        #self.n=n
        #self.N=N
        #self.ord=ord
    
# for decagonal QCs
class Qnsym_Deca(qna.QnNdarray):
    def __new__(cls):
        nr=20
        n=5
        N=5
        shape=(nr,n,n)
        return super().__new__(cls,shape,N)
        
    def __init__(self):
        # gemeratprs R10 and M
        nr=20
        n=5
        N=5
        # two generating elements
        r10= np.zeros((n, n),dtype=np.int64)
        rm = np.zeros((n, n),dtype=np.int64) # nxn integer rotation matrix
        r10[0][3]=1; #R10
        r10[1][0]=1; r10[1][1]=1; r10[1][2]=1; r10[1][3]=1 # R10
        r10[2][3]=1; r10[3][1]=-1 #R10 
        rm[0][3]=1; rm[3][0]=1; rm[1][2]=1; rm[2][1]=1 # M
        #r=np.zeros((nr, n, n))
        shape=(nr,n,n) # for nr nxn rotation matrices
        print("shape",shape) # for test
        r=np.array(shape,dtype=np.int63)
        set_r(r10,10,rm,2,r)  # set all integer rotation matrices
        qnmr=intr2qnmr(r,n,N,nr)  # integer rotation operators to qnmat rotation operator
        prj0=prj.Qnprj_Deca()
        prji=prj.Qnprj_Deca()
        qmt.qnmatinv(prji,n)
        get_qnr(prj0,prji,self)  # block diagonakl symmetry operator for ext and int comp.
        #self=qnr
        #self.order=nr
    

## for dodecagonal QCs
class Qnsym_Dode(qna.QnNdarray):
    def __new__(cls):
        nr=24
        n=5
        N=3
        shape=(nr,n,n)
        return super().__new__(cls,shape,N)
    
    def __init__(self):
        # generators R8 and M
        nr=24
        n=5
        N=3
        # two generating elements
        r12= np.zeros((n, n),dtype=np.int64)
        rm = np.zeros((n, n),dtype=np.int64) # 6x6 integer rotation matrix
        r12[0][1]=1; r12[1][2]=1; r12[2][3]=1; r12[3][0]=-1; r12[3][2]=1 # R12 
        rm[1][2]=1; rm[2][1]=1; rm[0][3]=1; rm[3][0]=1 # M
        shape=(nr,n,n) # for nr nxn rotation matrices
        r=np.array(shape,dtype=np.int64)
        set_r(r12,12,rm,2,r)  # set all integer rotation matrices
        N=3 # for sqrt(3)
        qnmr=intr2qnmr(r,n,N,nr)  # integer rotation operators to qnmat rotation operator
        prj0=prj.Qnprj_Dode()
        prji=prj.Qnprj_Dode()
        qmt.qnmatinv(prji,n)
        qnr=get_qnr(prj0,prji,n,N,nr) # block diagonakl symmetry operator for ext and int comp,.
        self=qnr
        #self.order=nr
    
def get_qnr(prj,prji,nr,n,N):
    qmt.qnmatinv(prji,n) # inverse matrix of prj
    qm0=qnm.Qnmat(n,N) # nxn qn zeromatrix
    qnr=[qm0]*nr # qnmatrix array
    qmt.matrixtr(prj)  # transposed prj matrix
    qmt.matrixtr(prji) # transposed prji matrix 
    for i in range(nr):
        qnr[i]=qnm.copy(prj@qnr@prji)
    return qnr
    
# gemerate all rotation matrices from
# only for two generating elements
def set_r(rg1,ng1,rg2,ng2,r):
    #print("r.shape",r.shape)  # for test
    nr=r.shape[0]
    n=r.shape[1]
    r[0]=np.identity(n,dtype=np.int64) # this should be a unit matrix
    for i in range(ng1-1):
        r[i+1]=get_r(rg1,r[i],n)
    for j in range(ng1):
        r[j+ng1]=get_r(rg2,r[j-1+ng1],n)
        
# matrix multiple
# this can be replaced by r1*r2
# when r1 and r2 are qnmatrices
def get_r(r1,r2,n):
    r=np.zeros((n, n),dtype=np.int64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                r[i][j]+=r1[i][k]*r2[k][j]
        
    return r

def print_r(r):
    shape=r.shape
    ndim=r.ndim
    #print("r.shape",r.shape)
    #print("r.ndim",ndim)
    nr=shape[0]
    n=shape[1]
    for i in range(nr):
        for j in range(n):
            print("[ ",end=" ")
            for k in range(n):
                print(r[i][j][k],end=" ")
            print("]")
        print(" ")


# integer matrix to qnnumber matrix transformation
def intr2qnmr(r,n,N,nr):
    qnmr=qna.QnNdarray((nr,n,n),N) #[qm0]*nr
    for i in range(nr):
        qnmr[i]=qnm.intm2qnm(r[i],n,N) # qnmat for i-th rotation operator r[i]
    return qnmr

def test_wt(str:str,nr,a:qna.QnNdarray):
    print("nr",nr)
    print(str)
    for i in range(nr):
        qnm.printqnm("Qnsym_",str,a.qnr[i])
        
# for test
if __name__ == '__main__':
    # test for qnnum projection operators
    isys=4
    n=5
    N=2
    a=Qnsym_Octa() # octagonal
    n=a.n
    print("a.n",n)
    qnm.printqnm("Octa a",a,n)
    #for i in range(nr):
    #    qnm.printqnm("Qnsym_Octa",a.qnr[i])
    
    isys=3
    n=5
    N=5
    a=Qnsym_Deca() # decagonal
    nr=a.order
    test_wt("Deca",nr,a)

    isys=5
    n=5
    N=3
    a=Qnsym_Dode() # dodecagonal
    nr=a.order
    test_wt("Dode",nr,a)

        