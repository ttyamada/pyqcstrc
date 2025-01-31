import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnmat.qnmat as qnm
import pyqcstrc.prjop.prjop as prj
#import pyqcstrc.qnmath.qnmath as qnm
#import pyqcstrc.qnclass.qnmath

class Qnsym_Octa:
    def __init__(self):
        # generators R8 and M
        n=6
        r8= np.zeros((n, n)); rm = np.zeros((n, n)) # 6x6 integer rotation matrix
        r8[0][1]=1; r8[1][2]=1; r8[2][3]=1; r8[3][0]=-1 # R8 
        rm[1][2]=1; rm[2][1]=1; rm[0][3]=1; rm[3][0]=1 # M
        nr=16
        r=np.zeros((nr, n, n)) # this should be array of int matrix
        set_r(r8,8,rm,2,r,nr) # set all integer symmetry operators r
        N=2 # for sqrt(2)
        qnmr=intr2qnmr(r,n,N,nr)  # integer rotation operators to qnmat rotation operator
        prj=prj.Qnprjop_Octa()
        prji=prj.Qnprjop_Octa()
        qnr=get_qnr(prj,prji,N,nr)
        qnmrq=np.array(nr,dtype=qnm.Qnmat)
        for i in range(nr):
            qnmrq[i]=qnmr[i]@prj
        self.qnmrq=qnmrq
        self.order=nr
    
# for decagonal QCs
class Qnsym_Deca:
    def __init__(self):
        # gemeratprs R10 and M
        n=6
        r10= np.zeros((n, n)); rm = np.zeros((n, n)) # 6x6 integer rotation matrix
        r10[0][3]=1; #R10
        r10[1][0]=1; r10[1][1]=1; r10[1][2]=1; r10[1][3]=1 # R10
        r10[2][3]=1; r10[3][1]=-1 #R10 
        rm[0][3]=1; rm[3][0]=1; rm[1][2]=1; rm[2][1]=1 # M
        nr=20
        r=np.zeros((nr, n, n))
        set_r(r10,10,rm,2,r,nr)
        N=5 # for sqrt(5)
        prj=prj.Qnprjop_Deca()
        prji=prj.Qnprjop_Deca()
        qnr=get_qnr(prj,prji,N,nr)
        self.qnr=qnr
        self.order=nr
    

## for dodecagonal QCs
class Qnsym_Dode:
    def __init__(self):
        # generators R8 and M
        n=6
        r12= np.zeros((n, n)); rm = np.zeros((n, n)) # 6x6 integer rotation matrix
        r12[0][1]=1; r12[1][2]=1; r12[2][3]=1; r12[3][0]=-1; r12[3][2]=1 # R12 
        rm[1][2]=1; rm[2][1]=1; rm[0][3]=1; rm[3][0]=1 # M
        nr=24
        r=np.zeros((nr, n, n))
        set_r(r12,12,rm,2,r,nr)
        N=3 # for sqrt(3)
        prj=prj.Qnprjop_Dode()
        prji=prj.Qnprjop_Dode()
        qnr=get_qnr(prj,prji,N,nr)
        self.qnr=qnr
        self.order=nr
    
def get_qnr(prj,prji,N,nr):
    qmt.qnmatinv(prji,6) # inverse matrix of prj
    qm0=qnm.zerom(6,N) # 6x6 qn zeromatrix
    qnr=[qm0]*nr # qnmatrix array
    matrixtr(prj)# transposed prj matrix
    matrixtr(prji) # transposed prji matrix 
    for i in range(nr):
        qnr[i]=prj@qnr@prji
    return qnr
    
def set_r(rg1,ng1,rg2,ng2,r,nr):
    r[0]=np.copy(rg1)
    for i in range(ng1-1):
        r[i+1]=get_r(rg1,r[i])
    for j in range(ng2):
        r[j+ng1]=get_r(rg2,r[j-1+ng1])
        
def get_r(r1,r2):
    r=np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                r[i][j]+=r1[i][k]*r2[k][j]
        
    return r

def intr2qnmr(r,n,N,nr):
    qnmr=np.array(nr,dtype=qnm.Qnmat)
    for i in range(nr):
        ri=qnm.intm2qnm(r[i],n,N) # qnmat for i-th rotation operator r[i]
        qnmr[i]=qnm.Qnmat(n,N)
        qnmr[i].mt=ri
        qnmr[i].n=n
        qnmr[i].N=N
    return qmr
    
if __name__ == '__main__':
    # test for qnnum projection operators
    
    a=Qnsym_Octa()
    nr=a.order
    print("nr",nr)
    print("Octa.a")
    for i in range(nr):
        qnm.printqnm("Qnsym_Octa",a.qnmrq[i])

