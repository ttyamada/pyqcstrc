import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnmat.qnmat as qnm
#import pyqcstrc.qnclass.qnmath

class Qnsym_Octa:
    # generators R8 and M
    r8=[0]*(6,6); rm=[0]*(6,6) # 6x6 rotation matrix
    r8[0][1]=1; r8[1][2]=1; r8[2][3]=1; r8[3][0]=-1 # R8 
    rm[1][2]=1; rm[2][1]=1; rm[0][3]=1; rm[3][0]=1 # M
    r=[0]*(16,6,6)
    set_r(r8,rm,8,2,r,16)
    N=2 # for sqrt(2)
    intr2qnr(r,qnr,16,N)
# for decagonal QCs
class Qnsym_Deca:
    # gemeratprs R10 and M
    r10=[0]*(6,6); rm=[0]*(6,6) # 6x6 rotation matrix
    r10[0][3]=1; #R10
    r10[1][0]=1; r10[1][1]=1; r10[1][2]=1; r10[1][3]=1 # R10
    r10[2][3]=1; r10[3][1]=-1 #R10 
    rm[0][3]=1; rm[3][0]=1; rm[1][2]=1; rm[2][1]=1 # M
    r=[0]*(20,6,6)
    set_r(r10,rm,10,2,r,20)
    N=5 # for sqrt(5)
    qnr=intr2qnr(r,20,N)

## for dodecagonal QCs
class Qnsym_Dode:
    # generators R8 and M
    r12=[0]*(6,6); rm=[0]*(6,6) # 6x6 rotation matrix
    r12[0][1]=1; r12[1][2]=1; r12[2][3]=1; r12[3][0]=-1; r12[3][2]=1 # R12 
    rm[1][2]=1; rm[2][1]=1; rm[0][3]=1; rm[3][0]=1 # M
    r=[0]*(24,6,6)
    set_r(r12,rm,12,2,r,24)
    N=3 # for sqrt(3)
    qnr=intr2qnr(r,24,N)

def set_r(rg1,ng1,rg2,ng2,r,nr):
    r[0]=rg1
    for i in range(ng1-1):
        r[i+1]=get_r(rg1,r[i])
    for j in range(ng2):
        r[j+ng1]=get(rg2,r[j-1+ng1])
        
def get_r(r1,r2):
    r=[0]*(6,6)
    for i in range(6):
        for j in range(6):
            for k in range(6):
                r[i][j]+=r1[i][k]*r2[k][j]
        
    return

def intr2qnr(r,nr,N):
    qm0=qnm.Qnmat(r,6,N)
    qmr=[qm0]*nr
    for i in range(nr):
        for j in range(6):
            for k in range(6):
                qmr[i].mt[j][k]=r[i][j][k]
    return qmr
    
if __name__ == '__main__':
    # test for qnnum projection operators
    
    a=Qnsym_Octa()
    la=a.order
    print("la",la)
    print("Octa.a")
    for i in range(la):
        qnm.printqnm("Qnsym_Octa",a.smo[i])

