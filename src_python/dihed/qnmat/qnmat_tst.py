import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
#from qnmat import (qnmat_init,zerom,unitm,copy,printqnm)

#if __name__ == '__main__':
# test
def qnmat_tst(str: str, isys:np.int64):
    print(str)
    crs.crsys_init(isys)
    qnn.qnnum_init()
    qnv.qnvec_init()
    qnm.qnmat_init()

    M0=qnn.Qnnum([ 0, 0, 1]) #  0
    M1=qnn.Qnnum([ 1, 0, 1]) #  1
    M2=qnn.Qnnum([-1, 0, 1]) # -1
    M3=qnn.Qnnum([ 0, 1, 2]) #  sqrt(2)/2 t1
    M4=qnn.Qnnum([ 0,-1, 2]) # -sqrt(2)/2 t2=-t1
    
    # define Qnnumber matrix prj0
    prj0=np.array([\
        #prj0=qnm.Qnmat([\
       [M1,M0,M1,M0,M0],\
       [M3,M3,M4,M3,M0],\
       [M0,M1,M0,M2,M0],\
       [M4,M3,M3,M3,M0],\
       [M0,M0,M0,M0,M1]\
    ],dtype=qnn.Qnnum)

    qnm.printqnm("prj0",prj0)
 
    n=crs.n
    N=crs.N
    print("n",n,"N",N)
    qnm0=qnm.zerom(n,n) # nxn qmnum zero matrix
    print("qnm.shape",qnm.shape)
    qnm.printqnm("qnm0",qnm0)

    qnm1=qnm.copy(qnm0)
    print("qnm1.shape",qnm1.shape)
    qnm.printqnm("qnm1",qnm1)

    unm1=qnm.unitm(n) # nxn qmnum zero matrix
    print("unm1.shape",unm1.shape)
    qnm.printqnm("unm1",unm1)

    unm2=qnm.unitm(n) # nxn qmnum zero matrix
    print("unm2.shape",unm2.shape)
    qnm.printqnm("unm2",unm2)

    unm3=unm2@unm1
    qnm.printqnm("unm2@unm1",unm3)
    
    M0=qnn.Qnnum([0,0,1])
    M1=qnn.Qnnum([1,0,1])
    M2=qnn.Qnnum([0,1,1])
    M3=qnn.Qnnum([1,1,2])
    M4=qnn.Qnnum([1,-1,2])
    if n==5:
        qnv1=qnv.anyv(np.array([M0,M1,M2,M3,M4]))
    elif n==6:
        qnv1=qnv.anyv(np.array([M0,M1,M2,M3,M4,M0]))
    qnv.printqnv("qnv1",qnv1)
    qnv2=unm2@qnv1
    qnv.printqnv("qnv2",qnv2)

isys=3
qnmat_tst("decagonal",isys)

isys=2
qnmat_tst("icosahedral",isys)



