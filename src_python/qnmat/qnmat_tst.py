import numpy as np
import cython

import crsys
import qnnum as qnn
import qnvec as qnv
from qnmat import (qnmat_init,zerom,unitm,copy,printqnm)

#if __name__ == '__main__':
# test
def qnmat_tst(str: str, isys:np.int64):
    print(str)
    crsys.crsys_init(isys)
    qnn.qnnum_init()
    qnv.qnvec_init()
    qnmat_init()
    
    n=crsys.n
    N=crsys.N
    print("n",n,"N",N)
    qnm=zerom((n,n)) # nxn qmnum zero matrix
    print("qnm.ndim",qnm.ndim)
    print("qnm.shape",qnm.shape)
    printqnm("qnm",qnm)

    qnm1=copy(qnm)
    print("qnm1.ndim",qnm1.ndim)
    print("qnm1.shape",qnm1.shape)
    printqnm("qnm1",qnm1)

    unm1=unitm(n) # nxn qmnum zero matrix
    print("unm1.ndim",unm1.ndim)
    print("unm1.shape",unm1.shape)
    printqnm("unm1",unm1)

    unm2=unitm(n) # nxn qmnum zero matrix
    print("unm2.ndim",unm2.ndim)
    print("unm2.shape",unm2.shape)
    printqnm("unm2",unm2)

    unm3=unm2@unm1
    printqnm("unm2@unm1",unm3)
    
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



