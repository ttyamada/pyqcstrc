import qnnum as qnn
import numpy as np

import crsys
from qnvec import (qnvec_init,anyv,printqnv,dot,cros,printqnvs,Qnvec)

#if __name__ == '__main__':
# test
def qnvec_tst(str,isys):
    crsys.crsys_init(isys)
    qnn.qnnum_init()
    qnvec_init()
    
    n=crsys.n
    N=crsys.N
    print(str)
    print("n",n,"N",N)
    M0=qnn.Qnnum([0,0,1])
    M1=qnn.Qnnum([1,0,1])
    M2=qnn.Qnnum([0,1,1])
    if n==5:
        vec1=np.array([M0,M1,M2,M0,M1],dtype=qnn.Qnnum)    
        vec2=np.array([M0,M1,M2,M0,M1],dtype=qnn.Qnnum)
        vec3=np.array([M1,M2,M0,M1,M2],dtype=qnn.Qnnum)
    elif n==6:
        vec1=np.array([M0,M1,M2,M0,M1,M2],dtype=qnn.Qnnum)    
        vec2=np.array([M0,M1,M2,M0,M1,M2],dtype=qnn.Qnnum)
        vec3=np.array([M1,M2,M0,M1,M2,M0],dtype=qnn.Qnnum)
    qnv1=anyv(vec1)
    qnv2=anyv(vec2)
    qnv3=anyv(vec3)

    printqnv("qnv1",qnv1)
    printqnv("qnv2",qnv2)
    printqnv("qnv3",qnv3)

    qnv4=qnv1+qnv2
    qnv5=qnv1-qnv3
    printqnv("qnv1+qnv2",qnv4)
    printqnv("qnv1-qnv3",qnv5)

    qnn1=dot(qnv1,qnv3)
    qnn.printqnn("dot(qnv1,qnv3)",qnn1)

    qnv6=cros(qnv1,qnv3)
    printqnv("cross(qnv1,qnv3)",qnv6)
    
    print("qnv1==qnv2",qnv1==qnv2)
    print("qnv1==qnv3",qnv1==qnv3)
    
    qnvs=np.zeros(1,dtype=Qnvec)
    qnvt=np.zeros(1,dtype=Qnvec)
    qnvs[0]=qnv1          # this is OK
    
    qnvt[0]=qnv2  # this is necessary
    qnvs=np.append(qnvs,qnvt)

    qnvt[0]=qnv3 # this is necessary
    qnvs=np.append(qnvs,qnvt)

    print("qnvs.shape",qnvs.shape)
    printqnvs("qnvs",qnvs)
    
isys=4
qnvec_tst("octagonal",isys) # octagonal
isys=3
qnvec_tst("decagonal",isys) # octabonal
isys=5
qnvec_tst("dodecagonal",isys) # octabonal
isys=2
qnvec_tst("icosahedral",isys) # icosahedral
    
    
    
