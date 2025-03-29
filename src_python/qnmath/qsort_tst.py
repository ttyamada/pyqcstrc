import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnndarray as qna
from qsort import (qsort_init,qsort,qsort_f)

isys=4 # octagonal
crsys=crs.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qsort_init()

nr=10
N=crs.N # for octagonal
shape=(nr)

fn=[0.0]*nr
for i in range(nr):
    print(fn[i])
    
ip=[0]*nr
for i in range(nr):
    #qn[i]=qnn.int2qnn(nr-1-i,N)
    fn[i]=(float)(nr-1-i)
    print("fn[i]",fn[i])
    #print()
qsort_f(fn,ip,nr)
for i in range(nr):
    print(fn[i])
    
#qn=qna.QnNdarray(shape) # qn should be Qnvec
qn=qnv.Qnvec(nr)
ip=[0]*nr
print("nr",nr)  # for test
for i in range(nr):
    #qn[i]=qnn.int2qnn(nr-1-i,N)
    qn[i]=qnn.Qnnum([nr-1-i,1,2])
    qnn.printqnn("qn[i]",qn[i])
print()
print("nr",nr)  # for test
qsort(qn,ip,nr)
for i in range(nr):
    qnn.printqnn("qn[i]",qn[i])
    
        
