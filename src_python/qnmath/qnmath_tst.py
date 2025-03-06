
import cython

import crsys
import qnarray as qna
from qnmath import (qnmath_init,qsort,qsort_f)

#if __name__ == '__main__':
isys=4 # octagonal
crsys=crsys_init(isys)
qnmath_init()

nr=10
N=crsys.N # for octagonal
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
    
qn=qna.QnNdarray(shape,N)
ip=[0]*nr
for i in range(nr):
    #qn[i]=qnn.int2qnn(nr-1-i,N)
    qn[i]=qnn.Qnnum([nr-1-i,1,2],N)
    qnn.printqnn("qn[i]",qn[i])
print()
qsort(qn,ip,nr)
for i in range(nr):
    qnn.printqnn("qn[i]",qn[i])
    
        
