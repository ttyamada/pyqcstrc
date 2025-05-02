import numpy as np
import cython
import time

import crsys
import qnnum as qnn
from qnnum import (zero,\
                    one,\
                    any,\
                    qnnum_init,\
                    printqnn,\
                    Qnnum,copy) 


isys=4 # for octagonal
crsys.crsys_init(isys)
qnnum_init()

start_time = time.time()

qnn0=zero()
printqnn("qnn0",qnn0)
qnn1=one()
printqnn("qnn1",qnn1)
qnn2=copy(qnn1)
qnn3=qnn.Qnnum([2,0,1])
printqnn("qnn2",qnn2)
printqnn("qnn3",qnn3)
if(qnn1==qnn2):
    print("qnn1==qnn2")
print("qnn1==qnn2",qnn1==qnn2)
print("qnn0==qnn1",qnn0==qnn1)
print("qnn0>qnn1",qnn0>qnn1)
print("qnn0<qnn1",qnn0<qnn1)
qnn4=any([1,1,2])
printqnn("qnn4",qnn4)
print("qnn4>qnn1",qnn4>qnn1)
print("qnn4<qnn1",qnn4<qnn1)
print("qnn4==qnn1",qnn4==qnn1)
print()

qnn5=-qnn4
printqnn("-qnn4",qnn5)
printqnn("qnn1+qnn2",qnn1+qnn2)
printqnn("qnn1-qnn2",qnn1-qnn2)
printqnn("qnn1*qnn3",qnn1*qnn3)
printqnn("qnn1/qnn3",qnn1/qnn3)
qnn5=qnn.Qnnum([1,1,2])
printqnn("qnn5",qnn5)
printqnn("qnn1/qnn5",qnn1/qnn5)
printqnn("qnn5/qnn5",qnn5/qnn5)
printqnn("qnn5*2",qnn5*2)
printqnn("qnn5/2",qnn5/2)

for i in range(10000):
    qnt1 = qnn1+qnn2
    qnt2 = qnn1-qnn2
    qnt3 = qnn1*qnn3
    qnt4 = qnn1/qnn3


end_time = time.time()
elapsed_time = end_time - start_time

print(f"elapsed time: {elapsed_time} [sec]")
print(f"elapsed time: {(elapsed_time)*1000} [ms]")
print(f"elapsed time: {(elapsed_time)*1000000} [µs]")



