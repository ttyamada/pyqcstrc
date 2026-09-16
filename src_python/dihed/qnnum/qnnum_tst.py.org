import crsys as crs
import numpy as np
import cython
import time

#from extended_int import int_inf, ExtendedIntegral
import crsys
from numpy.typing import NDArray
from typing import Self

import crsys as crs
import qnnum as qnn

isys = 4; # for octagonal
crs.crsys_init(isys)
qnn.qnnum_init(); # qnnum init
N = crs.N
print("N ",N)

start_time = time.time()

qnn0 = qnn.zero()
qnn.printqnn("qnn0", qnn0)
qnn1 = qnn.one()
qnn.printqnn("qnn1", qnn1)
qnn2 = qnn1
d1 = [2, 0, 1]
qnn3=qnn.Qnnum(d1)
qnn.printqnn("qnn2", qnn2)
qnn.printqnn("qnn3", qnn3)

#
if qnn1 == qnn2:
    print("qnn1==qnn2",qnn1==qnn2)

print("qnn1==qnn2: ",qnn1 == qnn2)
print("qnn0==qnn1: ",qnn0 == qnn1)
print("qnn0>qnn1: ",qnn0 > qnn1)
print("qnn0<qnn1: ", qnn0 < qnn1)
d2 = [1, 1, 2]
qnn4 = qnn.any(d2)
qnn.printqnn("qnn4", qnn4)
print("qnn4>qnn1: ",qnn4 > qnn1)
print("qnn4<qnn1: ",qnn4 < qnn1)
print("qnn4==qnn1: ",qnn4 == qnn1)
print()

qnn5 = qnn.copy(-qnn4)
qnn6 = qnn.copy(qnn1 + qnn2)
qnn7 = qnn.copy(qnn1 - qnn2)
qnn8 = qnn.copy(qnn1 * qnn3)
qnn9 = qnn.copy(qnn1 / qnn3)
qnn.printqnn("-qnn4", qnn5)
qnn.printqnn("qnn1+qnn2", qnn6)
qnn.printqnn("qnn1-qnn2", qnn7)
qnn.printqnn("qnn1*qnn3", qnn8)
qnn.printqnn("qnn1/qnn3", qnn9)
d3 = [1, 1, 2]
qnn5 = qnn.any(d3)
qnn6 = qnn1 / qnn5
qnn7 = qnn5 / qnn5
qnn8 = qnn5 * 2
qnn9 = qnn5 / 2
qnn.printqnn("qnn5", qnn5)
qnn.printqnn("qnn1/qnn5",qnn6)
qnn.printqnn("qnn5/qnn5",qnn7)
qnn.printqnn("qnn5*2", qnn8)
qnn.printqnn("qnn5/2", qnn9)

qnt1=qnn.zero(); qnt2=qnn.zero(); qnt3=qnn.zero(); qnt4=qnn.zero()
for i in range(100000):
    qnt1 = qnn.copy(qnn1+qnn2)
    qnt2 = qnn.copy(qnn1-qnn2)
    qnt3 = qnn.copy(qnn1*qnn3)
    qnt4 = qnn.copy(qnn1/qnn3)

elapsed_time = time.time() - start_time

print("elapsed time = ",elapsed_time*1000," [ms]")

