import numpy as np
import cython

import crsys
from qnnum import (zero,one,any,qnnum_init,printqnn,Qnnum,copy) 

#if __name__ == '__main__':
# test
isys=4 # for octagonal
crsys.crsys_init(isys)
qnnum_init()

qnn0=zero()
printqnn("qnn0",qnn0)
qnn1=one()
printqnn("qnn1",qnn1)
qnn2=copy(qnn1)
printqnn("qnn2",qnn2)
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

qnn5=-qnn4
printqnn("-qnn4",qnn5)

