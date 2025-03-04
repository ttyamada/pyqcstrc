import numpy as np
from qnnum import (printqnn,Qnnum,copy) 

#if __name__ == '__main__':
# test
N=np.int64(2)
qnn0=Qnnum([0,0,1],N)
printqnn("qnn0",qnn0)
qnn1=Qnnum([1,0,1],N)
printqnn("qnn1",qnn1)
qnn2=copy(qnn1)
printqnn("qnn2",qnn2)
if(qnn1==qnn2):
    print("qnn1==qnn2")
print("qnn1==qnn2",qnn1==qnn2)
print("qnn0==qnn1",qnn0==qnn1)
print("qnn0>qnn1",qnn0>qnn1)
print("qnn0<qnn1",qnn0<qnn1)
qnn4=Qnnum([1,1,2],N)
printqnn("qnn4",qnn4)
print("qnn4>qnn1",qnn4>qnn1)
print("qnn4<qnn1",qnn4<qnn1)
print("qnn4==qnn1",qnn4==qnn1)

qnn5=-qnn4
printqnn("-qnn4",qnn5)

