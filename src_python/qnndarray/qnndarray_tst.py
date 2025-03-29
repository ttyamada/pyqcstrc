import numpy as np

import crsys
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
from qnndarray import (qnndarray_init,QnNdarray,printqndm,copy,zeros)

#if __name__ == '__main__':
# test
isys=4
crsys.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qnndarray_init()
qnm.qnmat_init()

n=crsys.n
N=crsys.N

shape=(n,n)
qndm=qnm.zerom(shape) # nxn qmnum zero matrix
print("qndm.shape",qndm.shape)
printqndm("zero qnmat",qndm)
unitm=qnm.unitm(n)
print("unitm.shape",unitm.shape)
printqndm("unit qnmat",unitm)

unitmi=copy(unitm) # copy of qnmi
print("unitmi.shape",unitmi.shape)
printqndm("unitmi",unitmi)

nr=10
print("nr",nr,"n=",n)
shape=(nr,n,n)
print("shape",shape)
qnda=zeros(shape) # nr nxn qmnum zero matrices
print("qnda.shape",qnda.shape)
for i in range(nr):
    str="qnmat "+format(i+1)
    if i==0:
        qnda[i]+=unitm
    else:
        qnda[i]=qnda[i-1]+unitm
    printqndm(str,qnda[i])

    
