import numpy as np

import crsys
import qnnum as qnn
import qnvec as qnv
from qnndarray import (qnndarray_init,QnNdarray,printqndm,copy)

#if __name__ == '__main__':
# test
isys=4
crsys.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qnndarray_init()
n=crsys.n
N=crsys.N

#print("n=",n)
shape=(n,n)
qndm=QnNdarray(shape) # nxn qmnum zero matrix
print("qndm.ndim",qndm.ndim)
print("qndm.shape",qndm.shape)
printqndm("zero qnmat",qndm)

nr=10
print("n=",n)
shape=(nr,n,n)
qndm=QnNdarray(shape) # nxn qmnum zero matrix
print("qndm.ndim",qndm.ndim)
print("qndm.shape",qndm.shape)
for i in range(nr):
    printqndm("zero qnmat",qndm[i])
    
qndmi=copy(qndm) # copy of qnmi
print("qndmi.ndim",qndmi.ndim)
print("qndmi.shape",qndmi.shape)
printqndm("zero qnmat",qndmi)

qndma=QnNdarray((2,n,n))
print("qndma.ndim",qndma.ndim)
print("qndma.shape",qndma.shape)
printqndm("qndma",qndma)
    
