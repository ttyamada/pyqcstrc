
from qnndarray import (QnNdarray,printqndm)
#if __name__ == '__main__':
# test

N=2
n=5
print("n=",n)
shape=(n,n)
qndm=QnNdarray(shape,N) # nxn qmnum zero matrix
print("qndm.ndim",qndm.ndim)
print("qndm.shape",qndm.shape)
printqndm("zero qnmat",qndm)

N=2
n=5
nr=10
print("n=",n)
shape=(nr,n,n)
qndm=QnNdarray(shape,N) # nxn qmnum zero matrix
print("qndm.ndim",qndm.ndim)
print("qndm.shape",qndm.shape)
for i in range(nr):
    printqndm("zero qnmat",qndm[i])
    
qndmi=copy(qndm) # copy of qnmi
print("qndmi.ndim",qndmi.ndim)
print("qndmi.shape",qndmi.shape)
printqndm("zero qnmat",qndmi)

qndma=QnNdarray((2,n,n),N)
print("qndma.ndim",qndma.ndim)
print("qndma.shape",qndma.shape)
printqndm("qndma",qndma)
    
