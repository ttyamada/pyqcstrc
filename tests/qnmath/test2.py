#test qnmath operations
import sys
import numpy as np
from pyqcstrc.qnnum import qnnum
from pyqcstrc.qnvec import qnvec

def chkop(qnv1,qnv2):
    qnvec.printqnv("qnv1+qnv2",qnv1+qnv2)
    print("qnv1+qnv2",qnvec.qnv2flt(qnv1+qnv2))
    #qnvec.printqnv("qnv1+qnv2",qnvec.qnv2flt(qnv1)+qnvec.qnv2flt(qnv2))
    qnvec.printqnv("qnv1-qnv2",qnv2-qnv1)
    print("qnv1-qnv2",qnvec.qnv2flt(qnv1-qnv2))
    #qnvec.printqnv("qnv1-qnv2",qnvec.qnv2flt(qnv1)-qnvec.qnv2flt(qnv2))

def printqnv(str,qnv1):
    print(str,"[",end=" ")
    for i in qnv1:
        print(i,end=" ")
    print("]")

def printqnv2(str,qnv1,qnv2):
    print(str,"[",end=" ")
    for i in qnv1:
        print(i,end=" ")
    print("] [",end="")
    for i in qnv2:
        print(i,end="]")

np0=np.array([0,0,1])
np1=np.array([1,2,3])
np2=np.array([4,5,6])
np3=np.array([1,0,1])
np4=np.array([0,1,1])

print("np0",np0)
print("np1",np1)
print("np2",np2)
print("np3",np3)
print("np4",np4)

# octagonal qnnumber
print("octagonal")
qn1=qnnum.Qnnum(np1,2) # sqrt(2) type qnnumber
qn2=qnnum.Qnnum(np2,2)
qn3=qnnum.Qnnum(np3,2)
qn4=qnnum.Qnnum(np4,2)
qn5=qnnum.Qnnum(np0,2)
qn6=qnnum.Qnnum(np0,2)

print("qn1",qnnum.qn2npa(qn1))
print("qn2",qnnum.qn2npa(qn2))
print("qn3",qnnum.qn2npa(qn3))
print("qn4",qnnum.qn2npa(qn4))
print("qn5",qnnum.qn2npa(qn5))
print("qn6",qnnum.qn2npa(qn6))

v1=np.array([qn1,qn2,qn3,qn4,qn5,qn6],qnnum.Qnnum) #Qnnum array
v2=np.array([qn2,qn1,qn4,qn3,qn5,qn6],qnnum.Qnnum) #Qnnum array
qnv1=qnvec.Qnvec(v1) # qnvector for vec1
qnv2=qnvec.Qnvec(v2) # qnvector for vec2

print(type(qnv1))
print(type(qnv2))

qnvec.printqnv("qnv1",qnv1)
qnvec.printqnv("qnv2",qnv2)
chkop(qnv1,qnv2)
