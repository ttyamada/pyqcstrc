#test qnmath operations
import sys
import numpy as np
import qnnum as qnn
import qnvec as qnv
import crsys

def chkop(qnv1,qnv2):
    qnv.printqnv("qnv1+qnv2",qnv1+qnv2)
    print("qnv1+qnv2",qnv.qnv2flt(qnv1+qnv2))
    print("qnv1+qnv2",qnv.qnv2flt(qnv1)+qnv.qnv2flt(qnv2))
    qnv.printqnv("qnv1-qnv2",qnv2-qnv1)
    print("qnv1-qnv2",qnv.qnv2flt(qnv1-qnv2))
    print("qnv1-qnv2",qnv.qnv2flt(qnv1)-qnv.qnv2flt(qnv2))
    a=np.array([3,1,5])
    print("a",a)
    qna=qnv.intv2qnv(a,qnv1.vt[0].N)
    qnv.printqnv("qna",qna)

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
crsys.Crsys(4)
qn1=qnn.Qnnum(np1) # sqrt(2) type qnnumber
qn2=qnn.Qnnum(np2)
qn3=qnn.Qnnum(np3)
qn4=qnn.Qnnum(np4)
qn5=qnn.Qnnum(np0)
qn6=qnn.Qnnum(np0)

print("qn1",qnn.qn2npa(qn1))
print("qn2",qnn.qn2npa(qn2))
print("qn3",qnn.qn2npa(qn3))
print("qn4",qnn.qn2npa(qn4))
print("qn5",qnn.qn2npa(qn5))
print("qn6",qnn.qn2npa(qn6))

v1=np.array([qn1,qn2,qn3,qn4,qn5,qn6],qnn.Qnnum) #Qnnum array
v2=np.array([qn2,qn1,qn4,qn3,qn5,qn6],qnn.Qnnum) #Qnnum array
qnv1=qnv.Qnvec(v1) # qnvector for vec1
qnv2=qnv.Qnvec(v2) # qnvector for vec2

print(type(qnv1))
print(type(qnv2))

qnv.printqnv("qnv1",qnv1)
qnv.printqnv("qnv2",qnv2)
chkop(qnv1,qnv2)
