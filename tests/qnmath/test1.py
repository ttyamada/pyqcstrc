#test qnmath operations
import sys
import numpy as np
#from pyqcstrc.qnnum import qnnum
import pyqcstrc.qnnum.qnnum as qnn

def chkop(qn1,qn2):
    print("qn1+qn2",qnn.qn2npa(qn1+qn2))
    print("qn1+qn2",qnn.qn2flt(qn1+qn2),qnn.qn2flt(qn1)+qnn.qn2flt(qn2))
    print("qn1-qn2",qnn.qn2npa(qn2-qn1))
    print("qn1+qn2",qnn.qn2flt(qn1-qn2),qnn.qn2flt(qn1)-qnn.qn2flt(qn2))
    print("qn1*qn2",qnn.qn2npa(qn1*qn2))
    print("qn1*qn2",qnn.qn2flt(qn1*qn2),qnn.qn2flt(qn1)*qnn.qn2flt(qn2))
    print("qn1/qn2",qnn.qn2npa(qn1/qn2))
    print("qn1/qn2",qnn.qn2flt(qn1/qn2),qnn.qn2flt(qn1)/qnn.qn2flt(qn2))
    #print("qn1+=qn2",qnn.qn2npa(qn1 += qn2))
    #print("qn1-=qn2",qnn.qn2npa(qn1 -= qn2))
    qn12=qn1-qn2
    print("qn1-qn2",qn12)
    print("qn12==qn12", qn12==qn12)
    print("qn1<qn2",qn1<qn2)
    print("qn1<qn2",qnn.qn2flt(qn1)<qnn.qn2flt(qn2))
    print("qn1>qn2",qn1>qn2)
    print("qn1>qn2",qnn.qn2flt(qn1)>qnn.qn2flt(qn2))
    


np1=np.array([1,2,3])
np2=np.array([4,5,6])
print("np1",np1)
print("np2",np2)

# octagonal qnnumber
print("octagonal")
qn1=qnn.Qnnum(np1,2)  # sqrt(2) type
qn2=qnn.Qnnum(np2,2)
print("qn1",qnn.qn2npa(qn1))
print("qn2",qnn.qn2npa(qn2))
chkop(qn1,qn2)

# decagonal qnnumber
print("decagonal")
qn1=qnn.Qnnum(np1,5) # sqrt(5) type
qn2=qnn.Qnnum(np2,5)
print("qn1",qnn.qn2npa(qn1))
print("qn2",qnn.qn2npa(qn2))
chkop(qn1,qn2)

#dodecagonal qnnumber
print("dodecagonal")
qn1=qnn.Qnnum(np1,3) # sqrt(3) type
qn2=qnn.Qnnum(np2,3)
print("qn1",qnn.qn2npa(qn1))
print("qn2",qnn.qn2npa(qn2))
chkop(qn1,qn2)

