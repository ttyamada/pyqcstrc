#test qnmath operations
import sys
import numpy as np
import qnnum as qnn
import crsys as crs

def chkop(qn1,qn2):
    print("qn1",qnn.qn2flt(qn1))
    print("qn2",qnn.qn2flt(qn2))
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
    #print("qn1-qn2",qn12) # this does not work
    print("qn12==qn12", qn12==qn12)
    print("qn1==qn2",qn1==qn2)
    print("qn1<qn2",qn1<qn2)
    print("qn1<qn2",qnn.qn2flt(qn1)<qnn.qn2flt(qn2))
    print("qn1>qn2",qn1>qn2)
    print("qn1>qn2",qnn.qn2flt(qn1)>qnn.qn2flt(qn2))
    #qn3=qnn.int2qn(3,qn1.N)
    #qnn.printqnn('Qnnum(3)',qn3)
    
    qnn.printqnn("qn1",qn1)
    fl1=qnn.qn2flt(qn1)
    print("fl1",fl1)
    print("N",N)
    #qnt=qnn.flt2qn(fl1,N)
    #qnn.printqnn("qnt",qnt)
    
np1=np.array([1,2,3])
np2=np.array([4,5,6])
print("np1",np1)
print("np2",np2)

# octagonal qnnumber
print("octagonal")
isys=4
crs_o=crs.crsys_init(isys)
N=crs_o.N
qn1=qnn.qnnum_init(np1)  # sqrt(2) type
qn2=qnn.qnnum_init(np2)
print("qn1",qnn.qn2npa(qn1))
print("qn2",qnn.qn2npa(qn2))

chkop(qn1,qn2)

# decagonal qnnumber
print("decagonal")
isys=3
crs_d=crs.crsys_init(isys)
qn1=qnn.qnnum_init(np1) # sqrt(5) type
qn2=qnn.qnnum_init(np2)
print("qn1",qnn.qn2npa(qn1))
print("qn2",qnn.qn2npa(qn2))
chkop(qn1,qn2)

#dodecagonal qnnumber
print("dodecagonal")
isys=5
isys=crs.crsys_init(isys)
qn1=qnn.qnnum_init(np1) # sqrt(3) type
qn2=qnn.qnnum_init(np2)
print("qn1",qnn.qn2npa(qn1))
print("qn2",qnn.qn2npa(qn2))
chkop(qn1,qn2)

