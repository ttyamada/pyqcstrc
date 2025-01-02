#test qnmath operations
import sys
import numpy as np
from pyqcstrc.qnnum import qnnum
from pyqcstrc.qnvec import qnvec

def chkop(vec1,vec2):
    print("vec1+vec2",qnvec.vec2npa(vec1+vec2))
    print("vec1+vec2",qnvec.vec2flt(vec1+vec2),qnvec.vec2flt(vec1)+qnvec.vec2flt(vec2))
    print("vec1-vec2",qnvec.vec2npa(vec2-vec1))
    print("vec1-vec2",qnvec.vec2flt(vec1-vec2),qnvec.vec2flt(vec1)-qnvec.vec2flt(vec2))


np1=np.array([1,2,3])
np2=np.array([4,5,6])
np3=np.array([0,0,1])

print("np1",np1)
print("np2",np2)

# octagonal qnnumber
print("octagonal")
qn1=qnnum.Qnnum(np1,2)
qn2=qnnum.Qnnum(np2,2)
qn3=qnnum.Qnnum(np3,2)

print("qn1",qnnum.qn2npa(qn1))
print("qn2",qnnum.qn2npa(qn2))
print("qn2",qnnum.qn2npa(qn3))

vec1=[qn1,qn2,qn3] #qnnum array
vec2=[qn2,qn1,qn3] #qnnum array
qnv1=Qnvec(vec1) # qnvector for vec1
qnv2=Qnvec(vec2) # qnvector for vec2

chkop(qnv1,qnv2)
