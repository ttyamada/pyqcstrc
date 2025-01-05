#test qnmath operations
import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnmat.qnmat as qnm
import pyqcstrc.prjop.prjop as pro

def chkop(qnv1,qnv2):
    qnv.printqnv("qnv1+qnv2",qnv1+qnv2)
    print("qnv1+qnv2",qnv.qnv2flt(qnv1+qnv2))
    #qnv.printqnv("qnv1+qnv2",qnv.qnv2flt(qnv1)+qnv.qnv2flt(qnv2))
    qnv.printqnv("qnv1-qnv2",qnv2-qnv1)
    print("qnv1-qnv2",qnv.qnv2flt(qnv1-qnv2))
    #qnv.printqnv("qnv1-qnv2",qnv.qnv2flt(qnv1)-qnv.qnv2flt(qnv2))

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
qn0=qnn.Qnnum(np0,2) # sqrt(2) type qnnumber
qn1=qnn.Qnnum(np1,2)
qn2=qnn.Qnnum(np2,2)
qn3=qnn.Qnnum(np3,2)
qn4=qnn.Qnnum(np4,2)

print("qn0",qnn.qn2npa(qn0))
print("qn1",qnn.qn2npa(qn1))
print("qn2",qnn.qn2npa(qn2))
print("qn3",qnn.qn2npa(qn3))
print("qn4",qnn.qn2npa(qn4))

v1=np.array([qn1,qn2,qn3,qn4,qn0,qn0],qnn.Qnnum) #Qnnum array
v2=np.array([qn2,qn1,qn4,qn3,qn0,qn0],qnn.Qnnum) #Qnnum array
qnv1=qnv.Qnvec(v1) # qnvector for vec1
qnv2=qnv.Qnvec(v2) # qnvector for vec2
qnm1=qnm.Qnmat(v1) # 1D qnmatrix
qnm2=qnm.Qnmat(v2) # 2D qnmatrix
print("qnv1",type(qnv1))
print("qnm1",type(qnm1))

qnv.printqnv("qnv1",qnv1)
qnv.printqnv("qnv2",qnv2)
chkop(qnv1,qnv2)

# for octagonal
prj=pro.Octa()
prjm=prj.mt
print("prj",type(prj))
print("prjm",type(prjm))
qnm.printqnm("projection matrix for octaglnal lattice",prjm)
qnm.printqnm("qnm1",qnm1)
qnm3=prjm@qnm1 # qnv3=cls.mt@qnv1.vt
#print("qnv3",qnv3)
qnm.printqnm("qnv3 for octaglnal lattice",qnm3)
#qnv.printqnm("cls.mt@qnv1 for octaglnal lattice",cls.mt@qnv1)


# for deagonal
prj=pro.Deca()
prjm=prj.mt
qnm.printqnm("projection matrix for decagonal lattice",prjm)
qnm3=prjm@qnm1
qnm.printqnm("qnv3 for decagonal lattice",qnm3)
#qnv.printqnm("cls.mt@qnv1 for decagonal lattice",prj.mt@qnv1)

# for deagonal
prj=pro.Dode()
prjm=prj.mt
qnm.printqnm("projection matrix for dodecaglnal lattice",prjm)
qnm3=prj.mt@qnm1
qnm.printqnm("qnv3 for dodecagonal lattice",qnm3)
#qnv.printqnm("cls.mt@qnv1 for dodecagonal lattice",cls.mt@qnv1)



