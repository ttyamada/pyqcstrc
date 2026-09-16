# test qnnpyDelaunay module

import sys, os, math
import numpy as np
import cython
from typing import Self

import crsys as crs
import qnnum as qnn   # for qnnumber
import qnvec as qnv
import qnmat as qnm
import qnmath as qmt # for dot product
import qnndarray as qna
import delaunay2D as qnDl

isys=4
crs.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()

ni=crs.ni  # ni= 2 or 3 for dihedral or icosahedral QCs
print("ni",ni)
qna.qnndarray_init()
qmt.qnmath_init()
M0=qnn.Qnnum([0,0,1])
M1=qnn.Qnnum([0,-1,2])
M2=qnn.Qnnum([1,-1,1])
M3=qnn.Qnnum([-2,1,2])
M4=qnn.Qnnum([-2,1,4])
M5=qnn.Qnnum([0,-1,4])
M6=qnn.Qnnum([-2,-1,4])

pv=qnv.zerovs((5,ni)) # for pentagon
pv[0]=qnv.anyv(np.array([M0,M1]))
pv[1]=qnv.anyv(np.array([M0,M2]))
pv[2]=qnv.anyv(np.array([M3,M1]))
pv[3]=qnv.anyv(np.array([M4,M5]))
pv[4]=qnv.anyv(np.array([M4,M6]))
print("pv.shape",pv.shape)  # for test
qnv.printqnvs("pv",pv)

# add points for qnnpyDelaunay
center=qnv.zerov(2)
radius=qnn.Qnnum([999,0,1])
grph=qnDl.Delaunay2D(center,radius)  # points -> qna.QnNdarray
for i in range(5):
    pnt=grph.addPoint(pv[i])  # point list


# get triangles from qnD1
print ("Delaunay triangles:\n", qrph.exportTriangles())