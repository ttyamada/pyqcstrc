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
import qnnpyDelaunay as qnDl

isys=4
crs.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qna.qnndarray_init()
qmt.qnmath_init()
M0=qnn.Qnnum([0,0,1])
M1=qnn.Qnnum([0,-1,2])
M2=qnn.Qnnum([1,-1,1])
M3=qnn.Qnnum([-2,1,2])
M4=qnn.Qnnum([-2,1,4])
M5=qnn.Qnnum([0,-1,4])
M6=qnn.Qnnum([-2,-1,4])

ni=crs.ni
p=qnv.zerovs(5,ni) # for pentagon
p[0]=qnv.anyv([M0,M1])
p[1]=qnv.anyv([M0,M2])
p[2]=qnv.anyv([M3,M1])
p[3]=qnv.anyv([M4,M5])
p[4]=qnv.anyv({M4,M6})

# add points for qnnpyDelaunay
grph=qnD1.Glaph()
for i in range(5):
    grph.add_point(p[i])

grph.generateDelaunayMesh()
# number of triangles in the pentagon

# get triangles from qnD1.Glaph
for i,tri in enumerate(grph._triangles):
    print("tri",i,end=" ")
    qnv.printqnvs(" ",tri)
