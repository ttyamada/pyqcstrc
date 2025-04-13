# test qnnpyDelaunay module

import numpy as np

import crsys as crs
import qnnum as qnn   # for qnnumber
import qnvec as qnv
import qnmath as qmt # for dot product
import qnndarray as qna
import cython

from typing import Self
import qnnpyDelaunay as qnDl

isys=4
crs.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qna.qnndarray_init()
qmt.qnmath_init()

n=crs.n
p=qnv.zerovs(5,n) # for pentagon

# add points for qnnpyDelaunay
grph=qnD1.Glaph()
for i in range(5):
    grph.add_point(p[i])

grph.generateDelaunayMesh()

# get triangles from qnD1.Glaph




