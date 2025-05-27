# ====================================================================================================
# Produced by the Free Edition of C++ to Python Converter.
# Purchase a Premium Edition license at:
# https://www.tangiblesoftwaresolutions.com/order/order-cplus-to-python.html
# ====================================================================================================

import numpy
from typing import TypeVar

from qnnum import *
from qnvec import *
from geometry_n import *
from delaunay_n import *
from number_hpp import Number
import number
import off


#T = TypeVar('T',float, Qnnum)

# template version for using qnnumber points

# determinant of 2x2 matrix

# double version (using same template)
#def main():

isys = 4 # for octagonal
crsys.crsys_init(isys)
qnnum_init()
qnvec_init()

# data for qnnum points (including overlapped points)
# numpy.array([52,2,3])
vt =numpy.array([
    [[ -1, 0, 1], [0, 0, 1]],
    [[ -1, 1, 2],[ 0, 0, 1]],
    [[ -1, 1, 2], [ 1, 0, 2]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -6, -1, 4], [ -2, -1, 4]],
    [[ -3, 0, 2], [ -1, -1, 2]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -1, 0, 1], [ 1, 1, 2]],
    [[ -3, 0, 2], [ 1, 1, 2]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -2, 1, 4], [ -2, -1, 4]],
    [[ -1, 1, 2], [ -1, 0, 2]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -3, -1, 2], [ 0, 0, 1]],
    [[ -3, -1, 2], [ -1, 0, 2]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -2, 1, 4], [ 2, 1, 4]],
    [[ -1, 0, 2], [ 1, 1, 2]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -1, 0, 1], [ -1, -1, 2]],
    [[ -1, 0, 2], [ -1, -1, 2]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -6, -1, 4], [ 2, 1, 4]],
    [[ -3, -1, 2], [ 1, 0, 2]],
    [[ -2, 1, 4], [ 2, 1, 4]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -1, 1, 2], [ 1, 0, 2]],
    [[ -3, -1, 2], [ 0, 0, 1]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -3, -1, 2], [ 1, 0, 2]],
    [[ -2, 1, 4], [ -2, -1, 4]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -1, 0, 2], [ -1, -1, 2]],
    [[ -1, 0, 1], [ 1, 1, 2]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -1, 0, 2], [ 1, 1, 2]],
    [[ -6, -1, 4], [ -2, -1, 4]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -3, -1, 2], [ -1, 0, 2]],
    [[ -1, 1, 2], [ 0, 0, 1]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -1, 1, 2], [ -1, 0, 2]],
    [[ -6, -1, 4], [ 2, 1, 4]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -3, 0, 2], [ 1, 1, 2]],
    [[ -1, 0, 1], [ -1, -1, 2]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -3, 0, 2], [ -1, -1, 2]],
    [[ -3, 0, 2], [ 1, 1, 2]],
    [[ -1, 0, 1], [ -1, -1, 2]],
    [[ -1, 0, 1], [ 0, 0, 1]],
    [[ -3, 0, 2], [ -1, -1, 2]]
    ])

np=vt.shape[0] # 52=17*3+1 (one point is redundunt for 17 triangles?)
print("np",np)
#qnvf=numpy.zeros((np,2),dtype=float)
#print("ndim",qnvf.ndim)
#print("shape",qnvf.shape)

qnv=numpy.zeros((np,2),dtype=Qnnum)
print("qnv.shape() ",qnv.shape)
for i in range(0, np):
    for j in range(0, 2):
        qnv[i][j] = Qnnum(vt[i][j]) # qnnumber array
#print("np ",np)
printqnv("point",qnv)  # for test

# calculate independent qnnum points
#points = Globals.red_points(qnv, np)
#qnvi = numpy.unique(qnv) # numpy ndarray this does not work for qnvec
#np = len(qnvi)

qnvi = Number.unique(qnv,Qnnum)
print("qnvi.shape ",qnvi.shape)
printqnvs("qnvi",qnvi) # for test

np=qnvi.shape[0]
print("np ",np)

# points should be qnvec array
points=anyv(qnvi)
printqnvs("point ",points) # print qnnum coordinates

v = [[0 for _ in range(2)] for _ in range(np)]
pointsf = []
for i in range(0, np):
    x = qn2flt(points[i].x)
    y = qn2flt(points[i].y)
    v[i][0] = x
    v[i][1] = y
    #std::cout<<"x "<<points[i].x<< " y "<<points[i].y<<std::endl; // for test
    print(x, end = '')
    print(" ", end = '')
    print(y, end = '')
    print()
    pointsf.emplace_back(x,y)

# Insert them into a triangulation and draw a PDF
# delaunay::triangulate returns vector<triangle<double>>

tria = triangulate(pointsf) # Delaynay triangulation

n = len(tria)
print("number of triangles ",n)

if n == 0:
    sys.exit(0)

v1 = [0 for _ in range(2)]
v2 = [0 for _ in range(2)]
det = [0 for _ in range(n)] # determinant
# v1 = b-a v2=c-a
for i in range(0, n):
    v1[0] = tria[i].b.x - tria[i].a.x
    v1[1] = tria[i].b.y - tria[i].a.y

    v2[0] = tria[i].c.x - tria[i].a.x
    v2[1] = tria[i].c.y - tria[i].a.y
    det[i] = Globals.get_det(v1, v2)

a = point()
b = point()
c = point()
eps = 1.e-5
vol = 0.0
nzt = 0
i = 0
for t in tria:
    a.copy_from(t.a)
    b.copy_from(t.b)
    c.copy_from(t.c) # three points a,b,c
    print(a.x," ",a.y," ",b.x," ",b.y," ",c.x," ",c.y," det ",det[i])
    #print()

    #if Globals.abs(det[i]) > eps:
    if qnn.abs(det[i]) > qnn.zero():
        tria[nzt].a.copy_from(a)
        tria[nzt].b.copy_from(b)
        tria[nzt].c.copy_from(c)
        det[nzt] = det[i]
        vol += Globals.abs(det[nzt])
        nzt += 1
    i += 1
print("number of finite volume triangles ",nzt," total volume ",vol)
#print()

indx = [[0 for _ in range(3)] for _ in range(n)]
eps = 0.000001
for i in range(0, n):
    indx[i][0] = get_indx(tria[i].a.x, tria[i].a.y,pointsf,eps)
    indx[i][1] = get_indx(tria[i].b.x, tria[i].b.y,pointsf,eps)
    indx[i][2] = get_indx(tria[i].c.x, tria[i].c.y,pointsf,eps)

"""
    # for .off file
    if det[i] > 0.0:
        print(3," ",indx[i][0]," ",indx[i][1]," ",indx[i][2])
        #print()
    else:
        print(3," ",indx[i][0]," ",indx[i][2]," ",indx[i][1])
        #print()
print()
"""

off.wt_off(v, np, indx, n, ".", "dltst_n")
