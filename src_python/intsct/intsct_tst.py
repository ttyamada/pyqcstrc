import sys
import numpy as np
from numpy.typing import NDArray
import time # in object_subtraction_dev1, tetrahedron_not_obj
import itertools
import cython

import crsys as crs
import intsct as isct
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import numeric as num
import qnndarray as qna
import prjop as prj
import vesta as vst
import utils as utl
import tr7to5 as tr5
import tr6to5 as tr3

#if __name__ == '__main__':

# test


if len(sys.argv) != 2:
    print("Usage : python intsct_tst.py isys")
    print(" isys : 3,4 or 5 for decag, octag or dodecag QCs")
    exit()
isys = int(sys.argv[1])
print ("isys",isys)

#print ('argument list', sys.argv)

#isys=4  # for octabonal
crs.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qnm.qnmat_init()
prj.prjop_init()
qna.qnndarray_init()
if isys==4:  # octagonal
    M0=qnn.Qnnum([0,0,1])  # 0
    M1=qnn.Qnnum([1,0,2])  # 1/2
    M2=qnn.Qnnum([-1,0,2]) #-1/2
    od_asym0_=np.array([\
        [M0,M0,M0,M0,M0],\
        [M1,M2,M0,M1,M0],\
        [M1,M2,M2,M1,M0]\
        ],dtype=qnn.Qnnum)
    od_asym1_=np.array([\
        [M0,M0,M0,M0,M0],\
        [M2,M1,M0,M2,M0],\
        [M2,M1,M2,M2,M0]\
        ],dtype=qnn.Qnnum)
    od_asym0=qna.anya(od_asym0_,(3,5))
    od_asym1=qna.anya(od_asym1_,(3,5))
elif isys==3:  # decagonal
    M0=qnn.Qnnum([0,0,1])  # 0
    M1=qnn.Qnnum([1,0,1])  # 1
    M2=qnn.Qnnum([-1,0,1]) #-1
    M3=qnn.Qnnum([1,0,2])  # 1/2
    od_asym0_=np.array([\
        [M0,M0,M0,M0,M0,M0],\
        [M1,M0,M0,M0,M0,M0],\
        [M3,M0,M0,M3,M0,M0]\
        ],dtype=qnn.Qnnum)
    od_asym1_=np.array([\
        [M0,M0,M0,M0,M0,M0],\
        [M2,M0,M0,M0,M0,M0],\
        [M3,M0,M0,M3,M0,M0]\
        ],dtype=qnn.Qnnum)
    od_asym0_nd=qna.anya(od_asym0_,(3,6))
    od_asym1_nd=qna.anya(od_asym1_,(3,6))
    od_asym0=tr5.tr6to5e(od_asym0_nd)
    od_asym1=tr5.tr6to5e(od_asym1_nd)
elif isys==5:  # dodecagonal
    M0=qnn.Qnnum([0,0,1])  # 0
    M1=qnn.Qnnum([1,0,1])  # 1
    M2=qnn.Qnnum([-1,0,1]) #-1
    M3=qnn.Qnnum([0,1,3])  # sqrt(3)/3=1/sqrt(3)
    od_asym0_=np.array([\
        [M0,M0,M0,M0,M0,M0,M0],\
        [M1,M0,M0,M0,M0,M0,M0],\
        [M3,M0,M0,M0,M0,M3,M0]\
        ],dtype=qnn.Qnnum) # stampfli tiling
    od_asym1_=np.array([\
        [M0,M0,M0,M0,M0,M0,M0],\
        [M2,M0,M0,M0,M0,M0,M0],\
        [M3,M0,M0,M0,M0,M3,M0]\
        ],dtype=qnn.Qnnum)   # stampfli tiling
    od_asym0_nd=qna.anya(od_asym0_,(3,7))
    od_asym1_nd=qna.anya(od_asym1_,(3,7))
    od_asym0=tr5.tr7to5e(od_asym0_nd)
    od_asym1=tr5.tr7to5e(od_asym1_nd)
else:
    print("not applicable to icosahedral QCs")
    exit()

triang_1=prj.projection3_sets_numerical(od_asym0) #.reshape((1,3,2))
triang_2=prj.projection3_sets_numerical(od_asym1) #.reshape((1,3,2))
triang_1=isct.counter_clockwise(triang_1).reshape(1,3,2)
triang_2=isct.counter_clockwise(triang_2).reshape(1,3,2)
shft=qnv.zerov(2)

shft[0]=qnn.one() # (1,0)
#qnv.printqnv("shft",shft)  # for test
triang_2=utl.shift_object(triang_2,shft)

print("triang_1.shape",triang_1.shape)  # for test
print("triang_2.shape",triang_2.shape)  # for test
qnv.printqnvs("triang_1",triang_1)  # for test
qnv.printqnvs("trinag_2",triang_2)  # for test
vst.write_vesta(triang_1, '.', 'triang_1', 'r', 'normal')
vst.write_vesta(triang_2, '.', 'triang_2', 'b', 'normal')
vst.write_xyz(triang_1,'.', 'triang_1')
vst.write_xyz(triang_2,'.', 'triang_2')
#a=num.check_intersection_two_segment_numerical_nd_tau(segment_1,segment_2)
#print(a)
nx,x=isct.intersection_two_triangles(triang_1[0], triang_2[0]) 
qnv.printqnvs("cross points of two triangles",x)

ny,y=isct.common_points(triang_1[0],triang_2[0])
qnv.printqnvs("common points in triangles",y)

n,z=isct.common_part(nx,x,ny,y)
print("z.shape",z.shape)  # for test
qnv.printqnvs("common part",z)  # for test
n,z=isct.rmv_overlapedx(z,n)
print("number of vertices in common part",n)
qnv.printqnvs("common part",z)  # for test
z=z.reshape((1,3,2))
vst.write_vesta(z, '.', 'common_part', 'y', 'normal')

# in general casse, z is a convex polygon and it is
# decomposed into triangles by Delaunay triangulation
# 