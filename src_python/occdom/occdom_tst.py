
import timeit
import os
import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnndarray as qna
import qnvec as qnv
import qnmat as qnm
import numeric as num
import utils as utl
import vesta as vst
import qnsym as qns
import lattice as lt
import sitesym as ssm
import intsct as isct
import prjop as prj
from occdom import (occdom_init,symmetric,write,shift)
import tr6to5 as tr5
import tr7to5 as tr7
from numpy.typing  import (NDArray)

print ('argument list', sys.argv)
if len(sys.argv) != 2:
    print("Usage : python occdom_tst.py isys")
    print(" isys : 3,4 or 5 for decag, octag or dodecag QCs")
    exit()
isys = int(sys.argv[1])
print ("isys",isys)

#isys=4 # for octagonal
crs.crsys_init(isys)
qnn.qnnum_init()
qna.qnndarray_init()
qnv.qnvec_init()
qnm.qnmat_init()
prj.prjop_init()
qns.qnsym_init()
lt.lattice_init('p')
ssm.sitesym_init()

occdom_init()

#test_dir='../../tests/octa2/tests'
#xyz_dir='../../xyz/octa'
if isys==4:  # octagonal
    M0=qnn.Qnnum([0,0,1])  # 0
    M1=qnn.Qnnum([1,0,2])  # 1/2
    M2=qnn.Qnnum([-1,0,2]) #-1/2
    oc_asym0_=np.array([\
        [M0,M0,M0,M0,M0],\
        [M1,M2,M0,M1,M0],\
        [M1,M2,M2,M1,M0],\
        ],dtype=qnn.Qnnum)
    od_asym_nd=qna.anya(oc_asym0_,(3,5))
    od_asym=od_asym_nd.reshape((1,3,5))
elif isys==3:  # decagonal
    M0=qnn.Qnnum([0,0,1])  # 0
    M1=qnn.Qnnum([1,0,1])  # 1
    M2=qnn.Qnnum([-1,0,1]) #-1
    M3=qnn.Qnnum([1,0,2])  # 1/2
    od_asym_=np.array([\
        [M0,M0,M0,M0,M0,M0],\
        [M0,M0,M0,M0,M1,M0],\
        [M1,M0,M0,M0,M1,M0]\
        ],dtype=qnn.Qnnum)
    od_asym_nd=qna.anya(od_asym_,(3,6))
    od_asym=tr5.tr6to5e(od_asym_nd).reshape(1,3,5)
elif isys==5:  # dodecagonal
    M0=qnn.Qnnum([0,0,1])  # 0
    M1=qnn.Qnnum([1,0,1])  # 1
    M2=qnn.Qnnum([-1,0,1]) #-1
    M3=qnn.Qnnum([0,1,3])  # sqrt(3)/3=1/sqrt(3)
    od_asym_=np.array([\
        [M0,M0,M0,M0,M0,M0,M0],\
        [M1,M0,M0,M0,M0,M0,M0],\
        [M3,M0,M0,M0,M0,M3,M0]\
        ],dtype=qnn.Qnnum) # stampfli tiling
    od_asym_nd=qna.anya(od_asym_,(3,7))
    od_asym=tr7.tr7to5e(od_asym_nd).reshape(1,3,5)
else:
    print("isys=2 (icosahedral) not implemented yet")
    exit()
# if od_asym is represented by internal space compoenet of triangles/tetrahedra
# its symmetric version is obtained by the symmetry operator in the internal space
# this simplifies later calculations

# transform od_asym nD vertex coordinates to its internal space components
od_asym=prj.projection3_sets_numerical(od_asym)

# import asymmetric part of OD(occupation domain) located at origin,0,0,0,0,0,0.
#od_asym = vst.read_xyz(path=xyz_dir,basename='od_1_asym')
shape=od_asym.shape
ndim=len(shape)

print("type(od_asym)",type(od_asym),"od_saym.shape",shape,"ndim",ndim)  # for test
if ndim==1: # vertex
    qnv.printqnv("od_asym[i]",od_asym[i])
elif ndim==2: # triangle
    for i in range(shape[0]):
        qnv.printqnv("od_asym[i]",od_asym[i])
elif ndim==3: #triangles
    for i in range(shape[0]):
        for j in range(shape[1]):
            qnv.printqnv("od_asym[i][j]",od_asym[i][j])

if isys!=3:
    pos0 = qnv.zerov(5) # origin
    qnv.printqnv("pos0",pos0)  # for test
else:
    M0=qnn.Qnnum([0,0,1])  # 0
    M1=qnn.Qnnum([1,0,5])  # 1/5
    qnv_=np.array([M1,M1,M1,M1,M0])
    #pos_=NDArray(qnv_,(5))
    pos0=qnv.anyv(qnv_)
    qnv.printqnv("pos0",pos0)  # for test

irs=ssm.site_symmetry(pos0)  # site symmetry operator indices
qnv.printqnv("pos0",pos0)

od_sym = symmetric(irs,od_asym) # id_asyn : internal space component of od corner vectors

for i,tri in enumerate(od_sym):
    od_sym[i]=isct.counter_clockwise(tri)
vst.write_vesta(od_sym, '.', 'od_sym', 'r', 'normal')
vst.write_xyz(od_sym,'.', 'od_sym')

# move od_sym to a position 1 0 0 0 0
if isys != 3:
    M3=qnn.any([1,0,1])
    x0=np.array([M3,M0,M0,M0,M0]) # origin shift by (1,0,0,0,0)
else:
    x0=np.array([M1,M1,M1,M1,M0]) # origin shift by (1,1,1,1,0)/5
qnx0=qnv.anyv(x0)
v0=prj.projection3(qnx0)
qnv.printqnv("v0",v0)  # for test
# calculate shifted od
od_sym1=qnv.sub_vectors_qn(od_sym, v0) # shift by v0
#od_sym1=shift(od_sym, pos_b1)

vst.write_xyz(od_sym1, '.', 'od_sym1')
vst.write_vesta(od_sym1, '.', 'od_sym1', 'b')

# intersection of "asymmetric part of strt" and "strt at position pos_b1"
#    flag = 0, with rough intersection chacking (faster)
#    flag = 1, without rough intersection chacking
#twoODs=TWO_ODs(pod1=strt_asym, pod2=strt_pos1, path='.',filename='common.xyz',flag=0,verbose=0)
print("od_sym.shape",od_sym.shape)  # for test
print("od_sym1.shape",od_sym1.shape)  # for test

nod=0
ntr=0
for i,tri1 in enumerate(od_sym):
    for j,tri2 in enumerate(od_sym1): 
        nx,x=isct.intersection_two_triangles(od_sym[i],od_sym1[j])
        qnv.printqnvs("cross points of two triangles",x)
        ny,y=isct.common_points(od_sym[i],od_sym1[j])
        qnv.printqnvs("common points in triangles",y)
        if nx==0 and ny==0:
            continue
        n,z=isct.common_part(nx,x,ny,y)
        print("z.shape",z.shape)  # for test
        qnv.printqnvs("common part",z)  # for test

        n,z=isct.rmv_overlapedx(z,n)
        print("number of vertices in common part",n)
        qnv.printqnvs("common part",z)  # for test
        if n>=3:
            nod+=1
            print("nod",nod)
        if n==3: # triangle
            # stack triangle here
            if ntr==0:
                tria=z
            else:
                tria=np.vstack([tria,z])
            ntr+=1
print("ntr",ntr)
intsct=tria.reshape(ntr,3,2)
print("intsct.shape",intsct.shape)

for i in range(ntr):
    qnv.printqnvs("intsct[i]",intsct[i])
vst.write_vesta(intsct, '.', 'intsct', 'b', 'normal')