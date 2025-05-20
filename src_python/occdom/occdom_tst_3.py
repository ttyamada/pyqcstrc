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

# main program
print ('argument list', sys.argv)
if len(sys.argv) != 2:
    print("Usage : python occdom_tst.py isys")
    print(" isys : 3,4 or 5 for decag, octag or dodecag QCs")
    exit()
isys = int(sys.argv[1])
print ("isys",isys)
od_asym=progam_init(isys)

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
if isys!=3:
    od_sym1=qnv.sub_vectors_qn(od_sym, v0) # shift by v0
else:
    tau=-qnn.Qnnum([1,1,2])
    od_sym1_=od_sym*tau
    od_sym1=qnv.sub_vectors_qn(od_sym1_, v0) # shift by v0
#od_sym1=shift(od_sym, pos_b1)

vst.write_xyz(od_sym1, '.', 'od_sym1')
vst.write_vesta(od_sym1, '.', 'od_sym1', 'b')

# intersection of "asymmetric part of strt" and "strt at position pos_b1"
#    flag = 0, with rough intersection chacking (faster)
#    flag = 1, without rough intersection chacking
#twoODs=TWO_ODs(pod1=strt_asym, pod2=strt_pos1, path='.',filename='common.xyz',flag=0,verbose=0)
print("od_sym.shape",od_sym.shape)  # for test
print("od_sym1.shape",od_sym1.shape)  # for test

tria, ntr = get_triangles(od_sym,odsym_1)

print("ntr",ntr)
intsct=tria.reshape(ntr,3,2)
print("intsct.shape",intsct.shape)

for i in range(ntr):
    qnv.printqnvs("intsct[i]",intsct[i])
vst.write_vesta(intsct, '.', 'intsct', 'b', 'normal')
