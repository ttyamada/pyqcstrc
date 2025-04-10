
import timeit
import os
import sys
import numpy as np
import cython

#try:
#import qnmath as qmt #math12
#import dode2.math1 as math1
#import utils.utils as utils
#import dode2.symmetry as symmetry
#import dode2.intsct as intsct
#import dode2.projection12 as proj
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
import intsct as its
from occdom import (occdom_init,symmetric,write,shift)

isys=4 # for octagonal
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
M0=qnn.Qnnum([0,0,1])  # 0
M1=qnn.Qnnum([1,0,2])  # 1/2
M2=qnn.Qnnum([-1,0,2]) #-1/2
oc_asym0=np.array([\
    [M0,M0,M0,M0,M0],\
    [M1,M2,M0,M1,M0],\
    [M1,M2,M2,M1,M0],\
    ],dtype=qnn.Qnnum)
od_asym_nd=qna.anya(oc_asym0,(3,5))
od_asym_nd=od_asym_nd.reshape((1,3,5))
# if od_asym is represented by internal space compoenet of triangles/tetrahedra
# its symmetric version is obtained by the symmetry operator in the internal space
# this simplifies later calculations

# transform od_asym nD vertex coordinates to its internal space components
od_asym=prj.projection3_sets_numerical(od_asym_nd)

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

pos0 = qnv.zerov(5)
irs=ssm.site_symmetry(pos0)
qnv.printqnv("pos0",pos0)

od_sym = symmetric(irs,od_asym)
vst.write_vesta(od_sym, '.', 'od_sym', 'r', 'normal')
vst.write_xyz(od_sym,'.', 'od_sym')

# move od_sym to a position 1 0 0 0 0
M3=qnn.any([1,0,1])
x0=np.array([M3,M0,M0,M0,M0])
qnx0=qnv.anyv(x0)
v0=prj.projection3(qnx0)
qnv.printqnv("v0",v0)  # for test
# calculate shifted od
od_sym1=qnv.sub_vectors_qn(od_sym, v0)
#od_sym1=shift(od_sym, pos_b1)

vst.write_xyz(od_sym1, '.', 'obj_sym1')
vst.write_vesta(od_sym1, '.', 'obj_sym1', 'b')

# intersection of "asymmetric part of strt" and "strt at position pos_b1"
#    flag = 0, with rough intersection chacking (faster)
#    flag = 1, without rough intersection chacking
#twoODs=TWO_ODs(pod1=strt_asym, pod2=strt_pos1, path='.',filename='common.xyz',flag=0,verbose=0)
intersection=its.intersection_two_triangles(od_sym,od_sym1)
#common_part=twoODs.intersection()

# export common_part in VESTA formated file.
#write(obj=common_part, path='.', basename='common', format='vesta', color='r')