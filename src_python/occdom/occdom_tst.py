
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
import qnndarray as qna
from occdom import (occdom_init,symmetric,write)

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

test_dir='../../tests/octa2/tests'
xyz_dir='../../xyz/octa'
# import asymmetric part of OD(occupation domain) located at origin,0,0,0,0,0,0.
od_asym = vst.read_xyz(path=xyz_dir,basename='od_1_asym')
shape=od_asym.shape
ndim=len(shape)

print("type(od_asym)",type(od_asym),"od_saym.shape",shape,"ndim",ndim)  # for test
if ndim==2: # vertex
    for i in range(shape[0]):
        qnv.printqnv("od_asym[i]",od_asym[i])
elif ndim==3: # triangle
    for i in range(shape[0]):
        for j in range(shape[1]):
            qnv.printqnv("od_asym[i][j]",od_asym[i][j])

pos0 = qnv.zerov(5)
irs=ssm.site_symmetry(pos0)
qnv.printqnv("pos0",pos0)
od_sym = symmetric(irs,od_asym)
vst.write_vesta(od_sym, test_dir, 'od_sym', 'k', 'normal')
vst.write_xyz(od_sym, test_dir, 'od_sym')

# move STRT OD to a position 1 1 1 0 -1 0.
#pos_b1=np.array([[ 1, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 1],[ 0, 0, 1]]) # b_1
#strt_pos1=shift(obj = strt_sym, shift = pos_b1)
#write(pod=strt_pos1, path='.', basename='obj_strt', format='xyz')
#write(obj=strt_pos1, path='.', basename='obj_strt', format='vesta', color='b')

# intersection of "asymmetric part of strt" and "strt at position pos_b1"
#    flag = 0,    with rough intersection chacking (faster)
#    flag = 1, without rough intersection chacking
#twoODs=TWO_ODs(pod1=strt_asym, pod2=strt_pos1, path='.',filename='common.xyz',flag=0,verbose=0)
#intersection=Intersection(pod1=tmp1.reshape(1,4,6,3), pod2=tmp2.reshape(1,4,6,3), path='.',filename='common.xyz',flag=0,verbose=0)
#common_part=twoODs.intersection()

# export common_part in VESTA formated file.
#write(obj=common_part, path='.', basename='common', format='vesta', color='r')
    