#if __name__ == "__main__":
import timeit
import os
import sys
import numpy as np
#try:
#import qnmath as qmt #math12
#import dode2.math1 as math1
#import utils.utils as utils
#import dode2.symmetry as symmetry
#import dode2.intsct as intsct
#import dode2.projection12 as proj
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import numeric as num
import utils as utils
import vesta as vst
import qnsym as qns
import intsct as isct
import prjop as prj
import occdom as occ
    
test_dir='../../tests/dode2/tests'
xyz_dir='../../xyz/dode'
# import asymmetric part of OD(occupation domain) located at origin,0,0,0,0,0,0.
od_asym = vst.read_xyz(path=xyz_dir,basename='od_vertex_asymmetric')
print(od_asym)

pos0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
png = "p12m"
od_sym = occ.symmetric(obj = od_asym, centre = pos0, png = png)
occ.write(obj=od_sym, path=test_dir, basename = 'od_sym', format='vesta', color = 'k')
occ.write(obj=od_sym, path=test_dir, basename = 'od_sym', format='vesta', color = 'k')

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
    