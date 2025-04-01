#if __name__ == "__main__":
import timeit
import os
import sys
import numpy as np
import cython
from numpy.typing import NDArray
import octa2.occupation_domain as od
import octa2.vesta as vst
import octa2.utils as utl

#from occupation_domain import (
#    symmetric,read_xyz,write)

import symmetry as symmetry
if cython.compiled:
    DTYPE_int = cython.long
else:
    DTYPE_int = np.int64

test_dir='../../tests/octa/test'
xyz_dir='../../xyz/octa'
# import asymmetric part of OD(occupation domain) located at origin,0,0,0,0,0,0.
od_asym0 = utl.read_xyz(path=xyz_dir,basename='od_1_asym')
od_asym=cython.declare(cython.long[:,:],od_asym0)
print(od_asym)

pos0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]],dtype=np.int64)
pos=cython.declare(cython.long[:,:],pos0)
od_sym = od.symmetric(obj=od_asym, centre=pos)

vst.write_vesta(obj=od_sym, path=test_dir, basename = 'od_sym', color = 'k')
vst.write_vesta(obj=od_sym, path=test_dir, basename = 'od_sym', color = 'k')

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

