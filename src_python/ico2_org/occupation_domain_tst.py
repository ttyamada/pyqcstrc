#if __name__ == "__main__":
import timeit
import os
import sys
#sys.path.append('.')
import numpy as np
from occupation_domain import (read_xyz,symmetric,shift)

# import asymmetric part of STRT OD(occupation domain) located at origin,0,0,0,0,0,0.
strt_asym = read_xyz(path='../xyz',basename='strt_aysmmetric')
write(obj=strt_asym, path='.', basename = 'obj_seed', format='vesta', color = 'k')

# generat STRT OD located at 0,0,0,0,0,0 by symmetric operations (m-3-5).
pos0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
strt = symmetric(asymmetric_part_obj = strt_asym, position = pos0)

# move STRT OD to a position 1 1 1 0 -1 0.
pos_b1=np.array([[ 1, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 1],[ 0, 0, 1]]) # b_1
strt_pos1=shift(obj = strt_sym, shift = pos_b1)
write(pod=strt_pos1, path='.', basename='obj_strt', format='xyz')
write(obj=strt_pos1, path='.', basename='obj_strt', format='vesta', color='b')

# intersection of "asymmetric part of strt" and "strt at position pos_b1"
#    flag = 0,    with rough intersection chacking (faster)
#    flag = 1, without rough intersection chacking
#twoODs=TWO_ODs(pod1=strt_asym, pod2=strt_pos1, path='.',filename='common.xyz',flag=0,verbose=0)
#intersection=Intersection(pod1=tmp1.reshape(1,4,6,3), pod2=tmp2.reshape(1,4,6,3), path='.',filename='common.xyz',flag=0,verbose=0)
#common_part=twoODs.intersection()

# export common_part in VESTA formated file.
#write(obj=common_part, path='.', basename='common', format='vesta', color='r')
    