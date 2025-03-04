#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import time
import os
import sys
import cython
import numpy as np
import octa2.occupation_domain as od # use original code
import octa2.two_occupation_domains as ods # use original code
#import occdom.occdom as od
#import twoods.twoods as ods
DTYPE_int = int

opath='./test1'
try:
    os.makedirs(opath)
except FileExistsError:
    pass
    
# Three 6D vectors which define the asymmetric part of the occupation domain of Ammann–Beenker octagonal tiling.
# Note that 5-th and 6-th components of each 6D vectors are dummy, and they correspond to Z coordinate in Epar and Eperp, respectively.
v0: DTYPE_int=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]],dtype=int)
v1: DTYPE_int=np.array([[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]],dtype=int) # (1,0,0,0)
v2: DTYPE_int=np.array([[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1]],dtype=int) # (1,0,0,1)/2
od_asym=np.vstack([v0,v1,v2]).reshape(1,3,6,3)

# Output 
od.write(obj=od_asym, path=opath, basename='od_1_asym', format='vesta', color='r',select='normal')
od.write(obj=od_asym, path=opath, basename='od_1_asym', format='xyz')

# Read XYZ file
od_asym=od.read_xyz(path=opath,basename='od_1_asym',select='triangle')

#============================================
# OBJ_1 at (0,0,0,0)
#============================================
time1=time.time()
print("start time",time1)
# make symmetric OD
pos0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]) # 0,0,0,0,0
od_sym_1=od.symmetric(obj=od_asym, centre=pos0)
od.write(obj=od_sym_1, path=opath, basename='od_1_sym', format='vesta', color='r',select='normal')
od.write(obj=od_sym_1, path=opath, basename='od_1_sym', format='xyz')
# Outline of OBJ_1
od_sym_1_1=od.outline(od_sym_1)
od.write(obj=od_sym_1_1,path=opath,basename='od_1_sym_outline',format='vesta',color='r',select='egdes')
od.write(obj=od_sym_1_1,path=opath,basename='od_1_sym_outline',format='xyz',select='egdes')

area=od.volume(od_sym_1)
print('area=', area)
time2=time.time()
print("elapsed time for OBJ1",time2-time1,"sec")


#============================================
# OBJ_2 at (1,1,0,0)
#============================================
# shift the symmetric OD to (1,1,0,0)
pos1=np.array([[1,0,1],[1,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]) # 1,1,0,0
od_sym_2=od.shift(obj=od_sym_1, shift=pos1)
od.write(obj=od_sym_2, path=opath, basename='od_2_sym', format='vesta', color='b',select='normal')
od.write(obj=od_sym_2, path=opath, basename='od_2_sym', format='xyz')
# Outline of OBJ_2
od_sym_2_1=od.outline(od_sym_2)
od.write(obj=od_sym_2_1,path=opath,basename='od_2_sym_outline',format='vesta',color='b',select='egdes')
od.write(obj=od_sym_2_1,path=opath,basename='od_2_sym_outline',format='xyz',select='egdes')
time3=time.time()
print("elapsed time for OBJ2",time3-time2,"sec")

#============================================
# Intersection of two ODs, OBJ_1 and OBJ_2
#============================================
od_common=ods.intersection(od_sym_1,od_sym_2,verbose=0)
od.write(obj=od_common,path=opath,basename='od_common',format='vesta',color='g',select='normal')
od.write(obj=od_common,path=opath,basename='od_common',format='xyz')
time4=time.time()
print("elapsed time for intersection of od_sym1 od_sym2",time4-time3)

# Simplification of common part "od_common"
od_common_1_smpl=od.simplification(od_common)
od.write(obj=od_common_1_smpl,path=opath,basename='od_common_simpl',format='vesta',color='g',select='normal')
od.write(obj=od_common_1_smpl,path=opath,basename='od_common_simpl',format='xyz')
time5=time.time()
print("elapsed time for simplification of common_od",time5-time4,"sec")

# Outline of common part
od_common_1_smpl_outline=od.outline(od_common_1_smpl)
od.write(obj=od_common_1_smpl_outline,path=opath,basename='od_common_simpl_outline',format='vesta',color='g',select='egdes')
od.write(obj=od_common_1_smpl_outline,path=opath,basename='od_common_simpl_outline',format='xyz',select='egdes')
time6=time.time()
print("elapsed time for outline of common_od",time6-time5,"sec")

#============================================
# Intersection of two ODs, OBJ_1 and OBJ_2 by using 'convex' option
#============================================
od_common_2=ods.intersection_convex(od_sym_1,od_sym_2,verbose=0)
od.write(obj=od_common_2,path=opath,basename='od_common_2',format='vesta',color='g',select='normal')
od.write(obj=od_common_2,path=opath,basename='od_common_2',format='xyz')
time7=time.time()
print("elapsed time for intersection with convex option",time7-time6,"sec")

# Simplification of common part "od_common_2"
od_common_2_smpl=od.simplification(od_common_2)
od.write(obj=od_common_2_smpl,path=opath,basename='od_common_2_simpl',format='vesta',color='g',select='normal')
od.write(obj=od_common_2_smpl,path=opath,basename='od_common_2_simpl',format='xyz')
time8=time.time()
print("elapsed time for simplification of common part",time8-time7,"sec")

# Outline of common part
od_common_2=od.outline(od_common_2_smpl)
od.write(obj=od_common_2,path=opath,basename='od_common_2_outline',format='vesta',color='g',select='egdes')
od.write(obj=od_common_2,path=opath,basename='od_common_2_outline',format='xyz',select='egdes')
time9=time.time()
print("elapsed time for outline of common part",time9-time8,"sec")
