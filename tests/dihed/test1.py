#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import time
import os
import sys
import cython
import numpy as np

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnmath as qmt
import occdom as od
import vesta as vst

#import od.occupation_domain as od # use original code
#import od.two_occupation_domains as ods # use original code
#import occdom.occdom as od
#import twoods.twoods as ods
if cython.compiled:
    DTYPE_int = cython.long
else:
    DTYPE_int = np.int64

opath='./test1'
try:
    os.makedirs(opath)
except FileExistsError:
    pass
    
isys=4  # octagonal
crs.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qnm.qnmat_init()
qmt.qnmath_init()

M0=qnn.Qnnum([0,0,1])
M1=qnn.Qnnum([1,0,1])
M2=qnn.Qnnum([1,0,2])
v0: DTYPE_int=np.array([M0,M0,M0,M0,M0],dtype=DTYPE_int)
v1: DTYPE_int=np.array([M1,M0,M0,M0,M0],dtype=DTYPE_int) # (1,0,0,0)
v2: DTYPE_int=np.array([M2,M0,M0,M2,M0],dtype=DTYPE_int) # (1,0,0,1)/2
od_asym=np.vstack([v0,v1,v2]).reshape(1,3,6,3)


# Three 6D vectors which define the asymmetric part of the occupation domain of Ammann–Beenker octagonal tiling.
# Note that 5-th and 6-th components of each 6D vectors are dummy, and they correspond to Z coordinate in Epar and Eperp, respectively.
#v0: DTYPE_int=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]],dtype=DTYPE_int)
#v1: DTYPE_int=np.array([[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]],dtype=DTYPE_int) # (1,0,0,0)
#v2: DTYPE_int=np.array([[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1]],dtype=DTYPE_int) # (1,0,0,1)/2
#od_asym=np.vstack([v0,v1,v2]).reshape(1,3,6,3)

# Output 
vst.write_vesta(obj=od_asym, path=opath, basename='od_1_asym', color='r',select='normal')
vst.write_xyz(obj=od_asym, path=opath, basename='od_1_asym')

# Read XYZ file
od_asym=od.read_xyz(path=opath,basename='od_1_asym',select='triangle')

#============================================
# OBJ_1 at (0,0,0,0)
#============================================
time1=time.time()
print("start time",time1)
# make symmetric OD
pos0=np.array([M0,M0,M0,M0,M0]) # 0,0,0,0,0
od_sym_1=od.symmetric(obj=od_asym, centre=pos0)
vst.write_vesta(obj=od_sym_1, path=opath, basename='od_1_sym', color='r',select='normal')
vst.write_xyz(obj=od_sym_1, path=opath, basename='od_1_sym')
# Outline of OBJ_1
od_sym_1_1=od.outline(od_sym_1)
vst.write_vesta(obj=od_sym_1_1,path=opath,basename='od_1_sym_outline',color='r',select='egdes')
vst.write(_xyzobj=od_sym_1_1,path=opath,basename='od_1_sym_outline',select='egdes')

area=od.volume(od_sym_1)
print('area=', area)
time2=time.time()
print("elapsed time for OBJ1",time2-time1,"sec")


#============================================
# OBJ_2 at (1,1,0,0)
#============================================
# shift the symmetric OD to (1,1,0,0)
pos1=np.array([M1,M1,M0,M0,M0]) # 1,1,0,0
od_sym_2=od.shift(obj=od_sym_1, shift=pos1)
vst.write_vesta(obj=od_sym_2, path=opath, basename='od_2_sym', color='b',select='normal')
vst.write_xyz(obj=od_sym_2, path=opath, basename='od_2_sym')
# Outline of OBJ_2
od_sym_2_1=od.outline(od_sym_2)
vst.write_vesta(obj=od_sym_2_1,path=opath,basename='od_2_sym_outline',color='b',select='egdes')
vst.write_xyz(obj=od_sym_2_1,path=opath,basename='od_2_sym_outline',select='egdes')
time3=time.time()
print("elapsed time for OBJ2",time3-time2,"sec")

#============================================
# Intersection of two ODs, OBJ_1 and OBJ_2
#============================================
od_common=ods.intersection(od_sym_1,od_sym_2,verbose=0)
vst.write_vesta(obj=od_common,path=opath,basename='od_common',color='g',select='normal')
vst.write_xyz(obj=od_common,path=opath,basename='od_common')
time4=time.time()
print("elapsed time for intersection of od_sym1 od_sym2",time4-time3)

# Simplification of common part "od_common"
od_common_1_smpl=od.simplification(od_common)
vst.write_vesta(obj=od_common_1_smpl,path=opath,basename='od_common_simpl',color='g',select='normal')
vst.write_xyz(obj=od_common_1_smpl,path=opath,basename='od_common_simpl')
time5=time.time()
print("elapsed time for simplification of common_od",time5-time4,"sec")

# Outline of common part
od_common_1_smpl_outline=od.outline(od_common_1_smpl)
vst.write_vesta(obj=od_common_1_smpl_outline,path=opath,basename='od_common_simpl_outline',color='g',select='egdes')
vst.write_xyz(obj=od_common_1_smpl_outline,path=opath,basename='od_common_simpl_outline',select='egdes')
time6=time.time()
print("elapsed time for outline of common_od",time6-time5,"sec")

#============================================
# Intersection of two ODs, OBJ_1 and OBJ_2 by using 'convex' option
#============================================
od_common_2=ods.intersection_convex(od_sym_1,od_sym_2,verbose=0)
vst.write(_vestaobj=od_common_2,path=opath,basename='od_common_2',color='g',select='normal')
vst.write_xyz(obj=od_common_2,path=opath,basename='od_common_2')
time7=time.time()
print("elapsed time for intersection with convex option",time7-time6,"sec")

# Simplification of common part "od_common_2"
od_common_2_smpl=od.simplification(od_common_2)
vst.write_vesta(obj=od_common_2_smpl,path=opath,basename='od_common_2_simpl',color='g',select='normal')
vst.write_xyz(obj=od_common_2_smpl,path=opath,basename='od_common_2_simpl')
time8=time.time()
print("elapsed time for simplification of common part",time8-time7,"sec")

# Outline of common part
od_common_2=od.outline(od_common_2_smpl)
vst.write_vesta(obj=od_common_2,path=opath,basename='od_common_2_outline',color='g',select='egdes')
vst.write_xyz(obj=od_common_2,path=opath,basename='od_common_2_outline',select='egdes')
time9=time.time()
print("elapsed time for outline of common part",time9-time8,"sec")
