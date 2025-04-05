#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import time
import os
import sys
import cython
import numpy as np
from numpy.typing import (NDArray)

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnmath as qmt
import qnndarray as qna
import occdom as od
import intsct as isc
import vesta as vst
import twoods as ods
import prjop as prj
import numeric as num
import qnsym as qns
import sitesym as ssy
import lattice as lt

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
qna.qnndarray_init()
qnv.qnvec_init()
qnm.qnmat_init()
qmt.qnmath_init()
prj.prjop_init()
brv='p'
lt.lattice_init(brv)
num.numeric_init()
qns.qnsym_init()
qns.Qnsym()  # set symmetry operator
od.occdom_init()
#isc.intsct__init()

brv='p'
lt.lattice_init(brv)


M0=qnn.any([0,0,1])  # 0
M1=qnn.any([1,0,2])  # 1/2
M2=qnn.any([-1,0,2]) #-1/2
#M2=qnn.any([0,1,2])  # sqrt(2)/2
v0=np.array([M0,M0,M0,M0,M0],dtype=qnn.Qnnum)
v1=np.array([M1,M2,M0,M1,M0],dtype=qnn.Qnnum) # (1,-1,0,1,0)/2
v2=np.array([M1,M2,M2,M1,M0],dtype=qnn.Qnnum) # (1,-1,-1,1,0)/2
qnv0=qnv.anyv(v0)
qnv1=qnv.anyv(v1)
qnv2=qnv.anyv(v2)
qnv.printqnv("qnv0",qnv0)
qnv.printqnv("qnv1",qnv1)
qnv.printqnv("qnv2",qnv2)

od_asym=np.vstack([qnv0,qnv1,qnv2]).reshape(1,3,5)
qnv.printqnvs("od_asym",od_asym)
# Output 
vst.write_vesta(od_asym, opath, 'od_1_asym', 'r', 'normal')
vst.write_xyz(od_asym, opath, 'od_1_asym')

# Read XYZ file
#od_asym=vst.read_xyz(opath,'od_1_asym','triangle')

#============================================
# OBJ_1 at (0,0,0,0)
#============================================
time1=time.time()
print("start time",time1)

# make symmetric OD
M0=qnn.Qnnum([0,0,1])
M1=qnn.Qnnum([1,0,1]) 
x0=qna.anya([M0,M0,M0,M0,M0],(5,))  # OD at (0,0,0,0,0)
#x1=qna.anya([M1,M0,M0,M0,M0],(5,))  # OD at (1,0,0,0,0)
x=qnv.anyv(x0)  # origine
#x=qnv.anyv(x1)
qnv.printqnv("x",x)  # for test

# calculate symmetric OD from od_asym by true or fictitious site symmetry operators
irs=ssy.site_symmetry(x)  # true site symmetry operator indices
print("irs before od.symmetric",irs) # for test
for i in range(3):
    qnv.printqnv("od_asym",od_asym[0][i])
od_sym_1=od.symmetric_od(irs,od_asym) # symmetric OD at the origin

vst.write_vesta(od_sym_1, opath, 'od_1_sym', 'r', 'normal')
vst.write_xyz(od_sym_1, opath, 'od_1_sym')

# Outline of OBJ_1
od_sym_1_1=od.outline(od_sym_1)
vst.write_vesta(obj=od_sym_1_1,path=opath,basename='od_1_sym_outline',color='r',select='egdes')
vst.write(_xyzobj=od_sym_1_1,path=opath,basename='od_1_sym_outline',select='egdes')

area=od.volume(od_sym_1)
print('area=', area)
time2=time.time()
print("elapsed time for OBJ1",time2-time1,"sec")

#============================================
# OBJ_2 at (1,0,0,0)
#============================================
# shift the symmetric OD to (1,0,0,0)
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
od_common=isc.intersection(od_sym_1,od_sym_2,verbose=0)
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
