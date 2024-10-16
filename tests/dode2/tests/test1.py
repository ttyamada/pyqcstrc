#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

#import timeit
#import os
import sys
import numpy as np
import pyqcstrc.dode2.occupation_domain as od
import pyqcstrc.dode2.two_occupation_domains as ods

opath='./test1'
#xyzpath='../../../xyz/dode'

# import asymmetric part of OD(occupation domain) located at origin, 0,0,0,0,0,0.
#od_asym=od.read_xyz(path=xyzpath,basename='od_vertex_asymmetric')

# Three 6D vectors which define the asymmetric part of the occupation domain of Nizeki-Gahler dodecagonal tiling.
# Note that 5-th and 6-th components of each 6D vectors are dummy, and they correspond to Z coordinate in Epar and Eperp, respectively.
v0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v1=np.array([[ 0,-2, 3],[ 0, 2, 3],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v2=np.array([[ 0, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
od_asym=np.vstack([v0,v1,v2]).reshape(1,3,6,3)
od.write(obj=od_asym, path=opath, basename='od_1_asym', format='xyz')

# make symmetric OD
pos0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]) # 0,0,0,0,0
od_sym_1=od.symmetric(obj=od_asym, centre=pos0)
od.write(obj=od_sym_1, path=opath, basename='od_1_sym', format='vesta', color='r',select='normal')
od.write(obj=od_sym_1, path=opath, basename='od_1_sym', format='xyz')
# Outline of OBJ_1
od_sym_1_1=od.outline(od_sym_1)
od.write(obj=od_sym_1_1,path=opath,basename='od_1_sym_outline',format='vesta',color='r',select='egdes')
od.write(obj=od_sym_1_1,path=opath,basename='od_1_sym_outline',format='xyz',select='egdes')

# shift the symmetric OD
pos1=np.array([[0,0,1],[1,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]) # 0,1,0,0,0
od_sym_2=od.shift(obj=od_sym_1, shift=pos1)
od.write(obj=od_sym_2, path=opath, basename='od_2_sym', format='vesta', color='b',select='normal')
od.write(obj=od_sym_2, path=opath, basename='od_2_sym', format='xyz')
# Outline of OBJ_2
od_sym_2_1=od.outline(od_sym_2)
od.write(obj=od_sym_2_1,path=opath,basename='od_2_sym_outline',format='vesta',color='b',select='egdes')
od.write(obj=od_sym_2_1,path=opath,basename='od_2_sym_outline',format='xyz',select='egdes')

# intersection of two ODs, OBJ_1 and OBJ_2
od_common=ods.intersection(od_sym_1,od_sym_2,verbose=0)
od.write(obj=od_common,path=opath,basename='od_common',format='vesta',color='g',select='normal')
od.write(obj=od_common,path=opath,basename='od_common',format='xyz')
# Outline of common part
od_common_1=od.outline(od_common)
od.write(obj=od_common_1,path=opath,basename='od_common_outline',format='vesta',color='g',select='egdes')
od.write(obj=od_common_1,path=opath,basename='od_common_outline',format='xyz',select='egdes')
