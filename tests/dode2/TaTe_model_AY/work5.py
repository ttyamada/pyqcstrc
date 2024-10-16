#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

#import timeit
#import os
import sys
import numpy as np
import pyqcstrc.dode2.math1 as math1
import pyqcstrc.dode2.occupation_domain as od
import pyqcstrc.dode2.two_occupation_domains as ods

#------------------------------------------------------
# Akiji Yamamoto, 5D model of Ta-Te quasicrystals
# Acta Cryst. (2004). A60, 142±145
#------------------------------------------------------

# Wyckoff positions:
V_1a=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,0,0,0)
V_4a=np.array([[ 0, 0, 1],[ 2, 0, 3],[ 0, 0, 1],[ 1, 0, 3],[ 0, 0, 1],[ 0, 0, 1]]) # (0,2/3,0,1/3)
V_6a=np.array([[ 0, 0, 1],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,1/2,0,0)
V_6b=np.array([[ 0, 0, 1],[ 1, 0, 2],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,1/2,1/2,0)

# OD D

opath='./work5_OD_D'

# define the asymmetric unit
v0=np.array([[ 0,0,1],[ 0,0,1],[ 0,0,1],[ 0,0,1],[0,0,1],[0,0,1]]) # 0,0,0,0,0
v1=np.array([[-5,0,1],[ 0,0,1],[ 0,0,1],[-5,0,1],[0,0,1],[0,0,1]]) # along -1,0,0,-1,0
v2=np.array([[ 0,0,1],[ 0,0,1],[10,0,1],[ 0,0,1],[0,0,1],[0,0,1]]) # along  0,0,1,0,0
tmp=np.vstack([v0,v1,v2]).reshape(1,3,6,3)
asym=od.shift(obj=tmp,shift=V_6b)
#od.write(obj=asym, path=opath, basename='asym', format='vesta', color='r',select='normal')

# OD D-1
tmp=od.read_xyz(path='./work2_OD_A',basename='od_a_4_sym')
#od.write(obj=tmp, path=opath, basename='tmp', format='vesta', color='r',select='normal')
od_common=ods.intersection(tmp,asym)
#v=math1.mul_vector(V_6b,np.array([-1,0,1]))
#od_common=od.shift(obj=od_common,shift=v)
od.write(obj=od_common, path=opath, basename='od_d_1_asym', format='vesta', color='r',select='normal')
od.write(obj=od_common, path=opath, basename='od_d_1_asym', format='xyz')
# make symmetric OD
od_sym=od.symmetric(obj=od_common, centre=V_6b)
od.write(obj=od_sym, path=opath, basename='od_d_1_sym', format='vesta', color='r',select='normal')
od.write(obj=od_sym, path=opath, basename='od_d_1_sym', format='xyz')

# OD D-2
tmp=od.read_xyz(path='./work2_OD_A',basename='od_a_5_sym')
#od.write(obj=tmp, path=opath, basename='tmp', format='vesta', color='r',select='normal')
od_common=ods.intersection(tmp,asym)
#v=math1.mul_vector(V_6b,np.array([-1,0,1]))
#od_common=od.shift(obj=od_common,shift=v)
od.write(obj=od_common, path=opath, basename='od_d_2_asym', format='vesta', color='r',select='normal')
od.write(obj=od_common, path=opath, basename='od_d_2_asym', format='xyz')
# make symmetric OD
od_sym=od.symmetric(obj=od_common, centre=V_6b)
od.write(obj=od_sym, path=opath, basename='od_d_2_sym', format='vesta', color='r',select='normal')
od.write(obj=od_sym, path=opath, basename='od_d_2_sym', format='xyz')

# OD D-3
tmp=od.read_xyz(path='./work2_OD_A',basename='od_a_6_sym')
#od.write(obj=tmp, path=opath, basename='tmp', format='vesta', color='r',select='normal')
od_common=ods.intersection(tmp,asym)
#v=math1.mul_vector(V_6b,np.array([-1,0,1]))
#od_common=od.shift(obj=od_common,shift=v)
od.write(obj=od_common, path=opath, basename='od_d_3_asym', format='vesta', color='r',select='normal')
od.write(obj=od_common, path=opath, basename='od_d_3_asym', format='xyz')
# make symmetric OD
od_sym=od.symmetric(obj=od_common, centre=V_6b)
od.write(obj=od_sym, path=opath, basename='od_d_3_sym', format='vesta', color='r',select='normal')
od.write(obj=od_sym, path=opath, basename='od_d_3_sym', format='xyz')
