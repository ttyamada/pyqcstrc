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

#------------------------------------------------------
# Akiji Yamamoto, 5D model of Ta-Te quasicrystals
# Acta Cryst. (2004). A60, 142±145
#------------------------------------------------------

# OD A 

opath='./work2_OD_A'
pos0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]) # 0,0,0,0,0

#0 0 1 1 0 2 1 0 2 1 0 2 0 0 1 0 0 1
#2 0 1 2 0 1 1 0 1 0 0 1 0 0 1 0 0 1
# OD A-1
v0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v1=np.array([[ 0, 0, 1],[ 1, 0, 2],[ 1, 0, 2],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1]])
v2=np.array([[ 2, 0, 1],[ 2, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
od_a_1_asym=np.vstack([v0,v1,v2]).reshape(1,3,6,3)
od.write(obj=od_a_1_asym, path=opath, basename='od_a_1_asym', format='vesta', color='r',select='normal')
od.write(obj=od_a_1_asym, path=opath, basename='od_a_1_asym', format='xyz')
#
# make symmetric OD
od_sym=od.symmetric(obj=od_a_1_asym, centre=pos0)
od.write(obj=od_sym, path=opath, basename='od_a_1_sym', format='vesta', color='r',select='normal')
od.write(obj=od_sym, path=opath, basename='od_a_1_sym', format='xyz')

# OD A-2
v0=np.array([[ 0, 0, 1],[ 1, 0, 2],[ 1, 0, 2],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1]])
v1=np.array([[-1, 0, 1],[-2, 0, 1],[-3, 0, 1],[-3, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v2=np.array([[ 2, 0, 1],[ 2, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
od_asym=np.vstack([v0,v1,v2]).reshape(1,3,6,3)
od.write(obj=od_asym, path=opath, basename='od_a_2_asym', format='vesta', color='b',select='normal')
od.write(obj=od_asym, path=opath, basename='od_a_2_asym', format='xyz')
#
# make symmetric OD
od_sym=od.symmetric(obj=od_asym, centre=pos0)
od.write(obj=od_sym, path=opath, basename='od_a_2_sym', format='vesta', color='b',select='normal')
od.write(obj=od_sym, path=opath, basename='od_a_2_sym', format='xyz')

# OD A-3
v0=np.array([[ 0, 0, 1],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v1=np.array([[-2, 0, 1],[-1, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v2=np.array([[ 1, 0, 1],[ 0, 0, 1],[-2, 0, 1],[-3, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v3=np.array([[ 0, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v4=np.array([[-1, 0, 1],[-2, 0, 1],[-3, 0, 1],[-3, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v5=np.array([[ 2, 0, 1],[ 2, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
od_asym=np.vstack([v0,v1,v2,v0,v2,v3,v0,v3,v4,v0,v4,v5]).reshape(4,3,6,3)
od.write(obj=od_asym, path=opath, basename='od_a_3_asym', format='vesta', color='y',select='normal')
od.write(obj=od_asym, path=opath, basename='od_a_3_asym', format='xyz')
#
# make symmetric OD
od_sym=od.symmetric(obj=od_asym, centre=pos0)
od.write(obj=od_sym, path=opath, basename='od_a_3_sym', format='vesta', color='y',select='normal')
od.write(obj=od_sym, path=opath, basename='od_a_3_sym', format='xyz')

# OD A-4
v0=np.array([[-1, 0, 2],[ 0, 0, 1],[-1, 0, 2],[-1, 0, 2],[ 0, 0, 1],[ 0, 0, 1]])
v1=np.array([[ 0, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v2=np.array([[ 1, 0, 1],[ 0, 0, 1],[-2, 0, 1],[-3, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v3=np.array([[-2, 0, 1],[-1, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v4=np.array([[ 2, 0, 1],[ 3, 0, 1],[ 2, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v5=np.array([[-1, 0, 1],[-1, 0, 1],[-2, 0, 1],[-2, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
od_asym=np.vstack([v0,v1,v2,v0,v2,v3,v0,v3,v4,v0,v4,v5]).reshape(4,3,6,3)
od.write(obj=od_asym, path=opath, basename='od_a_4_asym', format='vesta', color='k',select='normal')
od.write(obj=od_asym, path=opath, basename='od_a_4_asym', format='xyz')
#
# make symmetric OD
od_sym=od.symmetric(obj=od_asym, centre=pos0)
od.write(obj=od_sym, path=opath, basename='od_a_4_sym', format='vesta', color='k',select='normal')
od.write(obj=od_sym, path=opath, basename='od_a_4_sym', format='xyz')


# OD A-5
v0=np.array([[ 0, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v1=np.array([[ 2, 0, 1],[ 3, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v2=np.array([[ 0, 0, 1],[ 3, 0, 2],[ 1, 0, 2],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1]])
v3=np.array([[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 0, 0, 1],[ 0, 0, 1]])
v4=np.array([[ 1, 0, 1],[ 1, 0, 1],[-1, 0, 1],[-2, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v5=np.array([[ 1, 0, 2],[ 3, 0, 2],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v6=np.array([[-1, 0, 2],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v7=np.array([[-1, 0, 1],[-1, 0, 1],[-2, 0, 1],[-2, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v8=np.array([[ 1, 0, 2],[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 2],[ 0, 0, 1],[ 0, 0, 1]])
v9=np.array([[ 0, 0, 1],[ 1, 0, 1],[ 1, 0, 2],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1]])
v10=np.array([[-2, 0, 1],[-1, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
od_asym=np.vstack([v0,v1,v2,v0,v3,v4,v0,v4,v5,v0,v6,v7,v0,v7,v8,v0,v9,v10]).reshape(6,3,6,3)
od.write(obj=od_asym, path=opath, basename='od_a_5_asym', format='vesta', color='r',select='normal')
od.write(obj=od_asym, path=opath, basename='od_a_5_asym', format='xyz')
#
# make symmetric OD
od_sym=od.symmetric(obj=od_asym, centre=pos0)
od.write(obj=od_sym, path=opath, basename='od_a_5_sym', format='vesta', color='r',select='normal')
od.write(obj=od_sym, path=opath, basename='od_a_5_sym', format='xyz')



# OD A-6
v0=np.array([[ 0,-2, 3],[ 0, 2, 3],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v1=np.array([[ 1, 0, 2],[ 3, 0, 2],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v2=np.array([[ 0, 0, 1],[ 2, 0, 1],[ 2, 0, 1],[ 2, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
od_asym=np.vstack([v0,v1,v2]).reshape(1,3,6,3)
od.write(obj=od_asym, path=opath, basename='od_a_6_asym', format='vesta', color='r',select='normal')
od.write(obj=od_asym, path=opath, basename='od_a_6_asym', format='xyz')
#
# make symmetric OD
od_sym=od.symmetric(obj=od_asym, centre=pos0)
od.write(obj=od_sym, path=opath, basename='od_a_6_sym', format='vesta', color='r',select='normal')
od.write(obj=od_sym, path=opath, basename='od_a_6_sym', format='xyz')

# Merge, asymmetric units of ODs1,2,3
# --> od_a_asym.xyz
