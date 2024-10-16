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

opath='./work1'
pg = '-12m2'

# Three 6D vectors which define the asymmetric part of the occupation domain of dodecagonal tiling.
# Note that 5-th and 6-th components of each 6D vectors are dummy, and they correspond to Z coordinate in Epar and Eperp, respectively.
v0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v1=np.array([[-1, 0, 2],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v2=np.array([[ 0, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
od_asym=np.vstack([v0,v1,v2]).reshape(1,3,6,3)
od.write(obj=od_asym, path=opath, basename='od_1_asym', format='xyz')

# make symmetric OD
pos0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]) # 0,0,0,0,0
od_sym_1=od.symmetric(obj=od_asym, centre=pos0, pg=pg)
od.write(obj=od_sym_1, path=opath, basename='od_1_sym', format='vesta', color='r',select='normal')
od.write(obj=od_sym_1, path=opath, basename='od_1_sym', format='xyz')
#print('od_sym_1:',od_sym_1.shape)

# Outline of OBJ_1
od_sym_1_1=od.outline(od_sym_1)
od.write(obj=od_sym_1_1,path=opath,basename='od_1_sym_outline',format='vesta',color='r',select='egdes')
od.write(obj=od_sym_1_1,path=opath,basename='od_1_sym_outline',format='xyz',select='egdes')

# make smaller OD by similarity order of 2
pos0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]) # 0,0,0,0,0
od_sym_2=od.similarity(obj=od_sym_1, m=2)
od.write(obj=od_sym_2, path=opath, basename='od_2_sym', format='xyz')
# Outline of OBJ_2
od_sym_2_1=od.outline(od_sym_2)
od.write(obj=od_sym_2_1,path=opath,basename='od_2_sym_outline',format='vesta',color='r',select='egdes')
od.write(obj=od_sym_2_1,path=opath,basename='od_2_sym_outline',format='xyz',select='egdes')

# shift OD2 along (0,1,0,0)
od_sym_3=od.shift(obj=od_sym_2,shift=v2)
od.write(obj=od_sym_3, path=opath, basename='od_3_sym', format='vesta', color='r',select='normal')
od.write(obj=od_sym_3, path=opath, basename='od_3_sym', format='xyz')
# Outline of OBJ_3
od_sym_3_1=od.outline(od_sym_3)
od.write(obj=od_sym_3_1,path=opath,basename='od_3_sym_outline',format='vesta',color='r',select='egdes')
od.write(obj=od_sym_3_1,path=opath,basename='od_3_sym_outline',format='xyz',select='egdes')

v3=np.array([[ 0, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
od_sym_4=od.shift(obj=od_sym_2,shift=v3)
od.write(obj=od_sym_4, path=opath, basename='od_4_sym', format='vesta', color='r',select='normal')
od.write(obj=od_sym_4, path=opath, basename='od_4_sym', format='xyz')
# Outline of OBJ_4
od_sym_4_1=od.outline(od_sym_4)
od.write(obj=od_sym_4_1,path=opath,basename='od_4_sym_outline',format='vesta',color='r',select='egdes')
od.write(obj=od_sym_4_1,path=opath,basename='od_4_sym_outline',format='xyz',select='egdes')

#Xx 0.250000 -0.066987 0.000000 #   4-the triangle 1-th vertex # 0 0 1 0 0 1 -1 0 2 -1 0 2 0 0 1 0 0 1

v4=np.array([[ 0, 0, 1],[ 0, 0, 1],[-1, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
od_sym_5=od.shift(obj=od_sym_2,shift=v4)
od.write(obj=od_sym_5, path=opath, basename='od_5_sym', format='vesta', color='r',select='normal')
od.write(obj=od_sym_5, path=opath, basename='od_5_sym', format='xyz')
# Outline of OBJ_5
od_sym_5_1=od.outline(od_sym_5)
od.write(obj=od_sym_5_1,path=opath,basename='od_5_sym_outline',format='vesta',color='r',select='egdes')
od.write(obj=od_sym_5_1,path=opath,basename='od_5_sym_outline',format='xyz',select='egdes')
