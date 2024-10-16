#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

#import timeit
#import os
import sys
import numpy as np
import pyqcstrc.dode2.occupation_domain as od

opath='./test3'
xyzpath='../../../xyz/dode'

# import asymmetric part of OD(occupation domain) located at origin, 0,0,0,0,0,0.
od_asym=od.read_xyz(path=xyzpath,basename='od_1_asym')

# make symmetric OD
pos0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
od_sym_1=od.symmetric(obj=od_asym, centre=pos0)
od.write(obj=od_sym_1, path=opath, basename='od_1_sym', format='vesta', color='k',select='normal')
od.write(obj=od_sym_1, path=opath, basename='od_1_sym', format='xyz')
# Outline of OBJ_1
od_sym_1_1=od.outline(od_sym_1)
od.write(obj=od_sym_1_1,path=opath,basename='od_1_sym_outline',format='vesta',color='r',select='egdes')
od.write(obj=od_sym_1_1,path=opath,basename='od_1_sym_outline',format='xyz')

# Similarity transformation
od_asym_2=od.similarity(obj=od_asym, m=-2)
od_sym_2=od.symmetric(obj=od_asym_2, centre=pos0)
od.write(obj=od_sym_2, path=opath, basename='od_2_sym', format='vesta', color='k')
od.write(obj=od_sym_2, path=opath, basename='od_2_sym', format='xyz')
# Outline of OBJ_2
od_sym_2_1=od.outline(od_sym_2)
od.write(obj=od_sym_2_1,path=opath,basename='od_2_sym_outline',format='vesta',color='b',select='egdes')
od.write(obj=od_sym_2_1,path=opath,basename='od_2_sym_outline',format='xyz')

# Similarity transformation
od_asym_3=od.similarity(obj=od_asym, m=2)
od_sym_3=od.symmetric(obj=od_asym_3, centre=pos0)
od.write(obj=od_sym_3, path=opath, basename='od_3_sym', format='vesta', color='k')
od.write(obj=od_sym_3, path=opath, basename='od_3_sym', format='xyz')
# Outline of OBJ_3
od_sym_3_1=od.outline(od_sym_3)
od.write(obj=od_sym_3_1,path=opath,basename='od_3_sym_outline',format='vesta',color='g',select='egdes')
od.write(obj=od_sym_3_1,path=opath,basename='od_3_sym_outline',format='xyz')

