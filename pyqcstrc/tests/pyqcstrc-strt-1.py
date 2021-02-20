#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import timeit
import os
import sys
import numpy as np
import pyqcstrc.icosah.occupation_domain as od
import pyqcstrc.icosah.two_occupation_domains as ods

POS0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
POS_B1=np.array([[ 1, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 1],[ 0, 0, 1]]) # b_1

path_xyz='../xyz'
path_work='./test1'


# import asymmetric part of STRT OD(occupation domain) located at origin,0,0,0,0,0,0.
strt_asym = od.read_xyz(path=path_xyz,basename='strt_aysmmetric')
od.write(obj=strt_asym, path=path_work, basename = 'obj_seed', format='vesta', color = 'k')

# generat STRT OD located at 0,0,0,0,0,0 by symmetric operations (m-3-5).
strt_sym = od.symmetric(asymmetric_part_obj = strt_asym, position = POS0)

# move STRT OD to a position 1 1 1 0 -1 0.
strt_pos1=od.shift(obj = strt_sym, shift = POS_B1)
od.write(obj=strt_pos1, path=path_work, basename='obj_strt', format='xyz')
od.write(obj=strt_pos1, path=path_work, basename='obj_strt', format='vesta', color='b')

# intersection, obj1 AND obj2
common=ods.intersection(obj1=strt_asym, obj2=strt_pos1, flag=[0,0,1], verbose=1)
# export common part in xyz file
od.write(obj=common, path=path_work, basename='common', format='xyz')
od.write(obj=common, path=path_work, basename='common', format='vesta', color='r')

# subtraction, obj1 NOT obj2
#obj_a=ods.subtraction(obj1=strt_asym, obj2=strt_pos1, obj_common=common, verbose=1)
# export common part in xyz file
#od.write(obj=obj_a, path=path_work, basename='obj_a', format='xyz')
#od.write(obj=obj_a, path=path_work, basename='obj_a', format='vesta', color='r')
