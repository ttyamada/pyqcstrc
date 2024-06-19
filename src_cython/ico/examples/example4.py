#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import timeit
import os
import sys
import numpy as np
import pyqcstrc.ico.occupation_domain as od
import pyqcstrc.ico.two_occupation_domains as ods

V0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
BC = np.array([[ 1, 0, 2],[ 1, 0, 2],[ 1, 0, 2],[ 1, 0, 2],[ 1, 0, 2],[ 1, 0, 2]])
EC = np.array([[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])

common_od_asym=od.read_xyz(path='./example3', basename='common_od_asym')
# back to the origin
v4 = np.array([[-1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
common_od_asym=od.shift(common_od_asym,v4)

vtx=od.write_vesta(obj=common_od_asym, path='./example4', basename='common_od_asym', color = 'k', select = 'podatm')
od.write_xyz(obj=vtx, path='./example4', basename='common_od_asym', select='vertex')

# Generate atm and pod files
vlst=[\
[4,2,5,6,1,2,4,1,2,5],\
[9,1,2,5,1,2,3,2,3,12,1,3,7,3,7,11,3,11,12,11,12,14,8,11,14,7,8,11,7,8,13]]

vtx=od.read_xyz(path='./example4', basename='common_od_asym', select ='vertex')
od.write_podatm(obj=vtx, position=EC, vlist=vlst, path='./example4', basename='common_od_asym', verbose=1)

