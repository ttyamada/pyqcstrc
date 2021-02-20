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
import pyqcstrc.icosah.utils as utils
import pyqcstrc.icosah.occupation_domain as od
import pyqcstrc.icosah.two_occupation_domains as ods

# Vertices of tetrahedron, v0,v1,v2,v3, which
# defines the asymmetric part.
v0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v1 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
v2 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
v3 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 0, 0, 2],[-1, 0, 2],[ 0, 0, 2]])
tmp = np.append(v0,v1)
tmp = np.append(tmp,v2)
tmp = np.append(tmp,v3)
seed = tmp.reshape(4,6,3)
rtod0 = od.as_it_is(seed)
od.write(rtod0, path='./test1', basename='rtod0', format='vesta', color='r')
od.write(rtod0, path='./test1', basename='rtod0', format='xyz')

# generate symmetric OD, symmetric centre is v0.
rtod1 = od.symmetric(rtod0, v0)
od.write(rtod1, path='./test1', basename='rtod1', format='vesta', color='r')
od.write(rtod1, path='./test1', basename='rtod1', format='xyz')

# coordinate of position_1
pos1 = np.array([[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],
                 [ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
rtod2 = od.shift(rtod1,pos1)  # move to position_1
od.write(rtod2, path='./test1', basename='rtod2', format='vesta', color='p')
od.write(rtod2, path='./test1', basename='rtod2', format='xyz')

common_od = ods.intersection_convex(rtod1, rtod2, verbose=0)

# intermediate position between v0 and pos1
pos2 = utils.middle_position(v0,pos1)
common_od = od.simpl_add_point(common_od, pos2, verbose=0)
od.write(common_od, path='./test1', basename='common_od', format='vesta', color='b')
od.write(common_od, path='./test1', basename='common_od', format='xyz')

#rtod1 = od.read_xyz(path='./test1', basename='rtod1')
