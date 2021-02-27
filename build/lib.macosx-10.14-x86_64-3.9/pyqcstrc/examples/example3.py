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

# Vertices of tetrahedron, w0,w1,w2,w3, which
# defines the asymmetric part of OD
w0 =  np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
w1 = np.array([[ 3, 0, 2],[-3, 0, 2],[-3, 0, 2],[-3, 0, 2],[-3, 0, 2],[-3, 0, 2]])
w2 = np.array([[ 0, 0, 1],[-3, 0, 4],[-3, 0, 4],[ 3, 0, 2],[-3, 0, 2],[ 3, 0, 2]])
w3 = np.array([[ 0, 0, 1],[ 3, 0, 4],[-3, 0, 2],[ 3, 0, 2],[-3, 0, 2],[ 3, 0, 4]])
seed = np.vstack([w0,w1,w2,w3]).reshape(4,6,3)
aum5 = od.as_it_is(seed)
posEC = np.array([[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
aum5 = od.shift(aum5,posEC)  # move to (1,0,0,0,0,0)/2
od.write(aum5, path='./test3', basename='aum5', format='vesta', color='r')
od.write(aum5, path='./test3', basename='aum5', format='xyz')

# Vertices of tetrahedron, v0,v1,v2,v3, which
# defines the asymmetric part of OD
v0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v1 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
v2 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
v3 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 0, 0, 2],[-1, 0, 2],[ 0, 0, 2]])
seed = np.vstack([v0,v1,v2,v3]).reshape(4,6,3)
rtod0 = od.as_it_is(seed)
od.write(rtod0, path='./test3', basename='rtod0', format='vesta', color='r')
od.write(rtod0, path='./test3', basename='rtod0', format='xyz')

# generate symmetric OD, symmetric centre is v0.
rtod1 = od.symmetric(rtod0, v0)
od.write(rtod1, path='./test3', basename='rtod1', format='vesta', color='r')
od.write(rtod1, path='./test3', basename='rtod1', format='xyz')

# coordinate of position_1
pos1 = np.array([[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],
                 [ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
rtod2 = od.shift(rtod1,pos1)  # move to position_1
od.write(rtod2, path='./test3', basename='rtod2', format='vesta', color='p')
od.write(rtod2, path='./test3', basename='rtod2', format='xyz')

rtod1_asym = ods.intersection(rtod1, aum5, verbose=1)
od.write(rtod1_asym, path='./test3', basename='rtod1_asym', format='vesta', color='b')
od.write(rtod1_asym, path='./test3', basename='rtod1_asym', format='xyz')

common_od = ods.intersection(rtod1_asym, rtod2, verbose=1)
od.write(common_od, path='./test3', basename='common_od_asym', format='vesta', color='r')
od.write(common_od, path='./test3', basename='common_od_asym', format='xyz')

#rtod1 = od.read_xyz(path='./test2', basename='rtod1')
