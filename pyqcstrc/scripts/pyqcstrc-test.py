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
seed = np.vstack([v0,v1,v2,v3]).reshape(4,6,3)
rtod0 = od.as_it_is(seed)

print('\n/////// Speed check /////////\n')

# coordinate of position_1
pos1 = np.array([[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],
                 [ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])

loop = 1000
print('\nTEST: utils.middle_position()')
result1 = timeit.timeit('utils.middle_position(v0,pos1)',globals=globals(), number=loop)
print(result1 / loop)

loop = 1
# generate symmetric OD, symmetric centre is v0.
print('\nTEST: od.symmetric()')
result1 = timeit.timeit('od.symmetric(rtod0, v0)', globals=globals(), number=loop)
print(result1 / loop)
rtod1 = od.symmetric(rtod0, v0)
rtod2 = rtod1.reshape((len(rtod1)*4,6,3))

loop = 10

print('\nTEST: utils.remove_doubling_dim4_in_perp_space()')
result1 = timeit.timeit('utils.remove_doubling_dim4_in_perp_space(rtod1)', globals=globals(), number=loop)
print(result1 / loop)

print('\nTEST: utils.remove_doubling_dim3_in_perp_space()')
result1 = timeit.timeit('utils.remove_doubling_dim3_in_perp_space(rtod2)', globals=globals(), number=loop)
print(result1 / loop)

print('\nTEST: utils.remove_doubling_dim3()')
result1 = timeit.timeit('utils.remove_doubling_dim3(rtod2)', globals=globals(), number=loop)
print(result1 / loop)

loop = 1
"""
print('\nTEST: utils.generator_surface_1()')
result1 = timeit.timeit('utils.generator_surface_1(rtod1,1)', globals=globals(), number=loop)
print(result1 / loop)
surface=utils.generator_surface_1(rtod1,0)

print('\nTEST: utils.generator_edge()')
result1 = timeit.timeit('utils.generator_edge(surface,1)', globals=globals(), number=loop)
print(result1 / loop)
"""

print('\nTEST: od.shift()')
result1 = timeit.timeit('od.shift(rtod1,pos1)', globals=globals(), number=loop)
print(result1 / loop)
rtod2 = od.shift(rtod1,pos1)  # move to position_1

#print('\nTEST: ods.intersection_convex()')
#result1 = timeit.timeit('ods.intersection_convex(rtod1, rtod2, verbose=0)', globals=globals(), number=loop)
#print(result1 / loop)

#rtod1=rtod1[10].reshape(1,4,6,3)

print('\nTEST: ods.intersection()')
result1 = timeit.timeit('ods.intersection(rtod1, rtod2, verbose=1)', globals=globals(), number=loop)
print(result1 / loop)


