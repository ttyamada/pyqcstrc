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

# Vertices of tetrahedron, v0,v1,v2,v3, which
# defines the asymmetric part.
v0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],
                [ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
pos1 = np.array([[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],
                [ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])

loop = 1000
print('\nTEST: utils.middle_position()')
result1 = timeit.timeit('utils.middle_position(v0,pos1)', globals=globals(), number=loop)
print(result1 / loop)

rtod1 = od.read_xyz(path='./test1', basename='rtod1')
rtod2=rtod1.reshape((len(rtod1)*4,6,3))

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
print('\nTEST: utils.generator_surface_1()')
result1 = timeit.timeit('utils.generator_surface_1(rtod1,1)', globals=globals(), number=loop)
print(result1 / loop)

surface=utils.generator_surface_1(rtod1,0)
print('\nTEST: utils.generator_edge()')
result1 = timeit.timeit('utils.generator_edge(surface,1)', globals=globals(), number=loop)
print(result1 / loop)
