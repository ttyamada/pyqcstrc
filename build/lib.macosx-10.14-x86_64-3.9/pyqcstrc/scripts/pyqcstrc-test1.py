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
import pyqcstrc.icosah.math1 as math1
import pyqcstrc.icosah.numericalc as numericalc
import pyqcstrc.icosah.occupation_domain as od
import pyqcstrc.icosah.two_occupation_domains as ods

TAU=(1+np.sqrt(5))/2

# Vertices of tetrahedron, v0,v1,v2,v3, which
# defines the asymmetric part.
v0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v1 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
v2 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
v3 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 0, 0, 2],[-1, 0, 2],[ 0, 0, 2]])
seed = np.vstack([v0,v1,v2,v3]).reshape(4,6,3)
rtod0 = od.as_it_is(seed)

print('\n/////// Speed check /////////\n')


##################  projection ####################
loop=10000

v4 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[ 1, 0, 2]])

a1=(v4[0][0]+v4[0][1]*TAU)/v4[0][2]
a2=(v4[1][0]+v4[1][1]*TAU)/v4[1][2]
a3=(v4[2][0]+v4[2][1]*TAU)/v4[2][2]
a4=(v4[3][0]+v4[3][1]*TAU)/v4[3][2]
a5=(v4[4][0]+v4[4][1]*TAU)/v4[4][2]
a6=(v4[5][0]+v4[5][1]*TAU)/v4[5][2]

print('\nTEST: numericalc.projection3_numerical()')
result1 = timeit.timeit('numericalc.projection3_numerical(a1,a2,a3,a4,a5,a6)', globals=globals(), number=loop)
print(result1 / loop)
b1=numericalc.projection3_numerical(a1,a2,a3,a4,a5,a6)
print('%6.4f %6.4f %6.4f'%(b1[0],b1[1],b1[2]))

print('\nTEST: numericalc.projection_numerical()')
result1 = timeit.timeit('numericalc.projection_numerical(a1,a2,a3,a4,a5,a6)', globals=globals(), number=loop)
print(result1 / loop)
b1=numericalc.projection_numerical(a1,a2,a3,a4,a5,a6)
print('%6.4f %6.4f %6.4f'%(b1[3],b1[4],b1[5]))

print('\nTEST: math1.projection3()')
result1 = timeit.timeit('math1.projection3(v1[0],v1[1],v1[2],v1[3],v1[4],v1[5])', globals=globals(), number=loop)
print(result1 / loop)
b1,b2,b3=math1.projection3(v4[0],v4[1],v4[2],v4[3],v4[4],v4[5])
print('%6.4f %6.4f %6.4f'%((b1[0]+b1[1]*TAU)/b1[2],(b2[0]+b2[1]*TAU)/b2[2],(b3[0]+b3[1]*TAU)/b3[2]))

print('\nTEST: math1.projection()')
result1 = timeit.timeit('math1.projection(v1[0],v1[1],v1[2],v1[3],v1[4],v1[5])', globals=globals(), number=loop)
print(result1 / loop)
_,_,_,b1,b2,b3=math1.projection(v4[0],v4[1],v4[2],v4[3],v4[4],v4[5])
print('%6.4f %6.4f %6.4f'%((b1[0]+b1[1]*TAU)/b1[2],(b2[0]+b2[1]*TAU)/b2[2],(b3[0]+b3[1]*TAU)/b3[2]))
############################################





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

print('\nTEST: od.shift()')
result1 = timeit.timeit('od.shift(rtod1,pos1)', globals=globals(), number=loop)
print(result1 / loop)
rtod2 = od.shift(rtod1,pos1)  # move to position_1


################  intersection #################

#rtod1=rtod1[10].reshape(1,4,6,3)

print('\nTEST: ods.intersection()')
result1 = timeit.timeit('ods.intersection(rtod1, rtod2, verbose=1)', globals=globals(), number=loop)
print(result1 / loop)

print('\nTEST: ods.intersection_convex()')
result1 = timeit.timeit('ods.intersection_convex(rtod1, rtod2, verbose=1)', globals=globals(), number=loop)
print(result1 / loop)

print('\nTEST: ods.intersection()')
result1 = timeit.timeit('ods.intersection_old(rtod1, rtod2, verbose=1)', globals=globals(), number=loop)
print(result1 / loop)

################################################
"""
################  generator_surface & generator_edge ################
print('\nTEST: utils.generator_surface_1()')
result1 = timeit.timeit('utils.generator_surface_1(rtod1,1)', globals=globals(), number=loop)
print(result1 / loop)
surface=utils.generator_surface_1(rtod1,0)

print('\nTEST: utils.generator_edge()')
result1 = timeit.timeit('utils.generator_edge(surface,1)', globals=globals(), number=loop)
print(result1 / loop)
#####################################################################
"""
