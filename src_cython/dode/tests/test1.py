#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import os
import sys
import numpy as np
#sys.path.append('..')
import pyqcstrc.dode.occupation_domain as od

SIN=np.sqrt(3)/2.0

# Three 6D vectors which define the asymmetric part of the occupation domain for Nizeki-Gahler dodecagonal tiling
# Note that 5-th and 6-th components of each 6D vectors are dummy, and they correspond to Z coordinate in Epar and Eperp, respectively.
v0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,0,0,0)
v1=np.array([[ 0,-2, 3],[ 0, 2, 3],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # new v1
v2=np.array([[ 0,-2, 3],[ 0, 2, 3],[ 3,-4, 3],[-3, 4, 3],[ 0, 0, 1],[ 0, 0, 1]]) # new v2
asym_obj=np.vstack([v0,v1,v2]).reshape(3,6,3)

# OBJ_1: occupation domain for Nizeki-Gahler dodecagonal tiling
obj1=od.symmetric(asym_obj,v0)
od.write(obj=obj1,path='./test1',basename='obj1',format='vesta',color='r',dmax=10.0)
od.write(obj=obj1,path='./test1',basename='obj1',format='xyz')
# Outline of OBJ_1
obj1_1=od.outline1(obj1)
od.write(obj=obj1_1,path='./test1',basename='obj1_outline',format='vesta',color='r',dmax=10.0)
od.write(obj=obj1_1,path='./test1',basename='obj1_outline',format='xyz')

# OBJ_2
# Shift vector of OBJ2
pos=np.array([[ 0, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # ( 0,1,0,0)
obj2=od.shift(obj1,pos)
od.write(obj=obj2,path='./test1',basename='obj2',format='vesta',color='b',dmax=10.0)
od.write(obj=obj2,path='./test1',basename='obj2',format='xyz')
# Outline of OBJ_2
obj2_1=od.outline1(obj2)
od.write(obj=obj2_1,path='./test1',basename='obj2_outline',format='vesta',color='b',dmax=10.0)
od.write(obj=obj2_1,path='./test1',basename='obj2_outline',format='xyz')

# Intersection of OBJ1 and OBJ2
obj3=od.intersection(obj1,obj2,verbose=0)
od.write(obj=obj3,path='./test1',basename='obj3',format='vesta',color='k',dmax=10.0)
od.write(obj=obj3,path='./test1',basename='obj3',format='xyz')

print('outline of obj3:')
#a=od.outline(obj3)
# Outline of OBJ3
obj3_1=od.outline1(obj3)
od.write(obj=obj3_1,path='./test1',basename='obj3_outline',format='vesta',color='k',dmax=10.0)
od.write(obj=obj3_1,path='./test1',basename='obj3_outline',format='xyz')

print('triangulation of obj3:')
"""
pos1=np.array([[ 0, 0, 1],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # ( 0,1/2,0,0)
obj4=od.triangulation(obj3,pos1)
od.write(obj=obj4,path='./test1',basename='obj4',format='vesta',color='k',dmax=10.0)
od.write(obj=obj4,path='./test1',basename='obj4',format='xyz')

print(obj4.ndim)
w1=np.array([[ 0, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,1,0,0)
w2=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,0,1,0)
vec_au=np.vstack([w1,w2]).reshape(2,6,3)
obj4_asym=od.asymmetric(symmetric_obj=obj4, position=pos1, vecs=vec_au)
od.write(obj=obj4_asym,path='./test1',basename='obj4_asym',format='vesta',color='k',dmax=10.0)
"""
