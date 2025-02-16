#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import os
import sys
import numpy as np
sys.path.append('..')
import occupation_domain as od

SIN=np.sqrt(3)/2.0

v0=np.array([0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1]).reshape(6,3)
#v1=np.array([[-1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (-1,1,0,0)
#v2=np.array([[-1, 0, 1],[ 1, 0, 1],[-2, 2, 1],[ 2,-2, 1],[ 0, 0, 1],[ 0, 0, 1]])
v1=np.array([[ 0,-2, 3],[ 0, 2, 3],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # new v1
v2=np.array([[ 0,-2, 3],[ 0, 2, 3],[ 3,-4, 3],[-3, 4, 3],[ 0, 0, 1],[ 0, 0, 1]]) # new v2

w0=np.array([0,0,1,1,0,1,0,0,1,0,0,1,0,0,1,0,0,1]).reshape(6,3)
w1=np.array([1,0,1,0,0,1,1,0,1,-1,0,1,0,0,1,0,0,1]).reshape(6,3)
w2=np.array([-1,2,1,2,-2,1,1,0,1,-1,0,1,0,0,1,0,0,1]).reshape(6,3)

obj1=np.vstack([v0,v1,v2]).reshape(1,3,6,3)
obj2=np.vstack([w0,w1,w2]).reshape(1,3,6,3)

od.write(obj=obj1,path='./test2',basename='obj1',format='vesta',color='r',dmax=10.0)
od.write(obj=obj1,path='./test2',basename='obj1',format='xyz')

od.write(obj=obj2,path='./test2',basename='obj2',format='vesta',color='b',dmax=10.0)
od.write(obj=obj2,path='./test2',basename='obj2',format='xyz')

# Intersection of OBJ1 and OBJ2
obj3=od.intersection(obj1,obj2,verbose=9)
od.write(obj=obj3,path='./test2',basename='obj3',format='vesta',color='k',dmax=10.0)

# Outline of OBJ3
obj3_1=od.outline1(obj3)
od.write(obj=obj3_1,path='./test2',basename='obj3_1',format='vesta',color='k',dmax=10.0)
