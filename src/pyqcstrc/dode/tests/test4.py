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
asym_obj=np.vstack([v0,v1,v2]).reshape(1,3,6,3)
obj1=od.symmetric(asym_obj,v0)

od.write(obj=obj1,path='./test4',basename='obj1',format='vesta',color='r',dmax=10.0)
od.write(obj=obj1,path='./test4',basename='obj1',format='xyz')

objs=[obj1]
pos=[v0]
atm=['Xx']

od.qcstrc(obj=objs,positions=pos, path='./test4',basename='atm',atm=atm, phason_matrix=np.array([[[0]]]), nmax = 5, verbose = 1)

