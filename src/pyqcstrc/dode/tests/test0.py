#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import os
import sys
import numpy as np
sys.path.append('..')
import occupation_domain as od

import math2

SIN=np.sqrt(3)/2.0

# Three 6D vectors which define the asymmetric part of the occupation domain for Nizeki-Gahler dodecagonal tiling
# Note that 5-th and 6-th components of each 6D vectors are dummy, and they correspond to Z coordinate in Epar and Eperp, respectively.
v0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,0,0,0)
v1=np.array([[-1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (-1,1,0,0)
v2=np.array([[-1, 0, 1],[ 1, 0, 1],[-2, 2, 1],[ 2,-2, 1],[ 0, 0, 1],[ 0, 0, 1]])
v4=np.array([[ 0,-2, 3],[ 0, 2, 3],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # new v1
v3=np.array([[ 0,-2, 3],[ 0, 2, 3],[ 3,-4, 3],[-3, 4, 3],[ 0, 0, 1],[ 0, 0, 1]]) # new v2
a=od.get_perp_component(v2)
b=math2.div(1,0,1,a[0][0],a[0][1],a[0][2])
print(b)

fac=[0, 2, 3]

v2=math2.mul_vector(v2,fac)
print(v2)

v1=math2.mul_vector(v1,fac)
print(v1)




a=od.get_perp_component(v3)
b=math2.div(1,0,1,a[0][0],a[0][1],a[0][2])
print(b)
