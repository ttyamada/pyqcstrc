#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import os
import sys
import numpy as np
sys.path.append('..')
import math2
import symmetry2

SIN=np.sqrt(3)/2.0

v0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,0,0,0)
v1=np.array([[-1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (-1,1,0,0)
v2=np.array([[-1, 0, 1],[ 1, 0, 1],[-1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (-1,1,-1,1)
v3=np.array([[-1, 0, 1],[ 1, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (-1,1,-1,0)
v4=np.array([[-1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (-1,0,0,-1)

# Line segments
seg1=np.vstack([v1,v2]).reshape(2,6,3) # v1-v2
seg2=np.vstack([v3,v4]).reshape(2,6,3) # v3-v4
p=intersection_two_segments(seg1,seg2) # intersection
print('intersecting point:')
print('[[%d,%d,%d],[%d,%d,%d],[%d,%d,%d],[%d,%d,%d],[%d,%d,%d],[%d,%d,%d]]'%(\
            p[0][0],p[0][1],p[0][2],\
            p[1][0],p[1][1],p[1][2],\
            p[2][0],p[2][1],p[2][2],\
            p[3][0],p[3][1],p[3][2],\
            p[4][0],p[4][1],p[4][2],\
            p[5][0],p[5][1],p[5][2]))
v1i,v2i,v3i=math2.projection3(p[0],p[1],p[2],p[3],p[4],p[5])
print('%8.6f %8.6f %8.6f'%(((v1i[0]+v1i[1]*SIN)/v1i[2],(v2i[0]+v2i[1]*SIN)/v2i[2],(v3i[0]+v3i[1]*SIN)/v3i[2])))

for di in [v1,v2,v3,v4]:
    v1i,v2i,v3i=math2.projection3(di[0],di[1],di[2],di[3],di[4],di[5])
    print('%8.6f %8.6f %8.6f'%(((v1i[0]+v1i[1]*SIN)/v1i[2],(v2i[0]+v2i[1]*SIN)/v2i[2],(v3i[0]+v3i[1]*SIN)/v3i[2])))
