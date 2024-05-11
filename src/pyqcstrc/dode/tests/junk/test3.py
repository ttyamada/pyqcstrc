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
v2=np.array([[-1, 0, 1],[ 1, 0, 1],[-2, 2, 1],[ 2,-2, 1],[ 0, 0, 1],[ 0, 0, 1]])
#vectors=np.vstack([v1]).reshape(1,6,3)
vectors=np.vstack([v0,v1,v2]).reshape(3,6,3)
centre=v0
vsym=symmetry2.generator_obj_symmetric_vec(vectors,centre)
for i in range(len(vsym)):
    vi=vsym[i]
    for j in range(len(vi)):
        vj=vi[j]
        v1i,v2i,v3i=math2.projection3(vj[0],vj[1],vj[2],vj[3],vj[4],vj[5])
        print('%8.6f %8.6f %8.6f'%(((v1i[0]+v1i[1]*SIN)/v1i[2],(v2i[0]+v2i[1]*SIN)/v2i[2],(v3i[0]+v3i[1]*SIN)/v3i[2])))