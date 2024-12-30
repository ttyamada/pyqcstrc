#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import os
import sys
import numpy as np
import pyqcstrc.dode2.occupation_domain as od
import pyqcstrc.dode2.symmetry as sym
import pyqcstrc.dode2.numericalc as numericalc

from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

#SIN = np.sqrt(3)/2.0

#===========================
# Drawing Occupation Domain
#===========================

myName = 'local_env_vertex'
xyzpath='../../../xyz/dode'
opath='./test5'

# import asymmetric part of OD(occupation domain), which correspnd to Fig.2 a--e.
od_1_asym=od.read_xyz(path=xyzpath,basename='OD_local_env_1')
od_2_asym=od.read_xyz(path=xyzpath,basename='OD_local_env_2')
od_3_asym=od.read_xyz(path=xyzpath,basename='OD_local_env_3')
od_4_asym=od.read_xyz(path=xyzpath,basename='OD_local_env_4')

V_a=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # 1a, (0,0,0,0)

plt.figure(figsize=(8, 8))
#plt.axis('equal')
xylim=1.2
plt.xlim([-xylim, xylim])
plt.ylim([-xylim, xylim])


# local env no1
od_sym=sym.generator_obj_symmetric_obj(od_1_asym,V_a)
for triangle in od_sym:
    x=[]
    y=[]
    for vt in triangle:
        xy=numericalc.get_internal_component_numerical(vt)
        x.append(float(xy[0]))
        y.append(float(xy[1]))
    #plt.fill(x,y,'#000000') # black
    plt.fill(x,y,'#00008B') # darkblue
    plt.fill(x,y,'b') # blue

# local env no2
od_sym=sym.generator_obj_symmetric_obj(od_2_asym,V_a)
for triangle in od_sym:
    x=[]
    y=[]
    for vt in triangle:
        xy=numericalc.get_internal_component_numerical(vt)
        x.append(float(xy[0]))
        y.append(float(xy[1]))
    #plt.fill(x,y,'#808080') # gray
    plt.fill(x,y,'#0000FF') # blue
    plt.fill(x,y,'pink') # pink

# local env no3
od_sym=sym.generator_obj_symmetric_obj(od_3_asym,V_a)
for triangle in od_sym:
    x=[]
    y=[]
    for vt in triangle:
        xy=numericalc.get_internal_component_numerical(vt)
        x.append(float(xy[0]))
        y.append(float(xy[1]))
    #plt.fill(x,y,'#D3D3D3') # lightgrey
    plt.fill(x,y,'#4169E1') # royalblue
    plt.fill(x,y,'r') #red
# local env no4
od_sym=sym.generator_obj_symmetric_obj(od_4_asym,V_a)
for triangle in od_sym:
    x=[]
    y=[]
    for vt in triangle:
        xy=numericalc.get_internal_component_numerical(vt)
        x.append(float(xy[0]))
        y.append(float(xy[1]))
    #plt.fill(x,y,'#F5F5F5') # whitesmoke
    plt.fill(x,y,'#ADD8E6') # lightblue
    plt.fill(x,y,'lime') # lime


plt.savefig('%s/%s_OD.png'%(opath,myName), format="png", dpi=300, transparent=True)
