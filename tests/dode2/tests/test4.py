#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

#import timeit
#import os
import sys
import numpy as np
import pyqcstrc.dode2.occupation_domain as od
import pyqcstrc.dode2.symmetry as sym
import pyqcstrc.dode2.numericalc as numericalc

opath='./test4'
xyzpath='../../../xyz/dode'

# Generate a simple decoration model
# see T. Yamada, Acta Cryst. (2022). B78, 247–252

# import asymmetric part of OD(occupation domain), which correspnd to Fig.2 a--e.
od_a_asym=od.read_xyz(path=xyzpath,basename='od_vertex_asymmetric')
od_b_asym=od.read_xyz(path=xyzpath,basename='od_edge_asymmetric')
od_c_asym=od.read_xyz(path=xyzpath,basename='od_triangle_asymmetric')
od_d_asym=od.read_xyz(path=xyzpath,basename='od_square_asymmetric')
od_e_asym=od.read_xyz(path=xyzpath,basename='od_rhombus_asymmetric')

# Wyckoff positions:
V_a=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # 1a, (0,0,0,0)
V_b=np.array([[ 0, 0, 1],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # 6a, (0,1/2,0,0)
V_c=np.array([[ 0, 0, 1],[ 2, 0, 3],[ 0, 0, 1],[ 1, 0, 3],[ 0, 0, 1],[ 0, 0, 1]]) # 4b, (0,2/3,0,1/3)
V_d=np.array([[ 0, 0, 1],[ 1, 0, 2],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # 3a, (0,1/2,1/2,0)
V_e=np.array([[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1]]) # 6b, (1/2,0,0,1/2)

V0=V_a

#-----------
# Structure
#-----------
mystrc=[\
[od_a_asym,V_a,'Xx',V0],\
[od_b_asym,V_b,'Yy',V0],\
[od_c_asym,V_c,'Zz',V0],\
[od_d_asym,V_d,'Ww',V0],\
[od_e_asym,V_e,'Vv',V0],\
]

#---------------------------
print('\nGenerating structure...')
#---------------------------
h5max=0
#h1max=6
#h1max=5
h1max=2
oshift=[0.00178250, 0.00137613, 0.003987675, 0.2387783, 0, 0]
basename='simple_deco'
# Phason matrix for Sigma-phase
u11 = 0
u12 = 0
u21 = 0
u22 = 0
verbose=1

od.qcstrc(mystrc,\
        path=opath,\
        basename='%s_hmax%d'%(basename,h1max),\
        phason_matrix= np.array([\
                        [u11, u12],\
                        [u21, u22],\
                        ]),\
        n1max=h1max,\
        n5max=h5max,\
        origin_shift=oshift,
        option=2,
        verbose=verbose)
