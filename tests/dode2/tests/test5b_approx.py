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

opath='./test5'
xyzpath='../../../xyz/dode'

# import asymmetric part of OD(occupation domain)
od_1_asym=od.read_xyz(path=xyzpath,basename='OD_local_env_1')
od_2_asym=od.read_xyz(path=xyzpath,basename='OD_local_env_2')
od_3_asym=od.read_xyz(path=xyzpath,basename='OD_local_env_3')
od_4_asym=od.read_xyz(path=xyzpath,basename='OD_local_env_4')

# Wyckoff positions:
V_a=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # 1a, (0,0,0,0)

V0=V_a

#-----------
# Structure
#-----------
mystrc=[\
[od_1_asym,V_a,'Aa',V0],\
[od_2_asym,V_a,'Bb',V0],\
[od_3_asym,V_a,'Cc',V0],\
[od_4_asym,V_a,'Dd',V0],\
]

#---------------------------
print('\nGenerating structure...')
#---------------------------
h1max=5
#h1max=3
h5max=0
#oshift=[0.00178250, 0.00137613, 0.003987675, 0.2387783, 0, 0]
verbose=1

#"""
####################
#   Phase 1
####################
myName='local_env_vertex_phase1'
# Phason matrix for Phase 1
u11 = -0.0717967697244909
u12 =  0
u21 =  0
u22 = -0.0717967697244909
# 4D point cut throught the 2D external space.
m01,m02,m03,m04 = 0, 0, 0, 0.5                                   # for phase 1
#"""

"""
####################
#   Sigma phase
####################
myName='local_env_vertex_phase-sigma'
# Phason matrix for Sigma-phase
u11 = 0.267949192431123
u12 = 0
u21 = 0
u22 = 0.267949192431123
# 4D point cut throught the 2D external space.
m01,m02,m03,m04 = 0.5, 0.0, 0.5, 0.5                             # for sigma-phase
"""



od.qcstrc(mystrc,\
        path=opath,\
        basename='%s_hmax%d'%(myName,h1max),\
        phason_matrix= np.array([\
                        [u11, u12],\
                        [u21, u22],\
                        ]),\
        n1max=h1max,\
        n5max=h5max,\
        origin_shift=np.array([m01,m02,m03,m04, 0.0, 0.0]),\
        option=2,\
        verbose=verbose,\
        )
