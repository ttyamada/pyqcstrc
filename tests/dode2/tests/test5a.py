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
myName='local_env_vertex'

# Generate vertices of NG tiling with 4 local envs.
# see T. Yamada, Acta Cryst. (2022). B78, 247–252

# import asymmetric part of ODs (occupation domains)
od_1_asym=od.read_xyz(path=xyzpath,basename='OD_local_env_1')
od_2_asym=od.read_xyz(path=xyzpath,basename='OD_local_env_2')
od_3_asym=od.read_xyz(path=xyzpath,basename='OD_local_env_3')
od_4_asym=od.read_xyz(path=xyzpath,basename='OD_local_env_4')

# Wyckoff positions:
V_a=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # 1a, (0,0,0,0)
V0=V_a

# shift of OD along Epar:
eshift_1=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
eshift_2=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
eshift_3=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
eshift_4=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])

#-----------
# Structure
#-----------
mystrc=[\
[od_1_asym,V_a,'Aa',eshift_1],\
[od_2_asym,V_a,'Bb',eshift_2],\
[od_3_asym,V_a,'Cc',eshift_3],\
[od_4_asym,V_a,'Dd',eshift_4],\
]

#---------------------------
print('\nGenerating structure...')
#---------------------------
h5max=0
h1max=5
#h1max=3
#hmax=2
oshift=[0.00178250, 0.00137613, 0.003987675, 0.2387783, 0, 0]
# Phason matrix for Sigma-phase
u11 = 0
u12 = 0
u21 = 0
u22 = 0
verbose=1

od.qcstrc(mystrc,\
        path=opath,\
        basename='%s_hmax%d'%(myName,h1max),\
        phason_matrix= np.array([\
                        [u11, u12],\
                        [u21, u22],\
                        ]),\
        n1max=h1max,\
        n5max=h5max,\
        origin_shift=oshift,
        option=2,
        verbose=verbose)
