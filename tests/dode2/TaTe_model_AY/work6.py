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

opath='./work6_strc'

#------------------------------------------------------
# Akiji Yamamoto, 5D model of Ta-Te quasicrystals
# Acta Cryst. (2004). A60, 142±145
#------------------------------------------------------

#---------------------------
print('Generating ODs...')
#---------------------------

# import asymmetric part of OD(occupation domain), which correspnd to Fig.3 c--f in the Ref.
od_a_1_asym=od.read_xyz(path='work2_OD_A',basename='od_a_1_asym')
od_a_2_asym=od.read_xyz(path='work2_OD_A',basename='od_a_2_asym')
od_a_3_asym=od.read_xyz(path='work2_OD_A',basename='od_a_3_asym')
od_a_4_asym=od.read_xyz(path='work2_OD_A',basename='od_a_4_asym')
od_a_5_asym=od.read_xyz(path='work2_OD_A',basename='od_a_5_asym')
od_a_6_asym=od.read_xyz(path='work2_OD_A',basename='od_a_6_asym')
od_b_1_asym=od.read_xyz(path='work3_OD_B',basename='od_b_1_asym')
od_b_2_asym=od.read_xyz(path='work3_OD_B',basename='od_b_2_asym')
od_b_3_asym=od.read_xyz(path='work3_OD_B',basename='od_b_3_asym')
od_b_4_asym=od.read_xyz(path='work3_OD_B',basename='od_b_4_asym')
od_c_1_asym=od.read_xyz(path='work4_OD_C',basename='od_c_1_asym')
od_c_2_asym=od.read_xyz(path='work4_OD_C',basename='od_c_2_asym')
od_c_3_asym=od.read_xyz(path='work4_OD_C',basename='od_c_3_asym')
od_c_4_asym=od.read_xyz(path='work4_OD_C',basename='od_c_4_asym')
od_d_1_asym=od.read_xyz(path='work5_OD_D',basename='od_d_1_asym')
od_d_2_asym=od.read_xyz(path='work5_OD_D',basename='od_d_2_asym')
od_d_3_asym=od.read_xyz(path='work5_OD_D',basename='od_d_3_asym')

# Wyckoff positions:
u1=0.1
u2=0.1
u3=0.1
u4=0.1
u5=0.1
z1,z2,z3=1,0,4
V_1a =np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])         # ( 0,  0,  0,  0,  0)
V_2a =np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 1, 0, int(1/u1)],[ 0, 0, 1]]) # ( 0,  0,  0,  0, u1)
V_4a =np.array([[ 0, 0, 1],[ 2, 0, 3],[ 0, 0, 1],[ 1, 0, 3],[ 1, 0, int(1/u2)],[ 0, 0, 1]]) # ( 0,2/3,  0,1/3, u2)
V_6a =np.array([[ 0, 0, 1],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 1, 0, int(1/u3)],[ 0, 0, 1]]) # ( 0,1/2,  0,  0, u3)
V_6b =np.array([[ 0, 0, 1],[ 1, 0, 2],[ 1, 0, 2],[ 0, 0, 1],[ 1, 0, int(1/u4)],[ 0, 0, 1]]) # ( 0,1/2,1/2,  0, u4)
V_12a=np.array([[ 0, 0, 1],[ 1, 0, 2],[z1,z2,z3],[ 0, 0, 1],[ 1, 0, int(1/u5)],[ 0, 0, 1]]) # ( 0,1/2,  z,  0, u5)

# Square
xe0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,0,0,0)

#-----------
# Structure
#-----------
mystrc=[\
[od_a_1_asym,V_1a,'Xx',xe0],\
[od_a_2_asym,V_1a,'Xx',xe0],\
[od_a_3_asym,V_1a,'Xx',xe0],\
[od_a_4_asym,V_1a,'Xx',xe0],\
[od_a_5_asym,V_1a,'Xx',xe0],\
#[od_a_6_asym,V_1a,'Xx',xe0],\
#[od_b_1_asym,V_6a,'Zz',xe0],\
#[od_b_2_asym,V_6a,'Zz',xe0],\
#[od_b_3_asym,V_6a,'Zz',xe0],\
#[od_b_4_asym,V_6a,'Zz',xe0],\
[od_c_1_asym,V_4a,'Yy',xe0],\
[od_c_2_asym,V_4a,'Yy',xe0],\
[od_c_3_asym,V_4a,'Yy',xe0],\
#[od_c_4_asym,V_4a,'Yy',xe0],\
#[od_d_1_asym,V_6b,'Ww',xe0],\
#[od_d_2_asym,V_6b,'Ww',xe0],\
#[od_d_3_asym,V_6b,'Ww',xe0],\
#[od_d_1_asym,V_12a,'Vv',xe0],\
#[od_d_2_asym,V_12a,'Vv',xe0],\
#[od_d_3_asym,V_12a,'Vv',xe0],\
]

#---------------------------
print('done')
print('Generating atomic coordinates...')
#---------------------------
#h1max=6
#h1max=5
h1max=3
#h1max=2
#h5max=0
h5max=1
#oshift=[0.00178250, 0.00137613, 0.003987675, 0.2387783, 0, 0]
oshift=[0.00178250, 0.00137613, 0.003987675, 0.0002387, 0, 0]
#oshift=[0, 0, 0, 0, 0, 0]
basename='model_ay'
# Phason matrix for Sigma-phase
u11=0
u12=0
u21=0
u22=0

# Lattice constants, TaTe, Yamamoto, 2003
# Tokumoto et al, Nature Communications,(2024)15:1529
##### Tokumoto et al, Nature Communications,(2024)15:1529
# edge length = 2/np.sqrt(6)*apar
#lata=23.96 # in Angstr.
lata=6.74 # in Angstr.
latc=10.391 # in Angstr.

#ptg='12/mmm'
#ptg='12mm'
#ptg='-12m2'
ptg='-12'
#ptg='12'
od.qcstrc(lata,latc,\
        mystrc,\
        path=opath,\
        basename='%s_hmax%d'%(basename,h1max),\
        phason_matrix= np.array([\
                        [u11, u12],\
                        [u21, u22],\
                        ]),\
        n1max=h1max,\
        n5max=h5max,\
        origin_shift=oshift,\
        option=2,\
        pg=ptg,\
        verbose=1)
print('done')
