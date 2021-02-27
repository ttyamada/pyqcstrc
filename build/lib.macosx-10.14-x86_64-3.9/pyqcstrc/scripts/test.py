#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# PyQC - Python tools for Quasi-Crystallography
# Copyright (c) 2020 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import os
import sys
import numpy as np
import pyqcstrc.icosah.occupation_domain as od
import pyqcstrc.icosah.two_occupation_domains as ods
import pyqcstrc.icosah.utils as utils

# --------------------------------------
# PREDIFINED 6D POSITIONs in TAU style
# --------------------------------------
# Difinition of seven vertces A-G of asymmetric part of STRT OD
VRTX_A=np.array([[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
VRTX_B=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) 
VRTX_C=np.array([[ 2,-1, 2],[-2, 1, 2],[-2, 1, 2],[ 8,-5, 2],[-2, 1, 2],[ 8,-5, 2]])
VRTX_D=np.array([[ 2,-1, 2],[-2, 1, 2],[-2, 1, 2],[ 0, 0, 1],[-2, 1, 2],[ 0, 0, 1]])
VRTX_E=np.array([[ 2,-1, 2],[-2, 1, 2],[-2, 1, 2],[ 2,-1, 2],[-2, 1, 2],[-2, 1, 2]])
VRTX_F=np.array([[ 0, 1, 2],[ 2,-1, 2],[ 2,-1, 2],[ 2,-1, 2],[ 2,-1, 2],[ 2,-1, 2]])
VRTX_G=np.array([[ 2,-1, 2],[-2, 1, 2],[-2, 1, 2],[-8, 5, 2],[-2, 1, 2],[-2, 1, 2]])
# Difinition of three vertces H-I of asymmetric part of RT OD
VRTX_H=np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
VRTX_I=np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
VRTX_J=np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 0, 0, 2],[-1, 0, 2],[ 0, 0, 2]])
# Origin 0,0,0,0,0,0
POS_O0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
# b and c linckages
POS_B1=np.array([[ 1, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 1],[ 0, 0, 1]]) # b_1
POS_B2=np.array([[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 1, 0, 1]]) # b_2
POS_B3=np.array([[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 1, 0, 1],[ 1, 0, 1]]) # b_3
POS_B4=np.array([[ 1, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1]]) # b_4
POS_B5=np.array([[ 1, 0, 1],[ 0, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 1]]) # b_5
POS_B6=np.array([[-1, 0, 1],[ 0, 0, 1],[-1, 0, 1],[-1, 0, 1],[-1, 0, 1],[ 0, 0, 1]]) # b_22
POS_B7=np.array([[-1, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[-1, 0, 1],[-1, 0, 1]]) # b_30
POS_C1=np.array([[ 0, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 1]]) # c_1
POS_C2=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 1],[-1, 0, 1]]) # c_2
POS_C3=np.array([[ 0, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 1],[-1, 0, 1],[ 0, 0, 1]]) # c_3
POS_C4=np.array([[ 0, 0, 1],[ 0, 0, 1],[-1, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 1, 0, 1]]) # c_4
POS_C5=np.array([[ 0, 0, 1],[-1, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 1, 0, 1],[ 0, 0, 1]]) # c_5
POS_C6=np.array([[ 1, 0, 1],[ 0, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # c_14
POS_C7=np.array([[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 1, 0, 1]]) # c_19

#----------------------------------------------------
# STRT OD defined by three tetrahedra 1-3
#----------------------------------------------------
# Here we define three tetrahedra which define the 
# ansymmetric part of the STRT OD.
#----------------------------------------------------
# tetrahedron_1: B,C,D,E
# tetrahedron_2: B,C,G,E
# tetrahedron_3: B,A,C,G
tmp=np.append(VRTX_B,VRTX_C)
tmp=np.append(tmp,VRTX_D)
tmp=np.append(tmp,VRTX_E) # tetrahedron_1
tmp=np.append(tmp,VRTX_B)
tmp=np.append(tmp,VRTX_C)
tmp=np.append(tmp,VRTX_G)
tmp=np.append(tmp,VRTX_E) # tetrahedron_2
tmp=np.append(tmp,VRTX_B)
tmp=np.append(tmp,VRTX_A)
tmp=np.append(tmp,VRTX_C)
tmp=np.append(tmp,VRTX_G) # tetrahedron_3
od_strt_tetrahedron_3=tmp.reshape(12,6,3) # Objest defined by set of three tetrahedra
#
obj_tetrahedron=od_strt_tetrahedron_3

pos=[POS_B1,POS_B2,POS_B3,POS_B4,POS_B5,POS_B6,POS_B7,POS_C2,POS_C5,POS_C6,POS_C7]
#flg=[     0,     0,     0,     0,     0,     0,     0,     1,     1,     1,     1] # 1 (7,0,7)
#flg=[     1,     1,     1,     1,     1,     0,     0,     1,     1,     0,     0] # 2 (10,5,5)
flg=[     1,     0,     1,     1,     1,     1,     1,     0,     0,     0,     1] # 3 (10,6,4)
num=3

# -------------------
#  Intersection, obj1 and obj2
# -------------------
#i=1
#i=4
i=9
#i=10
print('%d-th pos'%(i))
path='./test_1a/seed_%d'%(i)
if os.path.exists(path) == False:
    os.makedirs(path)
else:
    pass

# obj1
seed=obj_tetrahedron
strt_1=od.as_it_is(seed)


utils.generator_xyz_dim4_tmp(strt_1,1)