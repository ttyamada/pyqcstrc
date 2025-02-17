#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import timeit
import os
import sys
import numpy as np
import ico2.occupation_domain as od
import ico2.two_occupation_domains as ods
import qnnum.qnnum as qnn
import qnvec.qnvec as qnv
import qnmat.qnmat as qnm

# Vertices of tetrahedron, v0,v1,v2,v3, which
# defines the asymmetric part of OD
M0=qnn.Qnnum([0,0,1],N) # 0
M1=qnn.Qnnum([1,0,2],N) # 1/2
M2=qnn.Qnnum([-1,0,2],N) # -1/2
M3=qnn.Qnnum([1,0,1],N)
M4=qnn.Qnnum([1,0,4],N)
M5=qnn.Qnnum([-1,0,4],N)

v0=np.array([M0,M0,M0,M0,M0,M0])
v1=np.array([M1,M2,M2,M2,M2,M2])
v2=np.array([M1,M2,M2,M1,M2,M2])
v3=np.array([M1,M2,M2,M0,M2,M0])

#v0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
#v1 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
#v2 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
#v3 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 0, 0, 2],[-1, 0, 2],[ 0, 0, 2]])
od0 = np.vstack([v0,v1,v2,v3]).reshape(1,4,6,3)
od.write_vesta(od0, path='./example3', basename='rtod0', color='r')
od.write_xyz(od0, path='./example3', basename='rtod0')

# generate symmetric OD, symmetric centre is v0.
od1 = od.symmetric(od0, v0)
od.write_vesta(od1, path='./example3', basename='rtod1', color='r')
od.write_xyz(od1, path='./example3', basename='rtod1')

# coordinate of position_1
pos1 = np.array([M3,M0,M0,M0,M0,M0]) # (1,0,0,0,0,0)
#pos1 = np.array([[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],
#                 [ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
od2 = od.shift(od1,pos1)  # move to position_1
od.write_vesta(od2, path='./example3', basename='rtod2', color='p')
od.write_xyz(od2, path='./example3', basename='rtod2')

# Three vectors w1,w2,w3, which defines the asymmetric unit.
w1 = np.array([M1,M2,M2,M2,M2,M2])
w2 = np.array([M0,M5,M5,M1,M2,M1])
w3 = np.array([M0,M4,M2,M1,M2,M4])

#w1 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
#w2 = np.array([[ 0, 0, 1],[-1, 0, 4],[-1, 0, 4],[ 1, 0, 2],[-1, 0, 2],[ 1, 0, 2]])
#w3 = np.array([[ 0, 0, 1],[ 1, 0, 4],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[ 1, 0, 4]])

vec_aum5 = np.vstack([w1,w2,w3]).reshape(3,6)  # reshape(3,6,3)
posEC = np.array([M1,M0,M0,M0,M0,M0])
#posEC = np.array([[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])

od1_asym = od.asymmetric(od1, posEC, vec_aum5)
od.write_vesta(od1_asym, path='./example3', basename='rtod1_asym', color='b')
od.write_xyz(od1_asym, path='./example3', basename='rtod1_asym')

common_od = ods.intersection(od1_asym, od2, verbose=0)
od.write_vesta(common_od, path='./example3', basename='common_od_asym', color='r')
od.write_xyz(common_od, path='./example3', basename='common_od_asym')
