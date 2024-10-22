#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import timeit
import os
import sys
import numpy as np
import pyqcstrc.ico2.occupation_domain as od
import pyqcstrc.ico2.two_occupation_domains as ods
import pyqcstrc.ico2.symmetry as sym

import pyqcstrc.ico2.utils as utils

# Vertices of tetrahedron, v0,v1,v2,v3, which
# defines the asymmetric part of OD
v0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v1 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
v2 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
v3 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 0, 0, 2],[-1, 0, 2],[ 0, 0, 2]])
od0 = np.vstack([v0,v1,v2,v3]).reshape(1,4,6,3)
od.write_vesta(od0, path='./example6', basename='rtod0', color='r')
od.write_xyz(od0, path='./example6', basename='rtod0')

# generate symmetric OD, symmetric centre is v0.
od1 = od.symmetric(od0, v0)
od.write_vesta(od1, path='./example6', basename='rtod1', color='r')
od.write_xyz(od1, path='./example6', basename='rtod1')

# coordinate of position_1
pos1 = np.array([[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
od2 = od.shift(od1,pos1)  # move to position_1
od.write_vesta(od2, path='./example6', basename='rtod2', color='p')
od.write_xyz(od2, path='./example6', basename='rtod2')

# Three vectors w1,w2,w3, which defines the asymmetric unit.
w1 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
w2 = np.array([[ 0, 0, 1],[-1, 0, 4],[-1, 0, 4],[ 1, 0, 2],[-1, 0, 2],[ 1, 0, 2]])
w3 = np.array([[ 0, 0, 1],[ 1, 0, 4],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[ 1, 0, 4]])
vec_aum5 = np.vstack([w1,w2,w3]).reshape(3,6,3)
posEC = np.array([[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])

od1_asym = od.asymmetric(od1, posEC, vec_aum5)
od.write_vesta(od1_asym, path='./example6', basename='rtod1_asym', color='b')
od.write_xyz(od1_asym, path='./example6', basename='rtod1_asym')

#common_od = ods.intersection(od1_asym, od2, verbose=0)
#od.write_vesta(common_od, path='./example6', basename='common_od_asym', color='r')
#od.write_xyz(common_od, path='./example6', basename='common_od_asym')



# using intersection_convex()
common_od_new = ods.intersection_convex(od1_asym, od2, verbose=2)
od.write_vesta(common_od_new, path='./example6', basename='common_od_asym', color='r')
od.write_xyz(common_od_new, path='./example6', basename='common_od_asym')

#common_od_new2 = od.simplification(common_od_new,verbose=2)
#od.write_vesta(common_od_new2, path='./example6', basename='common_od_new_asym', color='r')
#od.write_xyz(common_od_new2, path='./example6', basename='common_od_new_asym')

od1a=od.simple_hand_step1(obj=common_od_new, path='./example6', basename_tmp='tmp')
merge_list=[[3,7,8,9],[3,6,7,9],[3,6,9,13],[3,4,6,7]]
common_od_new2 = od.simple_hand_step2(obj=od1a, merge_list=merge_list)
# shift to V:(0,0,0,0,0,0,0)
common_od_new2 = od.shift(common_od_new2,np.array([[-1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]))
od.write_vesta(common_od_new2, path='./example6', basename='common_od_new_asym', color='r')
od.write_xyz(common_od_new2, path='./example6', basename='common_od_new_asym')

# make symmetric OD
brv='p'
idx_ssym,idx_coset = sym.site_symmetry_and_coset(posEC,brv,verbose=1)
print('idx_ssym:',idx_ssym)
print('idx_coset:',idx_coset)
common_od_new3 = np.zeros((len(idx_ssym),len(common_od_new2),4,6,3),dtype=np.int64)
for i1,idx in enumerate(idx_ssym):
    common_od_new3[i1] = sym.generator_obj_symmetric_tetrahedron_0(common_od_new2,v0,idx)
common_od_new3=common_od_new3.reshape((len(idx_ssym)*len(common_od_new2),4,6,3))
od.write_vesta(common_od_new3, path='./example6', basename='common_od_new_sym', color='r')
od.write_xyz(common_od_new3, path='./example6', basename='common_od_new_sym')



# make symmetric OD@EC
common_od_new2
obj=utils.shift_object(common_od_new2,posEC)
obj=sym.generator_obj_symmetric_obj_specific_symop(obj,posEC,idx_ssym)
n1,n2,_,_,_=obj.shape
obj=obj.reshape(n1*n2,4,6,3)
od.write_vesta(obj, path='./example6', basename='common_od_new1_sym', color='r')
od.write_xyz(obj, path='./example6', basename='common_od_new1_sym')
