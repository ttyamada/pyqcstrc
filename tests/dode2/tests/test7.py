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

opath='./test7'
xyzpath='../../../xyz/dode'

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



###############
#  Vertex
###############
# make symmetric OD
od_sym=sym.generator_obj_symmetric_obj(od_a_asym, V_a)
od.write(obj=od_sym, path=opath, basename='od_a_sym', format='vesta', color='k',select='normal')
od.write(obj=od_sym, path=opath, basename='od_a_sym', format='xyz')
# Outline of OBJ_1
od_sym_1=od.outline(od_sym)
od.write(obj=od_sym_1,path=opath,basename='od_a_sym_outline',format='vesta',color='r',select='egdes')
#od.write(obj=od_sym_1,path=opath,basename='od_a_sym_outline',format='xyz')

num_coset=sym.coset(V_a,5)
print(num_coset)
objs1=sym.generator_obj_symmetric_obj_specific_symop(od_sym,V0,num_coset)
od.write(obj=objs1, path=opath, basename='od_a_sym_1', format='vesta', color='k',select='normal')
print(objs1.shape)

###############
#  Edge centre
###############
# make symmetric OD
od_sym=sym.generator_obj_symmetric_obj(od_b_asym, V_b)
od.write(obj=od_sym, path=opath, basename='od_b_sym', format='vesta', color='k',select='normal')
od.write(obj=od_sym, path=opath, basename='od_b_sym', format='xyz')
# Outline of OBJ_1
od_sym_1=od.outline(od_sym)
od.write(obj=od_sym_1,path=opath,basename='od_b_sym_outline',format='vesta',color='r',select='egdes')
#od.write(obj=od_sym_1,path=opath,basename='od_b_sym_outline',format='xyz')

num_coset=sym.coset(V_b,5)
print(num_coset)
objs1=sym.generator_obj_symmetric_obj_specific_symop(od_sym,V0,num_coset)
od.write(obj=objs1, path=opath, basename='od_b_sym_1', format='vesta', color='k',select='normal')
print(objs1.shape)

###############
#  Triangle
###############

"""
# make symmetric OD
od_sym=sym.generator_obj_symmetric_obj(od_c_asym, V_c)
od.write(obj=od_sym, path=opath, basename='od_c_sym', format='vesta', color='k',select='normal')
od.write(obj=od_sym, path=opath, basename='od_c_sym', format='xyz')
# Outline of OBJ_1
od_sym_1=od.outline(od_sym)
od.write(obj=od_sym_1,path=opath,basename='od_c_sym_outline',format='vesta',color='r',select='egdes')
#od.write(obj=od_sym_1,path=opath,basename='od_c_sym_outline',format='xyz')



# make symmetric OD
od_sym=sym.generator_obj_symmetric_obj(od_d_asym, V_d)
od.write(obj=od_sym, path=opath, basename='od_d_sym', format='vesta', color='k',select='normal')
od.write(obj=od_sym, path=opath, basename='od_d_sym', format='xyz')
# Outline of OBJ_1
od_sym_1=od.outline(od_sym)
od.write(obj=od_sym_1,path=opath,basename='od_d_sym_outline',format='vesta',color='r',select='egdes')
#od.write(obj=od_sym_1,path=opath,basename='od_d_sym_outline',format='xyz')



# make symmetric OD
od_sym=sym.generator_obj_symmetric_obj(od_e_asym, V_e)
od.write(obj=od_sym, path=opath, basename='od_e_sym', format='vesta', color='k',select='normal')
od.write(obj=od_sym, path=opath, basename='od_e_sym', format='xyz')
# Outline of OBJ_1
od_sym_1=od.outline(od_sym)
od.write(obj=od_sym_1,path=opath,basename='od_e_sym_outline',format='vesta',color='r',select='egdes')
#od.write(obj=od_sym_1,path=opath,basename='od_e_sym_outline',format='xyz')
"""