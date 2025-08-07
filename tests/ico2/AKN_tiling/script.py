#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
import sys
import numpy as np
try:
    from pyqcstrc.ico2.strc import strc
    from pyqcstrc.ico2.math1 import mul_vectors
    import pyqcstrc.ico2.occupation_domain as od
except ImportError:
    print('import error\n')

#----------------------------------------------------------
# Generating a primitive AKN tiling with edge length of 1.
#----------------------------------------------------------
# Predefined 6D coordinates in TAU-style
pos_v  = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
pos_ec = np.array([[1,0,2],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
pos_bc = np.array([[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2]])

brv = 'p'
aico = 1.0 # icosahedral lattice constant
flag_od = 1  # asymmetric OD is used.
xyzpath='../../../xyz/ico'
od1=od.read_xyz(path=xyzpath,basename='rt_od_asym',  select='tetrahedron',verbose=0)

# edge centres
xyzpath='../examples/example3'
od2=od.read_xyz(path=xyzpath,basename='common_od_asym_smpl',  select='tetrahedron',verbose=0)

# dodecahedral star
xyzpath='../../../xyz/ico'
od3=od.read_xyz(path=xyzpath,basename='d_star0_asym_simple',  select='tetrahedron',verbose=0)
od3 = mul_vectors(od3.reshape(4,6,3),[1, 1, 1])
od3 = od3.reshape(1,4,6,3)


occ = 1.0 # occupancy
rmax = 1.0 # 
be = 1.0 # DW factor

# eshift (shift of OD in external space):
xe0 = [0., 0., 0.] 


myModel = {}
#             element, [OD,  OD shape, symmetric or asymmetric],  coordinate,   eshift, be, occ, rmax, mu(magnetic moment)
myModel[0] = ['H',     ['polyhedron', od1, flag_od],              pos_v,        xe0, be, occ, rmax, 0]
myModel[1] = ['He',    ['polyhedron', od2, flag_od],              pos_ec,       xe0, be, occ, rmax, 0]
myModel[2] = ['Li',    ['polyhedron', od3, flag_od],              pos_bc,       xe0, be, occ, rmax, 0]

nmax = 2 # maximum index for generating lattice points.

# Origin shift
oshift = [ 0.03, -0.03, -0.03, 0.00, -0.03, -0.02]

#=====================================
# three 6d vectors for eshift and mu corresponding to xe1,xe2,xe3 in QUASI
#=====================================
xe1 = [1, 0, 0, 0, 0, 0] # 5f
xe2 = [1, 0,-1, 0,-1, 0] # 3f
xe3 = [1, 0, 0, 0,-1, 0] # 2f

out = strc(aico = aico, brv = brv, model = myModel, nmax = nmax, oshift = oshift, x1 = xe1, x2 = xe2, x3 = xe3)
od.write_vesta(out, path = '.',basename = 'AKN_tiling_nmax%d'%(nmax), color = 'k', select = 'atom', verbose = 0)
