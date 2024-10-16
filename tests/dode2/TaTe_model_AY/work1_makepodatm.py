#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

#import timeit
#import os
import sys
import numpy as np
import pyqcstrc.dode2.math1 as math1
import pyqcstrc.dode2.occupation_domain as od
import pyqcstrc.dode2.two_occupation_domains as ods

#------------------------------------------------------
# Akiji Yamamoto, 5D model of Ta-Te quasicrystals
# Acta Cryst. (2004). A60, 142±145
#------------------------------------------------------

# Wyckoff positions:
V_1a=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,0,0,0)
V_4a=np.array([[ 0, 0, 1],[ 2, 0, 3],[ 0, 0, 1],[ 1, 0, 3],[ 0, 0, 1],[ 0, 0, 1]]) # (0,2/3,0,1/3)
V_6a=np.array([[ 0, 0, 1],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,1/2,0,0)
V_6b=np.array([[ 0, 0, 1],[ 1, 0, 2],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,1/2,1/2,0)

opath='./work1/qcstrc'
xyzpath='./work1'

lst_vlst = []
lst_vtx = []

vlst = [\
        [1,2,3],\
        ]
        
od_a=od.read_xyz(xyzpath,basename='od_1_asym')
vtx=od.write_vesta(od_a,opath,basename='od_1_asym',select='podatm')
lst_vtx.append(vtx)
lst_vlst.append(vlst)
#od.write_xyz(vtx,opath,basename='od_1_asym',select='vertex')

num=0

#vtx=od.read_xyz(opath,basename='od_1_asym',select='vertex')
od.write_podatm(obj=lst_vtx[num],position=V_1a,vlist=lst_vlst[num],path=opath,basename='od_1_asym')
