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

opath='./work7_podatm'


#==========
#  OD A
#==========
lst_vtx=[]
for num in range(6):
    od_a=od.read_xyz(path='./work2_OD_A',basename='od_a_%d_asym'%(num+1))
    vtx=od.write_vesta(od_a,opath,basename='od_a_%d_asym'%(num+1),select='podatm')
    lst_vtx.append(vtx)
    od.write_xyz(vtx,opath,basename='od_a_%d_asym'%(num+1),select='vertex')

lst_vlst=[]

# A1
vlst = [\
        [1,2,3],\
        ]
lst_vlst.append(vlst)

# A2
vlst = [\
        [1,2,3],\
        ]
lst_vlst.append(vlst)

# A3
vlst = [\
        [3,5,2,2,4,4,6,6,1],\
        ]
lst_vlst.append(vlst)

# A4
vlst = [\
        [4,3,5,5,1,1,6,6,2],\
        ]
lst_vlst.append(vlst)

# A5
vlst = [\
        [3,1,8,10,2,2,4,7,5,5,9,11,6],\
        ]
lst_vlst.append(vlst)

# A6
vlst = [\
        [1,2,3],\
        ]
lst_vlst.append(vlst)

for num in range(6):
    vtx=od.read_xyz(opath,basename='od_a_%d_asym'%(num+1),select='vertex')
    od.write_podatm(obj=lst_vtx[num],position=V_1a,vlist=lst_vlst[num],path=opath,basename='od_a_%d_asym'%(num+1))


#"""
#==========
#  OD C
#==========
v=math1.mul_vector(V_4a,np.array([-1,0,1]))
lst_vtx=[]
for num in range(4):
    od_c=od.read_xyz(path='./work4_OD_C',basename='od_c_%d_asym'%(num+1))
    od_c=od.shift(od_c,shift=v)
    vtx=od.write_vesta(od_c,opath,basename='od_c_%d_asym'%(num+1),select='podatm')
    lst_vtx.append(vtx)
    od.write_xyz(vtx,opath,basename='od_c_%d_asym'%(num+1),select='vertex')

lst_vlst=[]

# C1
vlst = [\
        [1,2,3],\
        ]
lst_vlst.append(vlst)

# C2
vlst = [\
        [4,1,5,5,2,2,3,3,6],\
        ]
lst_vlst.append(vlst)

# C3
vlst = [\
        [5,2,8,9,3,3,1,10,6,6,4,7,11],\
        [12,18,19,19,17,13,15,15,14,10,6,6,16]
        ]
lst_vlst.append(vlst)

# C4
vlst = [\
        [3,4,2,2,1],\
        ]
lst_vlst.append(vlst)

for num in range(4):
    vtx=od.read_xyz(opath,basename='od_c_%d_asym'%(num+1),select='vertex')
    od.write_podatm(obj=lst_vtx[num],position=V_4a,vlist=lst_vlst[num],path=opath,basename='od_c_%d_asym'%(num+1))
#"""