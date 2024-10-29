#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
import sys
import numpy as np
try:
    from pyqcstrc.ico2.strc import (strc,
                                    )
    import pyqcstrc.ico2.occupation_domain as od
    import pyqcstrc.ico2.symmetry as symmetry
    import pyqcstrc.ico2.numericalc as numericalc
    import pyqcstrc.ico2.symmetry_numerical as symmetry_numerical
except ImportError:
    print('import error\n')

# Predefined 6D coordinates in TAU-style
POS_V  = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
POS_C  = np.array([[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2]],dtype=np.int64)
POS_EC = np.array([[1,0,2],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)

V0 = np.array([ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])

if __name__ == "__main__":
    
    xyzpath='../../../xyz/ico/kumazawa'
    # Kumazawa's OD for icosahedron shell
    od0=od.read_xyz(path=xyzpath,basename='strt_aysmmetric',  select='tetrahedron',verbose=0)
    od1=od.read_xyz(path=xyzpath,basename='R3R3R5',  select='tetrahedron',verbose=0)
    od2=od.read_xyz(path=xyzpath,basename='R3R3R5R5',select='tetrahedron',verbose=0)
    od3=od.read_xyz(path=xyzpath,basename='R3R5',    select='tetrahedron',verbose=0)
    od4=od.read_xyz(path=xyzpath,basename='R3R5R5',  select='tetrahedron',verbose=0)
    od5=od.read_xyz(path=xyzpath,basename='R3R5R5R5',select='tetrahedron',verbose=0)
    od6=od.read_xyz(path=xyzpath,basename='R5',      select='tetrahedron',verbose=0)
    od7=od.read_xyz(path=xyzpath,basename='R5R5',    select='tetrahedron',verbose=0)
    od8=od.read_xyz(path=xyzpath,basename='R5R5R5',  select='tetrahedron',verbose=0)
    
    objs = [od0,od1,od2,od3,od4,od5,od6,od7,od8]
    
    point = np.array([ -1.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    oshift = np.array([ 0.03, -0.03, -0.03, 0.00, -0.03, -0.02]) # inside asymmetric unit
    point = numericalc.projection3_numerical(point+oshift)
    
    idx=0
    symop = symmetry.icosasymop_array()
    for i1,op in enumerate(symop):
        counter=0
        for obj in objs:
            od = symmetry.symop_obj(op,obj,centre=POS_V)
            for tetrahedron in od:
                if numericalc.inside_outside_tetrahedron_tau_v2(point,tetrahedron):
                    counter+=1
                    break
        if counter!=0:
            idx=i1
            print(i1)
    
    x1=np.array([1, 0, 0, 0, 0, 0],dtype=np.float64) #5f
    x2=np.array([1, 0,-1, 0,-1, 0],dtype=np.float64) #3f
    x3=np.array([1, 0, 0, 0,-1, 0],dtype=np.float64) #2f
    
    y1=symmetry_numerical.symop_vec(symop[idx],x1,centre=V0)
    y2=symmetry_numerical.symop_vec(symop[idx],x2,centre=V0)
    y3=symmetry_numerical.symop_vec(symop[idx],x3,centre=V0)
    print(y1)
    print(y2)
    print(y3)