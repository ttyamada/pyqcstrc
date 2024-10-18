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
except ImportError:
    print('import error\n')

# Predefined 6D coordinates in TAU-style
POS_V  = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
POS_C  = np.array([[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2]],dtype=np.int64)
POS_EC = np.array([[1,0,2],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)

if __name__ == "__main__":
    
    """
    ######################
    #        TEST        #
    model_name = 'test'
    brv='p'
    aico = 5.689 # in Ang. CdYb
    select='atom'
    #select='mag'
    ######################
    flag_od = 1  # asymmetric OD is used.
    xyzpath='../../../xyz/ico'
    od0=od.read_xyz(path=xyzpath,basename='rtod0_asymmeric',  select='tetrahedron',verbose=0)
    #od0=od.read_xyz(path=xyzpath,basename='strt_aysmmetric',  select='tetrahedron',verbose=0)
    #
    elm_A = 'Yb'
    elm_B = 'Cd'
    #
    occ = 1.0
    rmax = 1.0
    be = 1.519 # DW factor
    # eshift:
    xe0=[0,0,0]
    # magnetic moment
    mu=[0.75,0,0] # along 5f,3f,2f axces.
    #
    myModel = {}
    #             element, [OD,  OD shape, symmetric or asymmetric],  coordinate,   eshift, be, rmax, mu(magnetic moment)
    myModel[0] = [elm_A,['polyhedron', od0, flag_od],              POS_V,       xe0, be, occ, rmax, mu]
    """
    
    """
    ######################
    #   Tsai-type iQC    #
    model_name = 'ico'
    brv='p'
    aico = 5.689 # in Ang. CdYb
    #select='atom'
    select='mag'
    ######################
    flag_od = 1  # asymmetric OD is used.
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
    #
    elm_A = 'Yb'
    elm_B = 'Cd'
    #
    occ = 1.0
    rmax = 1.0
    be = 1.519 # DW factor
    # eshift:
    xe0=[0,0,0]
    # magnetic moment
    mu=[0.75,0,0] # along 5f,3f,2f axces.
    #
    myModel = {}
    #             element, [OD,  OD shape, symmetric or asymmetric],  coordinate,   eshift, be, rmax, mu(magnetic moment)
    myModel[0] = [elm_A,['polyhedron', od1, flag_od],              POS_V,       xe0, be, occ, rmax, mu]
    myModel[1] = [elm_A,['polyhedron', od2, flag_od],              POS_V,       xe0, be, occ, rmax, mu]
    myModel[2] = [elm_A,['polyhedron', od3, flag_od],              POS_V,       xe0, be, occ, rmax, mu]
    myModel[3] = [elm_A,['polyhedron', od4, flag_od],              POS_V,       xe0, be, occ, rmax, mu]
    myModel[4] = [elm_A,['polyhedron', od5, flag_od],              POS_V,       xe0, be, occ, rmax, mu]
    myModel[5] = [elm_A,['polyhedron', od6, flag_od],              POS_V,       xe0, be, occ, rmax, mu]
    myModel[6] = [elm_A,['polyhedron', od7, flag_od],              POS_V,       xe0, be, occ, rmax, mu]
    myModel[7] = [elm_A,['polyhedron', od8, flag_od],              POS_V,       xe0, be, occ, rmax, mu]
    myModel[8] = [elm_B,['polyhedron', od0, flag_od],              POS_V,       xe0, be, occ, rmax, 0]
    """
    
    """
    #############################
    #   P-type AKN tiling
    model_name = 'pakn'
    brv='s'
    aico = 5.0
    select='atom'
    #select='mag'
    #############################
    flag_od = 1  # asymmetric OD is used.
    xyzpath='../../../xyz/ico'
    # Kumazawa's OD for icosahedron shell
    od0=od.read_xyz(path=xyzpath,basename='rt_od_asym',  select='tetrahedron',verbose=0)
    #
    elm_A = 'Yb'
    elm_B = 'Cd'
    #
    occ = 1.0
    rmax = 1.0
    be = 1.519 # DW factor
    # eshift:
    xe0=[0,0,0]
    # magnetic moment
    mu=[0.75,0,0] # along 5f,3f,2f axces.
    #
    #
    pos_ec= np.array([[1,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
    #
    myModel = {}
    #             element, [OD,  OD shape, symmetric or asymmetric],  coordinate,   eshift, be, rmax, mu(magnetic moment)
    myModel[0] = [elm_A,['polyhedron', od0, flag_od],              POS_V,       xe0, be, occ, rmax, 0]
    """

    #"""
    #############################
    #   F-type AKN tiling
    model_name = 'fakn'
    brv='s'
    #aico = 5.0*2
    aico = 5.0
    select='atom'
    #select='mag'
    #############################
    flag_od = 1  # asymmetric OD is used.
    xyzpath='../../../xyz/ico'
    # Kumazawa's OD for icosahedron shell
    od0=od.read_xyz(path=xyzpath,basename='rt_od_asym',  select='tetrahedron',verbose=0)
    od1=od.read_xyz(path=xyzpath,basename='trt_od_asym',  select='tetrahedron',verbose=0)
    #
    elm_A = 'Yb'
    elm_B = 'Cd'
    #
    occ = 1.0
    rmax = 1.0
    be = 1.519 # DW factor
    # eshift:
    xe0=[0,0,0]
    # magnetic moment
    mu=[0.75,0,0] # along 5f,3f,2f axces.
    #
    #
    pos_ec= np.array([[1,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
    #
    myModel = {}
    #             element, [OD,  OD shape, symmetric or asymmetric],  coordinate,   eshift, be, rmax, mu(magnetic moment)
    myModel[0] = [elm_A,['polyhedron', od0, flag_od],              POS_V,       xe0, be, occ, rmax, 0]
    myModel[1] = [elm_B,['polyhedron', od1, flag_od],             pos_ec,       xe0, be, occ, rmax, 0]
    #myModel[1] = [elm_B,['polyhedron', od0, flag_od],             pos_ec,       xe0, be, occ, rmax, 0]
    #"""
    
    ######################
    #    COMMON
    ######################
    nmax = 1
    #oshift=np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #oshift=np.array([ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    oshift=np.array([ 0.01, -0.02, 0.03, -0.04, 0.05, 0.06])
    #
    out=strc(aico,brv,myModel,nmax,oshift,verbose=1)
    od.write_vesta(out,path='.',basename='%s_nmax%d_%s'%(model_name,nmax,select),color='k',select=select,verbose=0)
    
