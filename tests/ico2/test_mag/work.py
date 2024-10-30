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
    from pyqcstrc.ico2.numericalc import (projection_numerical,
                                         )
except ImportError:
    print('import error\n')

# Predefined 6D coordinates in TAU-style
POS_V  = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
POS_C  = np.array([[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2]],dtype=np.int64)
POS_EC = np.array([[1,0,2],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
TAU=(1+np.sqrt(5))/2.0
CONST1=1/np.sqrt(2.0+TAU)
PI=np.pi

def read_file(file):
    try:
        f=open(file,'r')
    except IOError as e:
        print(e)
        sys.exit(0)
    line=[]
    while 1:
        a=f.readline()
        if not a:
            break
        line.append(a[:-1])
    return line

def gen_reference(nmax,size):
    
    th1=-np.arctan(1/TAU)
    th2=-PI/10
    cos1=np.cos(th1)
    sin1=np.sin(th1)
    cos2=np.cos(th2)
    sin2=np.sin(th2)
    rotx=np.array([[   1,    0,    0],\
                   [   0, cos1, sin1],\
                   [   0, -sin1, cos1]],dtype=np.float64)
    rotz=np.array([[ cos2, sin2,    0],\
                   [-sin2, cos2,    0],\
                   [    0,    0,    1]],dtype=np.float64)
    
    lst=[]
    for h1 in range(-nmax,nmax+1):
        for h2 in range(-nmax,nmax+1):
            for h3 in range(-nmax,nmax+1):
                for h4 in range(-nmax,nmax+1):
                    for h5 in range(-nmax,nmax+1):
                        for h6 in range(-nmax,nmax+1):
                            h123456=np.array([h1,h2,h3,h4,h5,h6],dtype=np.float64)
                            vn=projection_numerical(h123456)
                            ve=vn[0:3]*CONST1
                            vi=vn[3:6]
                            if np.linalg.norm(vi)<=size:
                                ve=rotz@rotx@ve # z along 5f axis
                                lst.append([ve,vi,h123456])
    return lst

if __name__ == "__main__":
    
    aico=5.68930 # in Ang. CdYb
    
    fname='./work_mag/mag.mld'
    a=read_file(fname)
    lst=[]
    for i1 in range(2,len(a)):
        b=a[i1].split()
        site=np.array([float(b[0]),float(b[1]),float(b[2])],dtype=np.float64)/aico
        atom=int(b[3])
        mvec=np.array([float(b[4]),float(b[5]),float(b[6])],dtype=np.float64)/2.0
        lst.append([site,atom,mvec])
    
    # generating reference list
    nmax=5
    size=4.26 # (0,1,1,1,1,1)perp
    ref=gen_reference(nmax,size)
    
    # comparison
    counter=0
    out=[]
    for a in lst:
        xyz=a[0]
        for b in ref:
            ve=b[0]
            vi=b[1]
            if np.allclose(xyz, ve, rtol=1e-03, atol=1e-06):
                print('%d %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f'%(counter,-vi[0],-vi[1],-vi[2],a[2][0],a[2][1],a[2][2]))
                out.append(['Eu',-vi,counter+1,b[2],a[2]])
                counter+=1
                break
    ofname='mag'
    select='mag'
    od.write_vesta(out,path='./work_mag',basename='%s_nmax%d'%(ofname,nmax),color='k',select=select,verbose=0)
    
    
    
    
    
    """
    # generating reference structure
    ######################
    #   Tsai-type iQC    #
    model_name = 'ref_pico'
    brv='p'
    aico = 1.0 # in Ang.
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
    elm_X = 'X'
    #
    occ = 1.0
    rmax = 1.0
    be = 1.519 # DW factor
    # eshift:
    xe0=[0,0,0]
    # magnetic moment, mu
    #mu=[1.0, 0.0, 0.0] # along 5f,3f,2f axces.
    mu=0
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
    #myModel[8] = [elm_X,['polyhedron', od0, flag_od],              POS_V,       xe0, be, occ, rmax, 0]
    
    nmax =1
    #oshift=[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #oshift=[ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    #oshift=[ 0.01, -0.02, 0.03, -0.04, 0.05, 0.06]
    oshift=[ 0.03, -0.03, -0.03, 0.00, -0.03, -0.02]
    
    #=====================================
    # three 6d vectors for eshift and mu
    # corresponding to xe1,xe2,xe3 in QUASI
    #=====================================
    xe1=[1, 0, 0, 0, 0, 0] #5f
    #xe1=[0, 1, 0, 0, 0, 0] #5f
    #xe1=[0, 0, 1, 0, 0, 0] #5f
    #xe1=[0, 0, 0, 1, 0, 0] #5f
    #xe1=[0, 0, 0, 0, 1, 0] #5f
    #xe1=[0, 0, 0, 0, 0, 1] #5f
    xe2=[1, 0,-1, 0,-1, 0] #3f
    xe3=[1, 0, 0, 0,-1, 0] #2f
    
    strc_ref=strc(aico,brv,myModel,nmax,oshift,xe1,xe2,xe3,verbose=1)
    od.write_vesta(strc_ref,path='.',basename='%s_nmax%d_%s'%(model_name,nmax,select),color='k',select=select,verbose=0)
    
                            
                            
    
    th1=-np.arctan(1/TAU)
    th2=-PI/10
    cos1=np.cos(th1)
    sin1=np.sin(th1)
    cos2=np.cos(th2)
    sin2=np.sin(th2)
    rotx=np.array([[   1,    0,    0],\
                   [   0, cos1, sin1],\
                   [   0, -sin1, cos1]],dtype=np.float64)
    rotz=np.array([[ cos2, sin2,    0],\
                   [-sin2, cos2,    0],\
                   [    0,    0,    1]],dtype=np.float64)
    # comparison
    for b in strc_ref:
        c=rotz@rotx@b[1] # z along 5f axis
        for a in lst:
            if np.allclose(a[0], c, rtol=1e-03, atol=1e-06):
                print(b[5],a[2])
    """