#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
import sys
import numpy as np
import math
import cmath
try:
    from pyqcstrc.ico2.strc import (strc,
                                    )
    import pyqcstrc.ico2.occupation_domain as od
    from pyqcstrc.ico2.numericalc import (projection_numerical,
                                         )
    import pyqcdiff.common.atom as atm
except ImportError:
    print('import error\n')
import matplotlib.pyplot as plt

EPS=1e-6

# Predefined 6D coordinates in TAU-style
POS_V  = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
POS_C  = np.array([[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2]],dtype=np.int64)
POS_EC = np.array([[1,0,2],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
TAU=(1+np.sqrt(5))/2.0
CONST1=1/np.sqrt(2.0+TAU)
CONST2=1/2/np.sqrt(TAU+2)
PI=np.pi
TWOPI=2.0*PI

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

if __name__ == "__main__":
    
    wpath='./work_mag'
    basename='mag'
    nmax=5
    
    ofname='%s/%s_nmax%d.out'%(wpath,basename,nmax)
    
    x1a=[]
    y1a=[]
    x1b=[]
    y1b=[]
    a=read_file(ofname)
    for c in a:
        b=c.split()
        x=float(b[4])
        y=float(b[5])
        z=float(b[6])
        xyz=np.array([x,y,z])
        xyz=xyz/(np.linalg.norm(xyz)+1e-6)
        if z>0:
            x1a.append(float(xyz[0]))
            y1a.append(float(xyz[1]))
        else:
            x1b.append(float(xyz[0]))
            y1b.append(float(xyz[1]))

    fig=plt.figure(figsize=(8,4))
    ax1=fig.add_subplot(1,2,1)
    ax1.set_title('2fz(+)')
    ax1.scatter([0], [0], s=40000, marker='o', color='white', alpha=1.0, edgecolors='black')
    #ax1.scatter(x1b, y1b, s=200,   marker='o', color='white', alpha=1.0, edgecolors='black')
    ax1.scatter(x1a, y1a, s=40,    marker='o', color='black', alpha=1.0, edgecolors='black')
    ax1.set_xlim(-1.1,1.1)
    ax1.set_ylim(-1.2,1.2)
    ax1.axis("off")
    
    ax2=fig.add_subplot(1,2,2)
    ax2.set_title('2fz(-)')
    ax2.scatter([0], [0], s=40000, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax2.scatter(x1b, y1b, s=200,   marker='o', color='white', alpha=1.0, edgecolors='black')
    #ax2.scatter(x1a, y1a, s=40,    marker='o', color='black', alpha=1.0, edgecolors='black')
    ax2.set_xlim(-1.1,1.1)
    ax2.set_ylim(-1.2,1.2)
    ax2.axis("off")
    
    plt.savefig('%s/%s_nmax%d.png'%(wpath,basename,nmax))
    #plt.show()
    
