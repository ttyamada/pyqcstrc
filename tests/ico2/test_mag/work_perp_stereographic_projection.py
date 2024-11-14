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
import matplotlib.patches as patches


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
    #basename='mag'
    basename='T_10_m4_D10_L_mag'
    
    nmax=5
    
    ofname='%s/%s_nmax%d.out'%(wpath,basename,nmax)
    
    x1a=[]
    y1a=[]
    x1b=[]
    y1b=[]
    a=read_file(ofname)
    counter=0
    x_sum=0
    y_sum=0
    z_sum=0
    for c in a:
        b=c.split()
        xyz=np.array([float(b[4]),float(b[5]),float(b[6])])
        xyz=xyz/(np.linalg.norm(xyz)+1e-6)
        if xyz[2]>=0:
            X=xyz[0]/(1+xyz[2])
            Y=xyz[1]/(1+xyz[2])
            x1a.append(X)
            y1a.append(Y)
        else:
            X=xyz[0]/(1-xyz[2])
            Y=xyz[1]/(1-xyz[2])
            x1b.append(X)
            y1b.append(Y)
            counter+=1
            x_sum+=xyz[0]
            y_sum+=xyz[1]
            z_sum+=xyz[2]
    print('averaged:',x_sum/counter,y_sum/counter,z_sum/counter)
            
    fig=plt.figure(figsize=(12,12))
    ax1=fig.add_subplot(2,2,1)
    ax1.set_title('all')
    c = patches.Circle(xy=(0,0), radius=1.0, ec='k', fill=False)
    ax1.add_patch(c)
    #ax1.scatter([0], [0], s=40000, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax1.scatter(x1a, y1a, s=10,    marker='o', color='red', alpha=1.0, edgecolors='red')
    ax1.scatter(x1b, y1b, s=10,   marker='o', color='blue', alpha=1.0, edgecolors='blue')
    ax1.set_xlim(-1.1,1.1)
    ax1.set_ylim(-1.1,1.1)
    ax1.set(aspect=1)
    ax1.axis("off")
    
    ax2=fig.add_subplot(2,2,2)
    ax2.set_title('2fz(+)')
    #ax2.scatter([0], [0], s=40000, marker='o', color='white', alpha=1.0, edgecolors='black')
    c = patches.Circle(xy=(0,0), radius=1.0, ec='k', fill=False)
    ax2.add_patch(c)
    ax2.scatter(x1a, y1a, s=10,    marker='o', color='red', alpha=1.0, edgecolors='red')
    ax2.set_xlim(-1.1,1.1)
    ax2.set_ylim(-1.1,1.1)
    ax2.set(aspect=1)
    ax2.axis("off")
    
    ax3=fig.add_subplot(2,2,3)
    ax3.set_title('2fz(-)')
    c = patches.Circle(xy=(0,0), radius=1.0, ec='k', fill=False)
    ax3.add_patch(c)
    #ax3.scatter([0], [0], s=40000, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax3.scatter(x1b, y1b, s=10,   marker='o', color='blue', alpha=1.0, edgecolors='blue')
    ax3.set_xlim(-1.1,1.1)
    ax3.set_ylim(-1.1,1.1)
    ax3.set(aspect=1)
    ax3.axis("off")
    
    ax4=fig.add_subplot(2,2,4)
    ax4.set_title('2fz(-)')
    c = patches.Circle(xy=(0,0), radius=1.0, ec='k', fill=False)
    ax4.add_patch(c)
    ax4.scatter([x_sum/counter], [y_sum/counter], s=10,   marker='o', color='blue', alpha=1.0, edgecolors='blue')
    ax4.set_xlim(-1.1,1.1)
    ax4.set_ylim(-1.1,1.1)
    ax4.set(aspect=1)
    ax4.axis("off")
    
    
    plt.savefig('%s/%s_nmax%d.png'%(wpath,basename,nmax))
    #plt.show()
    
