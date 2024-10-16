#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

#import timeit
#import os
import sys
import numpy as np
import matplotlib.pyplot as plt
try:
    import pyqcstrc.dode2.numericalc as numericalc
except ImportError:
    print('import error\n')

EPS=1e-6

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

opath='./test5'

#h1max=6
#h1max=5
h1max=3
h5max=0 # 0th layer

#"""
##########
#  ddQC 
##########
basename='local_env_vertex'
# Phason matrix for ddQC
u11 = 0
u12 = 0
u21 = 0
u22 = 0
# 4D point cut throught the 2D external space.
m01,m02,m03,m04 = 0.00178250, 0.00137613, 0.003987675, 0.2387783 # for dd-QC
#"""

"""
####################
#   Sigma phase
####################
basename='local_env_vertex_phase-sigma'
# Phason matrix for Sigma-phase
u11 = 0.267949192431123
u12 = 0
u21 = 0
u22 = 0.267949192431123
# 4D point cut throught the 2D external space.
m01,m02,m03,m04 = 0.5, 0.0, 0.5, 0.5                             # for sigma-phase
"""

"""
####################
#   Phase 1
####################
basename='local_env_vertex_phase1'
# Phason matrix for phase1
u11 = -0.0717967697244909
u12 =  0
u21 =  0
u22 = -0.0717967697244909
# 4D point cut throught the 2D external space.
m01,m02,m03,m04 = 0, 0, 0, 0.5                                   # for phase 1
"""


phason_matrix= np.array([\
                [u11, u12],\
                [u21, u22],\
                ])
origin_shift=np.array([m01,m02,m03,m04, 0.0, 0.0])


v=numericalc.projection_numerical_phason(origin_shift,phason_matrix)
xe=v[3]
ye=v[4]


#===============================
# Drawing Tilling
#===============================
plt.figure(figsize=(8, 8))
#plt.axis('equal')
xylim=1.2
plt.xlim([-xylim, xylim])
plt.ylim([-xylim, xylim])

fname='%s/%s_hmax%d'%(opath,basename,h1max)
f=read_file('%s.xyz'%fname)
x1plot=[]
y1plot=[]
x2plot=[]
y2plot=[]
x3plot=[]
y3plot=[]
x4plot=[]
y4plot=[]
for i1 in range(2,len(f)):
    fi1=f[i1]
    a1=fi1.split()
    if int(a1[12])==h5max:
        if a1[0]=='Aa':
            x1plot.append(float(a1[5])-xe)
            y1plot.append(float(a1[6])-ye)
        elif a1[0]=='Bb':
            x2plot.append(float(a1[5])-xe)
            y2plot.append(float(a1[6])-ye)
        elif a1[0]=='Cc':
            x3plot.append(float(a1[5])-xe)
            y3plot.append(float(a1[6])-ye)
        else:
            x4plot.append(float(a1[5])-xe)
            y4plot.append(float(a1[6])-ye)

#plt.scatter(x1plot,y1plot,marker='o',s=40.,c='#000000',alpha=1.0) # black
#plt.scatter(x1plot,y1plot,marker='o',s=40.,c='#00008B',alpha=1.0) # darkblue
plt.scatter(x1plot,y1plot,marker='o',s=40.,c='b',alpha=1.0) # blue

#plt.scatter(x2plot,y2plot,marker='o',s=40.,c='#808080',alpha=1.0) # gray
#plt.scatter(x2plot,y2plot,marker='o',s=40.,c='#0000FF',alpha=1.0) # blue
plt.scatter(x2plot,y2plot,marker='o',s=40.,c='r',alpha=1.0) # red

#plt.scatter(x3plot,y3plot,marker='o',s=40.,c='#D3D3D3',alpha=1.0) # lightgrey
#plt.scatter(x3plot,y3plot,marker='o',s=40.,c='#4169E1',alpha=1.0) # royalblue
plt.scatter(x3plot,y3plot,marker='o',s=40.,c='c',alpha=1.0) # cyan

#plt.scatter(x4plot,y4plot,marker='o',s=40.,c='#F5F5F5',alpha=1.0) # whitesmoke
#plt.scatter(x4plot,y4plot,marker='o',s=40.,c='#ADD8E6',alpha=1.0) # lightblue
plt.scatter(x4plot,y4plot,marker='o',s=40.,c='m',alpha=1.0) # Magenta

plt.savefig('%s/%s_Eperp_plot.png'%(opath,basename), format="png", dpi=300)
