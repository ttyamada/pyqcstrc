#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

#import timeit
#import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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


#basename='local_env_vertex'
#basename='local_env_vertex_phase1'
basename='local_env_vertex_phase-sigma'

opath='./test5'

#h1max=6
h1max=5
#h1max=3
h5max=0 # 0th layer

#===============================
# Drawing Tilling
#===============================
plt.figure(figsize=(8, 8))
#plt.axis('equal')
xylim=7.75
plt.xlim([-xylim, xylim])
plt.ylim([-xylim, xylim])

fname='%s/%s_hmax%d'%(opath,basename,h1max)
f=read_file('%s.xyz'%fname)
xeplot=[]
yeplot=[]
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
        xeplot.append(float(a1[1]))
        yeplot.append(float(a1[2]))
        if a1[0]=='Aa':
            x1plot.append(float(a1[1]))
            y1plot.append(float(a1[2]))
        elif a1[0]=='Bb':
            x2plot.append(float(a1[1]))
            y2plot.append(float(a1[2]))
        elif a1[0]=='Cc':
            x3plot.append(float(a1[1]))
            y3plot.append(float(a1[2]))
        else:
            x4plot.append(float(a1[1]))
            y4plot.append(float(a1[2]))
for i1 in range(0,len(xeplot)-1):
    for i2 in range(i1+1,len(xeplot)):
        distance=np.sqrt((xeplot[i1]-xeplot[i2])**2+(yeplot[i1]-yeplot[i2])**2)
        if abs(distance-1.0)<EPS:
           plt.plot([xeplot[i1],xeplot[i2]],[yeplot[i1],yeplot[i2]],lw=0.5,c='k')
#plt.scatter(xeplot,yeplot,marker='o',s=1.,c='r',alpha=1.0)

#plt.scatter(x1plot,y1plot,marker='o',s=40.,c='#000000',alpha=1.0) # black
#plt.scatter(x1plot,y1plot,marker='o',s=40.,c='#00008B',alpha=1.0) # darkblue
plt.scatter(x1plot,y1plot,marker='o',s=100.,c='b',alpha=1.0) # blue

#plt.scatter(x2plot,y2plot,marker='o',s=40.,c='#808080',alpha=1.0) # gray
#plt.scatter(x2plot,y2plot,marker='o',s=40.,c='#0000FF',alpha=1.0) # blue
plt.scatter(x2plot,y2plot,marker='o',s=100.,c='pink',alpha=1.0) # pink

#plt.scatter(x3plot,y3plot,marker='o',s=40.,c='#D3D3D3',alpha=1.0) # lightgrey
#plt.scatter(x3plot,y3plot,marker='o',s=40.,c='#4169E1',alpha=1.0) # royalblue
plt.scatter(x3plot,y3plot,marker='o',s=100.,c='r',alpha=1.0) # red

#plt.scatter(x4plot,y4plot,marker='o',s=40.,c='#F5F5F5',alpha=1.0) # whitesmoke
#plt.scatter(x4plot,y4plot,marker='o',s=40.,c='#ADD8E6',alpha=1.0) # lightblue
plt.scatter(x4plot,y4plot,marker='o',s=100.,c='lime',alpha=1.0) # lime

plt.savefig('%s/%s_Epar_layer%d.png'%(opath,basename,h5max), format="png", dpi=300)
