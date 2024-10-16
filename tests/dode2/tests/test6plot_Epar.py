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


basename='deco'

opath='./test6'

#h1max=6
#h1max=5
h1max=3
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
x1plot=[]
y1plot=[]
x2plot=[]
y2plot=[]
for i1 in range(2,len(f)):
    fi1=f[i1]
    a1=fi1.split()
    if int(a1[12])==h5max:
        if a1[0]=='Xx':
            x1plot.append(float(a1[1]))
            y1plot.append(float(a1[2]))
        else:
            x2plot.append(float(a1[1]))
            y2plot.append(float(a1[2]))
for i1 in range(0,len(x1plot)-1):
    for i2 in range(i1+1,len(x1plot)):
        distance=np.sqrt((x1plot[i1]-x1plot[i2])**2+(y1plot[i1]-y1plot[i2])**2)
        if abs(distance-1.0)<EPS:
           plt.plot([x1plot[i1],x1plot[i2]],[y1plot[i1],y1plot[i2]],lw=0.5,c='k')
#plt.scatter(xeplot,yeplot,marker='o',s=1.,c='r',alpha=1.0)

#plt.scatter(x1plot,y1plot,marker='o',s=40.,c='#000000',alpha=1.0) # black
plt.scatter(x1plot,y1plot,marker='o',s=40.,c='#00008B',alpha=1.0) # darkblue

#plt.scatter(x2plot,y2plot,marker='o',s=40.,c='#808080',alpha=1.0) # gray
plt.scatter(x2plot,y2plot,marker='o',s=40.,c='#0000FF',alpha=1.0) # blue

plt.savefig('%s/%s_Epar_layer%d.png'%(opath,basename,h5max), format="png", dpi=300)
