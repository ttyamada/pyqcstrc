# qnnumber version of vesta
# qnnumber should be transformed to float by qnn.qn2frt 

import timeit
import os
import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnndarray as qna
import qnvec as qnv
import qnmath as qmt
import utils as utl
import numeric as num
import intsct as ints
import prjop as prj
import math1 as mth

def generator_off_dim4_triangle(obj,path,filename):
    #Generate object (set of triangles) object in XYZ format.

    f=open('%s/%s.xyz'%(path,filename),'w', encoding="utf-8", errors="ignore")
    f.write('%d\n'%(len(obj)*3))
    f.write('%s\n'%(filename))
    i1=0
    n=crs.n
    isys=crs.isys
    if isys>2:
        ni=2
    else:
        ni=3
    scly=crs.scly
    for i1,triangle in enumerate(obj):  # i1-th triangle
        for i2,vt in enumerate(triangle): # i2-th vertex
            vi=vt
            #qnv.printqnv("vi",vi)  # for test
            vif=qnv.qnv2flt(vi)

            if n==5:
                f.write('Xx %8.6f %8.6f %8.6f'%(vif[0],vif[1],0.0))
            else:
                f.write('Xx %8.6f %8.6f %8.6f'%(vif[0],vif[1],vif[2]))
            f.write(' # %s-th triangle %s-th vertex '%(i1,i2))
            f.write(' #')
            for i in range(ni-1):
                f.write('  %d %d %d'%(vt[i].n[0],vt[i].n[1],vt[i].n[2]))
            f.write('  %d %d %d\n'%(vt[ni-1].n[0],vt[ni-1].n[1],vt[ni-1].n[2]))

    v=utl.obj_area_nd(obj)
    f.write('volume = %d %d %d (%8.6f)\n'%(v.n[0],v.n[1],v.n[2],qnn.qn2flt(v)))
    for i1,triangle in enumerate(obj):
        v=utl.triangle_area_nd(triangle)
        f.write('%3d-the triangle, %d %d %d (%8.6f)\n'\
                %(i1,v.n[0],v.n[1],v.n[2],qnn.qn2flt(v)))
    f.closed
    return 0

 