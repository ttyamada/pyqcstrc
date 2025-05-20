import timeit
import os
import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnndarray as qna
import qnvec as qnv
import qnmat as qnm
import numeric as num
import utils as utl
import vesta as vst
import qnsym as qns
import lattice as lt
import sitesym as ssm
import intsct as isct
import prjop as prj
from occdom import (occdom_init,symmetric,write,shift)
import tr6to5 as tr5
import tr7to5 as tr7
from numpy.typing  import (NDArray)

def get_triangles(od_sym,odsym1):
    nod=0
    ntr=0
    for i,tri1 in enumerate(od_sym):
        for j,tri2 in enumerate(od_sym1): 
            nx,x=isct.intersection_two_triangles(od_sym[i],od_sym1[j])
            qnv.printqnvs("cross points of two triangles",x)
            ny,y=isct.common_points(od_sym[i],od_sym1[j])
            qnv.printqnvs("common points in triangles",y)
            if nx==0 and ny==0:
                continue
            n,z=isct.common_part(nx,x,ny,y)
            print("z.shape",z.shape)  # for test
            qnv.printqnvs("common part",z)  # for test

            n,z=isct.rmv_overlapedx(z,n)
            print("number of vertices in common part",n)
            qnv.printqnvs("common part",z)  # for test
            if n>=3:
                nod+=1
                print("nod",nod)
            if n==3: # triangle
                # stack triangle here
                if ntr==0:
                    tria=z
                else:
                    tria=np.vstack([tria,z])
                ntr+=1
    return tria, ntr
