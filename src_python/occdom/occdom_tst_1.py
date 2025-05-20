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

def set_primod4():
    M0=qnn.Qnnum([0,0,1])  # 0
    M1=qnn.Qnnum([1,0,2])  # 1/2
    M2=-M1                 #-1/2
    oc_asym0_=np.array([\
        [M0,M0,M0,M0,M0],\
        [M1,M2,M0,M1,M0],\
        [M1,M2,M2,M1,M0],\
        ],dtype=qnn.Qnnum)
    od_asym_nd=qna.anya(oc_asym0_,(3,5))
    od_asym=od_asym_nd.reshape((1,3,5))
    return od_asym

def set_primod3():
    M0=qnn.Qnnum([0,0,1])  # 0
    M1=qnn.Qnnum([1,0,1])  # 1
    M2=-M1                 #-1
    M3=qnn.Qnnum([1,0,2])  # 1/2
    od_asym_=np.array([\
        [M0,M0,M0,M0,M0,M0],\
        [M1,M0,M0,M0,M0,M0],\
        [M3,M0,M0,M3,M0,M0]\
        ],dtype=qnn.Qnnum)
    od_asym_nd=qna.anya(od_asym_,(3,6))
    od_asym=tr5.tr6to5e(od_asym_nd).reshape(1,3,5)
    return od_asym

def set_primod5():
    M0=qnn.Qnnum([0,0,1])  # 0
    M1=qnn.Qnnum([1,0,1])  # 1
    M2=-M1                 #-1
    M3=qnn.Qnnum([0,1,3])  # sqrt(3)/3=1/sqrt(3)
    od_asym_=np.array([\
        [M0,M0,M0,M0,M0,M0,M0],\
        [M1,M0,M0,M0,M0,M0,M0],\
        [M3,M0,M0,M0,M0,M3,M0]\
        ],dtype=qnn.Qnnum) # stampfli tiling
    od_asym_nd=qna.anya(od_asym_,(3,7))
    od_asym=tr7.tr7to5e(od_asym_nd).reshape(1,3,5)
    return od_asym

def program_init(isys):
    #isys=4 # for octagonal
    crs.crsys_init(isys)
    qnn.qnnum_init()
    qna.qnndarray_init()
    qnv.qnvec_init()
    qnm.qnmat_init()
    prj.prjop_init()
    qns.qnsym_init()
    lt.lattice_init('p')
    ssm.sitesym_init()

    occdom_init()

    #test_dir='../../tests/octa2/tests'
    #xyz_dir='../../xyz/octa'
    if isys==4:  # octagonal
        od_asym=set_primod4()
    elif isys==3:  # decagonal
        od_asym=set_primod3()
    elif isys==5:  # dodecagonal
        od_asym=set_primod5()
    else:
        print("isys=2 (icosahedral) not implemented yet")
        exit()

    # if od_asym is represented by internal space compoenet of triangles/tetrahedra
    # its symmetric version is obtained by the symmetry operator in the internal space
    # this simplifies later calculations

    # transform od_asym nD vertex coordinates to its internal space components
    od_asym=prj.projection3_sets_numerical(od_asym)
    return od_asym
