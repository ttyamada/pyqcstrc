import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import prjop as prj
import qnmath as qmt
import qnndarray as qna
import qnsym as qns
import lattice as lt
from numpy.typing import(NDArray)

def sitesym_init():
    global n,N
    n=crs.n
    N=crs.N

def site_symmetry(x: qnv.Qnvec) -> np.ndarray: # return irs
    global nr,mpltbl,r,rt
    """symmetry operator indeces irs in the site symmetry group G.
    
    Args:
        x (qnndarray):  -> (5D or 6D) qnnum lattice coordinates
 
    Returns:
        List of index of symmetry operators of the site symmetry group G (list):
            The symmetry operators leaves xyz identical.
    """
    if crs.isys==2:
        n_i=3
    else:
        n_i=2
    n=crs.n
    brv=lt.brv
    nr=qns.nr
    #print("n",n,"n_i",n_i) # for test
    r_qn=qns.r_qn  # symmetry operator for Q coordinates
    r_qn_i=qns.r_qn_i  # symmetry operator for Q coordinates
    mpltbl=qns.mpltbl
    r=qns.r  # integer rotation matrix
    rt=qns.rt # its transverse matrix
    #a_i=qnv.zerovs((nr,n_i))
    #qnx_i=prj.prjvec_i(x) # internal space component os nD vector x
    a=np.zeros((nr,n),dtype=qnn.Qnnum)
    #a=qnv.zerovs((nr,n))
    #qnx=prj.prjvec(x)
    #b=qnv.zerov(n)

    irs=np.zeros(0,dtype=np.int64)
    #tr=lt.get_tr(brv)
    trop=lt.get_tr()  # centering translation vectors including zero vector
    #print("type(trop[0])",type(trop[0]))  # for test
    for i in range(nr):
        a[i]=x@rt[i]  #r_[i]@x   # r (int) x qnv -> x x rt (int)
    reduce_x(a,brv) # -1/2 <=x[i] < 2/1
    for i in range(nr):
        if a[i]==x:
            irs=np.append(irs,i)
    #print('irs',irs)
    return irs

# new symmetry operator index
def newl(ics,ns0):
    #print("ics",ics,"ns0",ns0) # for test
    for i in range(ns0):
        if i not in ics:
            return i
    print("new i not found")
    print("ics",ics)
    exit()
 