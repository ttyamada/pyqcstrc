import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnndarray as qna
import qnvec as qnv
import qnmat as qnm
import prjop as prj
import qnmath as qmt
from numpy.typing import(NDArray)

def qnsym_init():
    global n,N,isys,r_qns,r,rt
    isys=crs.isys
    n=crs.n
    N=crs.N
    print("qnsym_init isys",isys,"n",n,"N",N)  # for test
    r_qn=Qnsym()  # set symmetry operator
    r=r_qn.r
    rt=r_qn.rt
    print_r(r_qn.r)  # for test
    test_wt_r_qn("r_qn",r_qn)  # for test
    test_wt_r_qn_e("r_qn_e",r_qn)  # for test

def Qnsym():
    global r_qn0,r_qn,r_qn_e,r_qn_i,mpltbl
    if isys==2:
        qns=Qnsym_Icos() #Pn35
    elif isys==3:
        qns=Qnsym_Deca() #P10mm
    elif isys==4:
        qns=Qnsym_Octa() #P8mm
    elif isys==5:
        qns=Qnsym_Dode() #P12mm
    else:
        print("isys should be 2,3,4 or 5 but",isys)
        exit()
    
    r_qn0=qns.r_qn0
    r_qn=qns.r_qn
    r_qn_e=qns.r_qn_e
    r_qn_i=qns.r_qn_i
    mpltbl=qns.mpltbl
    return qns
    
def rtor_qn(r):
    shape=r.shape # (nr,n,n)
    nr=shape[0]
    prj0_=prj.prj0
    prji_=prj.prji
    return get_r_qn(prji_,prj0_,r,nr)
    #prj0t_=prj.prj0t
    #prjit_=prj.prjit
    #return get_r_qn(prj0t_,prjit_,r,nr)

def rtor_qn_e(r):  # first 2x2 diaglnal block
    qr=rtor_qn(r)
    if isys==2:
        return qr[:,0:3,0:3] #3x3 diagonal block 
    else:
        return qr[:,0:2,0:2] #2x2 diagonal block not correct at the moment

def rtor_qn_i(r): # second 2x2 giagonal block
    qr=rtor_qn(r)
    #n_=r.shape[0]
    if isys==2:
        return qr[:,3:6,3:6] # 3x3 second diagonal block
    else:
        return qr[:,2:4,2:4] # 2x2 second diagonal block
    
def is_equal(r1:NDArray[np.int64],r2:NDArray[np.int64]):
    for i in range(n):
        for j in range(n):
            if r1[i][j]!=r2[i][j]:
                return False
    return True
        
def set_mpltbl(mpltbl:NDArray[np.int64],r:NDArray[np.int64]): # r: integer rotation matrices in nD lattice
    rt_=np.zeros((n,n),dtype=np.int64)
    for i in range(nr):
        for j in range(nr):
            rt_=r[i]@r[j] # integer matrix
            for k in range(nr):
                if is_equal(rt_,r[k]):
                    mpltbl[i][j]=k
                    break
    wt_mpltbl(mpltbl) # for test

def wt_mpltbl(mpltbl: np.ndarray):
    n_=(int)(shape[0]/2) # when centrosymmetric
    print("mpltbl 1st block")
    for i in range(n_):
        for j in range(n_):
            print("","{:2d}".format(mpltbl[i][j]),end="")
        print("")
    
    print("mpltbl 2nd block")
    for i in range(n_):
        for j in range(n_):
            print("",mpltbl[i][j+n_],end="")
        print("")
  