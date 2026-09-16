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

def print_r0(r0):
    shape=r0.shape
    ndim=r0.ndim
    #print("r.shape",r.shape)
    #print("r.ndim",ndim)
    nr=shape[0]
    #n=shape[1]
    for i in range(nr):
        print("#",i+1)
        qnm.printqnm("",r0[i])

# integer matrix elements to qnnumber matrix elements transformation
def intr2qnmr(r,nr):
    #qnmr=qna.QnNdarray((nr,n,n)) #[qm0]*nr
    qnmr=qnm.Qnmat[nr]
    for i in range(nr):
        qnmr[i]=qnm.intm2qnm(r[i],n) # qnmat for i-th rotation operator r[i]
    return qnmr

def test_wt_r_qn(str:str,qns:qna.QnNdarray):
    nr=qns.nr
    print("nr",nr)
    print(str)
    for i in range(nr):
        str="r_qn["+format(i)+"]"
        qnm.printqnm(str,qns.r_qn[i])
     
def test_wt_r_qn_e(str:str,qns:qna.QnNdarray):
    nr=qns.nr
    print("nr",nr)
    print(str)
    for i in range(nr):
        str="r_qn_e["+format(i)+"]"
        qnm.printqnm(str,qns.r_qn_e[i])   

def test_wt_r_qn_i(str:str,qns:qna.QnNdarray):
    nr=qns.nr
    print("nr",nr)
    print(str)
    for i in range(nr):
        str="r_qn_i["+format(i)+"]"
        qnm.printqnm(str,qns.r_qn_i[i])   
  