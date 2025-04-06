#if __name__ == '__main__':
import sys
import numpy as np
import cython

import crsys
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnmath as qmt
import qnndarray as qna
import prjop as prj
#from prjop import (prjop_init, Prjop,\
#                    qnm2flnm, printfm,tstwt_prjop)

# test for qnnum projection operators
def prj_tst(isys: np.int64):
    print("isys",isys)
    crsys.crsys_init(isys)
    qnn.qnnum_init()
    qnv.qnvec_init()
    qnm.qnmat_init()
    
    prj.prjop_init()
    prj.tstwt_prjop()
    
    prj0=prj.prj0
    prji=prj.prji
    unitm=prji@prj0
    qnm.printqnm("unitm",unitm)
    
    prj0f=prj.qnm2flnm(prj0)
    M0=qnn.Qnnum([0,0,1])
    M1=qnn.Qnnum([1,0,1])
    if isys==2:
        n=6
        x=np.array([M1,M0,M0,M0,M0,M0])
        qnx=qnv.anyv(x)
    else:
        n=5
        x=np.array([M1,M0,M0,M0,M0])
        qnx=qnv.anyv(x)

    prj.printfm("prj0f",prj0f,n)
    prjif=qmt.matinv_f(prj0f,n)
    prj.printfm("prjif",prjif,n)
    unitmf=prjif@prj0f
    prj.printfm("unitmf",unitmf,n)

    qnv.printqnv("qnx",qnx)
    qnei=prj.prjvec(qnx)
    qne=prj.prjvec_e(qnx)
    qni=prj.prjvec_i(qnx)
    qnv.printqnv("qnei",qnei)
    qnv.printqnv("qne",qne)
    qnv.printqnv("qni",qni)

prj_tst(4)
prj_tst(3)
prj_tst(5)
prj_tst(2)
