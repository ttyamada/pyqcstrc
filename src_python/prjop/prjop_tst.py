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
from prjop import (prjop_init, Prjop,\
                    qnm2flnm, printfm,tstwt_prjop)

# test for qnnum projection operators
def prj_tst(isys: np.int64):
    print("isys",isys)
    crsys.crsys_init(isys)
    qnn.qnnum_init()
    qnv.qnvec_init()
    qnm.qnmat_init()
    
    prjop_init()
    prj=Prjop()
    tstwt_prjop()
    
    prj0=prj.prj0
    prji=prj.prji
    unitm=prji@prj0
    qnm.printqnm("unitm",unitm)
    
    prj0f=qnm2flnm(prj0)
    if isys==2:
        n=6
    else:
        n=5
    printfm("prj0f",prj0f,n)
    prjif=qmt.matinv_f(prj0f,n)
    printfm("prjif",prjif,n)
    unitmf=prjif@prj0f
    printfm("unitmf",unitmf,n)

prj_tst(4)
prj_tst(3)
prj_tst(5)
prj_tst(2)
