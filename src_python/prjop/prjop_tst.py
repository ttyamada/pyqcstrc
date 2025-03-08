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
                    qnm2flnm, printfm)

# test for qnnum projection operators
def prj_tst(isys: np.int64):
    print("isys",isys)
    crsys.crsys_init(isys)
    qnn.qnnum_init()
    qnv.qnvec_init()
    qnm.qnmat_init()
    
    prjop_init()
    prj=Prjop()
    prj0=prj.prj0
    prji=prj.prji
    qnm.printqnm("prj0",prj0)
    #prji=qmt.qnmatinv(prj0,n)
    qnm.printqnm("prji",prji)
    unitm=prji@prj0
    qnm.printqnm("untm",unitm)
    
    prjf=qnm2flnm(prj0)
    n=5
    printfm("prjf",prjf,n)
    #prjif=qnm2flnm(prji)
    prjif=qmt.matinv_f(prjf,n)
    #prji3f=np.linalg.inv(prj3f)
    prjop.printfm("prjif",prjif,n)
    unitmf=prjif@prjf
    printfm("unitmf",unitmf,n)

prj_tst(4)
prj_tst(3)
prj_tst(5)
prj_tst(2)
