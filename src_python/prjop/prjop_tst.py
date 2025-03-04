#if __name__ == '__main__':
import sys
import numpy as np
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnmath as qmt
import qnndarray as qna
import prjop as prj
from prjop import (prjop_init)


# test for qnnum projection operators
def prj_tst(isys):
    prjop_init(isys)
    print("isys",isys)
    prj0=prj.prj0
    prji=prj.prji
    qnm.printqnm("prj0",prj0)
    #prji=qmt.qnmatinv(prj0,n)
    qnm.printqnm("prji",prji)
    unitm=prji@prj0
    qnm.printqnm("untm",unitm)
    
    prjf=prj.qnm2flnm(prj0)
    n=5
    prj.printfm("prjf",prjf,n)
    #prjif=qnm2flnm(prji)
    prjif=qmt.matinv_f(prjf,n)
    #prji3f=np.linalg.inv(prj3f)
    prj.printfm("prjif",prjif,n)
    unitmf=prjif@prjf
    prj.printfm("unitmf",unitmf,n)



prj_tst(4)
prj_tst(3)
prj_tst(5)
prj_tst(2)
#N=2
#prj4=Qnprj_Octa() # qnnum projection operator
#qnm.printqnm("prj4",prj4)  #
#qna.printqndm("prj4",prj4)  #

#N=5
#prj3=Qnprj_Deca() # float projection operator
#qnm.printqnm("prj3",prj3)  #
#qna.printqndm("prj3",prj3)  #

#N=3
#prj5=Qnprj_Dode() # float projection operator
#qnm.printqnm("prj5",prj3)  #
#qna.printqndm("prj5",prj3)  #

#N=5
#prj2=Qnprj_Icos() # float projection operator
#qnm.printqnm("prj2",prj2)  #
#qna.printqndm("prj2",prj2)  #

# check qnmatinv
#N=2
#n=5
#prj3=Qnprj_Octa()
#qnm.printqnm("prj3",prj3)  #

#prj3f=qnm2flnm(prj3)
#printfm("prj3f",prj3f,n)

#prji3f=qmt.matinv_f(prj3f,n)
#printfm("prji3f",prji3f,n)
#unitmf=prji3f@prj3f
#printfm("unitmf",unitmf,n)

#prji3=qmt.qnmatinv(prj3,n)
#qnm.printqnm("prji3",prji3)
#unitm=prji3@prj3
#qnm.printqnm("untm",unitm)
    
    

                    
                    
    
        
