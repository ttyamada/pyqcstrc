import cython

import crsys
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnndarray as qna
import prjop as prj
import qnmath as qmt
import qnndarray as qna
from qnsym import (qnsym_init,Qnsym,test_wt)

def qnsym_tst_init(isys):
    crsys.crsys_init(isys)
    print("crsys.isys",crsys.isys)  # for test
    qnn.qnnum_init()
    qnv.qnvec_init()
    qnm.qnmat_init()
    prj.prjop_init()
    qna.qnndarray_init()
    qmt.qnmath_init()
    qnsym_init()

# for test
#if __name__ == '__main__':
# test for qnnum projection operators

isys=3 # for decagonal
qnsym_tst_init(isys)
qns3=Qnsym()
test_wt("Deca",qns3)

isys=4 # for octabonal
qnsym_tst_init(isys)
qns4=Qnsym()
test_wt("Octa",qns4)

isys=5 # for dodecagonal
qnsym_tst_init(isys)
qns5=Qnsym()
test_wt("Dode",qns5)

isys=2 # for icosahedral
qnsym_tst_init(isys)
qns2=Qnsym() #qns2=qnsym_init(isys)
test_wt("Icos",qns2)
    

