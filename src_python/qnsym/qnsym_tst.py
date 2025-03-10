import cython

import crsys
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnndarray as qna
import prjop as prj
import qnmath as qmt
import qnndarray as qna
from qnsym import (qnsym_init,\
                   Qnsym,\
                   test_wt_qnr,\
                   test_wt_qnr_e,\
                   test_wt_qnr_i\
                   )

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
test_wt_qnr("Deca qnr",qns3)
test_wt_qnr_i("Deca qnr_i",qns3)

isys=4 # for octabonal
qnsym_tst_init(isys)
qns4=Qnsym()
test_wt_qnr("Octa qnr",qns4)
test_wt_qnr_i("Octa qnr_i",qns4)

isys=5 # for dodecagonal
qnsym_tst_init(isys)
qns5=Qnsym()
test_wt_qnr("Dode qnr",qns5)
test_wt_qnr_i("Dode qnr_i",qns5)

isys=2 # for icosahedral
qnsym_tst_init(isys)
qns2=Qnsym() #qns2=qnsym_init(isys)
test_wt_qnr("Icos qnr",qns2)
test_wt_qnr_i("Icos qnr_i",qns2)
    

