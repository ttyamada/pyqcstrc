import cython

import crsys
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnndarray as qna
import prjop as prj
from qnsym import (qnsym_init,Qnsym,test_wt)

# for test
#if __name__ == '__main__':
# test for qnnum projection operators
isys=4
crsys.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qnm.qnmat_init()
prj.prjop_init()
qna.qnndarray_init()
qnsym_init()
qns4=Qnsym()
test_wt("Octa",qns4)

isys=3
crsys.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qnm.qnmat_init()
prj.prjop_init()
qns3=qnsym_init()
qns3.test_wt("Deca",qns3)

isys=5
crsys.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qnm.qnmat_init()
prj.prjop_init()
qns5=qnsym_init(isys)
qns5.test_wt("Dode",qns5)

isys=2
crsys.crsys_init(isys)
prj2=prj.prjop_init(isys)
qns2=Qnsym(isys) #qns2=qnsym_init(isys)
qns2.test_wt("Icos",qns2)
    

