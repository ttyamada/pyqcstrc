import cython

import crsys
import prjop as prj
from qnsym import (Qnsym,test_wt)

# for test
#if __name__ == '__main__':
# test for qnnum projection operators
isys=4
crsys.crsys_init(isys)
prj4=prj.Prjop(isys)
qns4=Qnsym(isys) #qns4=qnsym_init(isys)
test_wt("Octa",qns4)

isys=3
prj3=prj.Prjop(isys)
qns3=Qnsym(isys) #qns3=qnsym_init(isys)
qbs3.test_wt("Deca",qns3)


isys=5
prj5=prj.Prjop(isys)
qns5=Qnsym(isys) #qns5=qnsym_init(isys)
qns5.test_wt("Dode",qns5)

isys=2
prj2=prj.Prjop(isys)
qns2=Qnsym(isys) #qns2=qnsym_init(isys)
qns2.test_wt("Icos",qns2)
    

