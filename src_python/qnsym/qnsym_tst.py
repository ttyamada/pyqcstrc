# for test
#if __name__ == '__main__':
# test for qnnum projection operators
isys=4
prj4=prj.prjop_init(isys)
qns4=qnsym_init(isys)
test_wt("Octa",qns4)

isys=3
prj3=prj.prjop_init(isys)
qns3=qnsym_init(isys)
test_wt("Deca",qns3)

isys=5
prj5=prj.prjop_init(isys)
qns5=qnsym_init(isys)
test_wt("Dode",qns5)

isys=2
prj2=prj.prjop_init(isys)
qns2=qnsym_init(isys)
test_wt("Icos",qns2)
    

