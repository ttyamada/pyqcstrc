import cython

import crsys
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnndarray as qna
import prjop as prj
import qnsym
from sitesym import (sitesym_init,\
                     site_symmetry,\
                     coset,\
                     equivalent_positions_in_unit_cell,\
                     equivalent_positions
                     )

# for test
#if __name__ == '__main__':
isys=3  # decagonal
crsys.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qna.qnndarray_init()
qnm.qnmat_init()
prj.prjop_init()
qnsym.qnsym_init()

brv='p'
N=crsys.N
n=crsys.n
M0=qnn.zero()         # 0
M1=qnn.Qnnum([1,0,2]) # 1/2
x0=qnv.anyv([M1,M0,M0,M0,M0])
qnv.printqnv("x0",x0)
#qnr=qns.qnr # symmetry operators
qns=qnsym.Qnsym()
nr=qns.nr
brv="p"
irs0=site_symmetry(x0,qns,brv)
r=qns.r # symmetry operators
print("irs0",irs0)
isk0=coset(irs0)
r0=qns.get_qnr0(r,nr)

xeq0=equivalent_positions(x0,brv,isk0,r0)
qnv.printqnv("xeq0",xeq0)

xeq1=equivalent_positions_in_unit_cell(x0,brv,isk0,r0)
qnv.printqnv("xeq1",xeq1)

x1=qnv.zerov(n)           #(0,0,0,0,0)
x1[0]=qnn.any([1,0,2])  #(1/2,0,0,0,0)
x1[1]=qnn.any([1,0,2])
irs1=site_symmetry(x1,qns,brv) 
isk1=coset(irs1)
xeq2=equivalent_positions(x1,brv,isk1,r0)
qnv.printqnv("xeq2",xeq2)

x2=qnv.zerov(n)           #(0,0,0,0,0)
x2[0]=qnn.any([1,0,1])  #(1/2,0,0,0,0)
x2[1]=qnn.any([1,0,2])
irs2=site_symmetry(x2,qns,brv) 
isk2=coset(irs2)
xeq3=equivalent_positions(x2,brv,isk2,r0)
qnv.printqnv("xeq3",xeq3)

x3=qnv.zerov(n)           #(0,0,0,0,0)
x3[0]=qnn.any([1,0,1])  #(1/2,0,0,0,0)
x3[1]=qnn.any([1,0,2])
x3[4]=qnn.any([1,0,4])
irs3=site_symmetry(x3,qns,brv) 
isk3=coset(irs3)
xeq4=equivalent_positions(x3,brv,isk3,r0)
qnv.printqnv("xeq4",xeq4)
    
    