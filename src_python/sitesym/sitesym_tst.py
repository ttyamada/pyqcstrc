import cython

import crsys
import qnnum as qnn
import qnvec as qnv
import prjop as prj
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
brv='p'
N=crsys.N
n=crsys.n
prj5=prj.prjop_init()
qns5=qns.qnsym_init()
x0=qnv.zerov(n,N)
x0[0]=qnn.Qnnum([1,0,2],N)  #(1/2,0,0,0,0)
qnv.printqnv("x0",x0)
#qnr=qns.qnr # symmetry operators
nr=qns5.nr
r=qns5.r
#for i in range(nr):
#    qnm.printqnm("qnr[i]",qns5.qnr[i])
sitesym_init()
irs0=site_symmetry(x0,qns5,brv)
print("irs0",irs0)
isk0=coset(irs0)
r0=qns.get_qnr0(r,nr,n,N)

xeq0=equivalent_positions(x0,brv,isk0,r0)
qnv.printqnv("xeq0",xeq0)

xeq1=equivalent_positions_in_unit_cell(x0,brv,isk0,r0)
qnv.printqnv("xeq1",xeq1)

x1=qnv.zerov(n,N)           #(0,0,0,0,0)
x1[0]=qnn.Qnnum([1,0,2],N)  #(1/2,0,0,0,0)
x1[1]=qnn.Qnnum([1,0,2],N)
irs1=site_symmetry(x1,qns5,brv) 
isk1=coset(irs1)
xeq2=equivalent_positions(x1,brv,isk1,r0)
qnv.printqnv("xeq2",xeq2)

x2=qnv.zerov(n,N)           #(0,0,0,0,0)
x2[0]=qnn.Qnnum([1,0,1],N)  #(1/2,0,0,0,0)
x2[1]=qnn.Qnnum([1,0,2],N)
irs2=site_symmetry(x2,qns5,brv) 
isk2=coset(irs2)
xeq3=equivalent_positions(x2,brv,isk2,r0)
qnv.printqnv("xeq3",xeq3)

x3=qnv.zerov(n,N)           #(0,0,0,0,0)
x3[0]=qnn.Qnnum([1,0,1],N)  #(1/2,0,0,0,0)
x3[1]=qnn.Qnnum([1,0,2],N)
x3[4]=qnn.Qnnum([1,0,4],N)
irs3=site_symmetry(x3,qns5,brv) 
isk3=coset(irs3)
xeq4=equivalent_positions(x3,brv,isk3,r0)
qnv.printqnv("xeq4",xeq4)
    
    