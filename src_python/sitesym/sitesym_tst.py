import cython
import numpy as np

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnndarray as qna
import prjop as prj
import lattice as lt
import qnsym as qns

from sitesym import (sitesym_init,\
                     site_symmetry,\
                     coset,\
                     equivalent_positions_reduced,\
                     equivalent_positions
                     )

# for test
#if __name__ == '__main__':

isys=3  # decagonal
crs.crsys_init(isys)
brv='p'
qnn.qnnum_init()
qnv.qnvec_init()
qna.qnndarray_init()
qnm.qnmat_init()
prj.prjop_init()
qns.qnsym_init()
lt.lattice_init(brv)
sitesym_init()

M0=qnn.zero()         # 0
M1=qnn.Qnnum([2,0,5]) # B at (2 2 2 2 0)/5 
x00=np.array([M1,M1,M1,M1,M0],dtype=qnn.Qnnum)  # lattice coordinates for B
x0=qnv.anyv(x00)
qnv.printqnv("x0",x0)
#qnr=qns.qnr # symmetry operators
qns=qns.Qnsym()
nr=qns.nr
brv=lt.brv
irs0=site_symmetry(x0) # use lattice coordinates
isk0=coset(irs0)
#xeq0=equivalent_positions(x0,brv,isk0,qns.rt)
#qnv.printqnv("xeq0",xeq0)

xeq1=equivalent_positions_reduced(x0,brv,isk0,qns.rt)
#qnv.printqnv("xeq1",xeq1)

n=crs.n
M1=qnn.any([1,0,2])  #(1/2,0,0,1/2,0)
x10=np.array([M1,M0,M0,M1,M0],dtype=qnn.Qnnum)
x1=qnv.anyv(x10)
qnv.printqnv("x1",x1)
irs1=site_symmetry(x1) 
isk1=coset(irs1)
xeq2=equivalent_positions(x1,brv,isk1,qns.rt)
#qnv.printqnv("xeq2",xeq2)

x20=np.array([M0,M1,M1,M0,M0],dtype=qnn.Qnnum)
x2=qnv.anyv(x20)
qnv.printqnv("x2",x2)
irs2=site_symmetry(x2) 
isk2=coset(irs2)
#xeq3=equivalent_positions(x2,brv,isk2,qns.rt)
