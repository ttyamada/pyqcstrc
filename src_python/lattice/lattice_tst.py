import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnndarray as qna
from lattice import (lattice_init,get_tr)

# for test
#if __name__ == '__main__':
isys=3  # for decagonal
brv='p'
crs.crsys_init(isys)
qnn.qnnum_init()
qna.qnndarray_init()
qnv.qnvec_init()
lattice_init(brv)

tr=get_tr('p')
for i in range(tr.shape[0]):
    qnv.printqnv("tr in p",tr[i])
n=6
tr=get_tr('i')
for i in range(tr.shape[0]):
    qnv.printqnv("tr in i",tr[i])

tr=get_tr('f')
for i in range(tr.shape[0]):
    qnv.printqnv("tr in f",tr[i])
  
