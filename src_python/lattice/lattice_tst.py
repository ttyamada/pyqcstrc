import cython

import crsys
import lattice as lt
import qnvec as qnv
from lattice import (lattice_init,get_tr)

# for test
#if __name__ == '__main__':
isys=3  # for decagonal
crsys.crsys_init(isys)
lattice_init()

tr=get_tr('p')
qnv.printqnv("tr in p",tr)
n=6
tr=get_tr('i')
qnv.printqnv("tr in i",tr)

tr=get_tr('f')
qnv.printqnv("tr in f",tr)
  
