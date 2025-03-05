import cython

import qnvec as qnv
from lattice import (get_tr)

# for test
#if __name__ == '__main__':
N=5  # decagonal or icosahedral
n=5
tr=get_tr('p',n,N)
qnv.printqnv("tr in p",tr)
n=6
tr=get_tr('i',n,N)
qnv.printqnv("tr in i",tr)

tr=get_tr('f',n,N)
qnv.printqnv("tr in f",tr)
  
