import numpy as np

import crsys
import qnnum as qnn
import qnvec as qnv
import qnndarray as qna

from tetrahd import (Qntet,\
                    qntet_init,\
                    zerotet,\
                    zerotets,\
                    anytet,\
                    wt_qntet)

isys=2
crsys.crsys_init(isys)
qntet_init()
qnn.qnnum_init()
qnv.qnvec_init()
M0=qnn.any([0,0,1])
M1=qnn.any([1,0,1])
M2=qnn.any([0,1,1])

v1=np.array([M0,M0,M0])
v2=np.array([M1,M0,M0])
v3=np.array([M0,M2,M0])
v4=np.array([M0,M0,M1])

qnv1=qnv.anyv(v1)
qnv2=qnv.anyv(v2)
qnv3=qnv.anyv(v3)
qnv4=qnv.anyv(v4)
qnv.printqnvs("vt",[qnv1,qnv2,qnv3,qnv4]) 

v=qna.anya(np.array([[M0,M0,M0],[M1,M0,M0],[M0,M2,M0],[M0,M0,M1]]),(4,3))

qntet1=anytet(v)
wt_qntet(qntet1)
