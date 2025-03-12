import numpy as np

import crsys
import qnnum as qnn
import qnvec as qnv
import qnndarray as qna

from triang import (Qntri,\
                    qntri_init,\
                    zerotri,\
                    zerotris,\
                    anytri,\
                    wt_qntri)

isys=4
crsys.crsys_init(isys)
qntri_init()
qnn.qnnum_init()
qnv.qnvec_init()
M0=qnn.any([0,0,1])
M1=qnn.any([1,0,1])
M2=qnn.any([0,1,1])

v1=np.array([M0,M0])
v2=np.array([M1,M0])
v3=np.array([M0,M2])

qnv1=qnv.anyv(v1)
qnv2=qnv.anyv(v2)
qnv3=qnv.anyv(v3)
qnv.printqnvs("vt",[qnv1,qnv2,qnv3]) 

v=qna.anya(np.array([[M0,M0],[M1,M0],[M0,M2]]),(3,2))

qntri1=anytri(v)
wt_qntri(qntri1)
