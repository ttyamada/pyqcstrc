import sys
import numpy as np
import cython

import crsys
import qnnum as qnn
import qnvec as qnv
import qnndarray as qna

def lattice_init():
    global n,N
    n=crsys.n
    N=crsys.N

def get_tr(brv):
    global ntr
    qn0=qnn.Qnnum([0,0,1]) # 0
    qn1=qnn.Qnnum([1,0,2]) # 1/2
    if brv=='p':
        ntr=1
        tr=np.zeros((ntr,n),dtype=qnn.Qnnum)
        if n==5: # dihedral
            tr[0]=[qn0,qn0,qn0,qn0,qn0]
        elif n==6: # icosahedral
            tr[0]=[qn0,qn0,qn0,qn0,qn0,qn0]
    elif brv=='i': # only icosahedral
        ntr=2
        tr=np.zeros((ntr,n),dtype=qnn.Qnnum)
        tr[0]=[qn0,qn0,qn0,qn0,qn0,qn0]
        tr[1]=[qn1,qn1,qn1,qn1,qn1,qn1]
    elif brv=='f':  # only icosahedral
        ntr=32
        tr=np.zeros((ntr,n),dtype=qnn.Qnnum)
        tr[0]=[qn0,qn0,qn0,qn0,qn0,qn0]
        tr[1]=[qn1,qn1,qn0,qn0,qn0,qn0]
        tr[2]=[qn1,qn0,qn1,qn0,qn0,qn0]
        tr[3]=[qn1,qn0,qn0,qn1,qn0,qn0]
        tr[4]=[qn1,qn0,qn0,qn0,qn1,qn0]
        tr[5]=[qn1,qn0,qn0,qn0,qn0,qn1]
        tr[6]=[qn0,qn1,qn1,qn0,qn0,qn0]
        tr[7]=[qn0,qn1,qn0,qn1,qn0,qn0]
        tr[8]=[qn0,qn1,qn0,qn0,qn1,qn0]
        tr[9]=[qn0,qn1,qn0,qn0,qn0,qn1]
        tr[10]=[qn0,qn0,qn1,qn1,qn0,qn0]
        tr[11]=[qn0,qn0,qn1,qn0,qn1,qn0]
        tr[12]=[qn0,qn0,qn1,qn0,qn0,qn1]
        tr[13]=[qn0,qn0,qn0,qn1,qn1,qn0]
        tr[14]=[qn0,qn0,qn0,qn1,qn0,qn1]
        tr[15]=[qn0,qn0,qn0,qn0,qn1,qn1]
        tr[16]=[qn1,qn1,qn1,qn1,qn0,qn0]
        tr[17]=[qn1,qn1,qn1,qn0,qn1,qn0]
        tr[18]=[qn1,qn0,qn0,qn1,qn1,qn0]
        tr[19]=[qn1,qn0,qn1,qn1,qn1,qn0]
        tr[20]=[qn0,qn1,qn1,qn1,qn1,qn0]
        tr[21]=[qn1,qn1,qn1,qn0,qn0,qn1]
        tr[22]=[qn1,qn1,qn0,qn1,qn0,qn1]
        tr[23]=[qn1,qn0,qn1,qn1,qn0,qn1]
        tr[24]=[qn0,qn1,qn1,qn1,qn0,qn1]
        tr[25]=[qn1,qn1,qn0,qn0,qn1,qn1]
        tr[26]=[qn1,qn0,qn1,qn0,qn1,qn1]
        tr[27]=[qn0,qn1,qn1,qn0,qn1,qn1]
        tr[28]=[qn1,qn0,qn0,qn1,qn1,qn1]
        tr[29]=[qn0,qn1,qn0,qn1,qn1,qn1]
        tr[30]=[qn0,qn0,qn1,qn1,qn1,qn1]
        tr[31]=[qn1,qn1,qn1,qn1,qn1,qn1]
    return tr
        
      