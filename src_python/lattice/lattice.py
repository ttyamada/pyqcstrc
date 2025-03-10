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

def get_tr(brv) -> qnv.Qnvec:
    global ntr
    qn0=qnn.Qnnum([0,0,1]) # 0
    qn1=qnn.Qnnum([1,0,2]) # 1/2
    if brv=='p':
        ntr=1
        tr=qnv.zerovs((ntr,n))
        if n==5: # dihedral
            tr[0]=qnv.anyv(np.array([qn0,qn0,qn0,qn0,qn0]))
        elif n==6: # icosahedral
            tr[0]=qnv.anyv(np.array([qn0,qn0,qn0,qn0,qn0,qn0]))
    elif brv=='i': # only icosahedral
        ntr=2
        tr=qnv.zerovs((ntr,n))
        tr[0]=qnv.anyv(np.array([qn0,qn0,qn0,qn0,qn0,qn0]))
        tr[1]=qnv.anyv(np.array([qn1,qn1,qn1,qn1,qn1,qn1]))
    elif brv=='f':  # only icosahedral
        ntr=32
        tr=qnv.zerovs((ntr,n))
        tr[0]=qnv.anyv(np.array([qn0,qn0,qn0,qn0,qn0,qn0]))
        tr[1]=qnv.anyv(np.array([qn1,qn1,qn0,qn0,qn0,qn0]))
        tr[2]=qnv.anyv(np.array([qn1,qn0,qn1,qn0,qn0,qn0]))
        tr[3]=qnv.anyv(np.array([qn1,qn0,qn0,qn1,qn0,qn0]))
        tr[4]=qnv.anyv(np.array([qn1,qn0,qn0,qn0,qn1,qn0]))
        tr[5]=qnv.anyv(np.array([qn1,qn0,qn0,qn0,qn0,qn1]))
        tr[6]=qnv.anyv(np.array([qn0,qn1,qn1,qn0,qn0,qn0]))
        tr[7]=qnv.anyv(np.array([qn0,qn1,qn0,qn1,qn0,qn0]))
        tr[8]=qnv.anyv(np.array([qn0,qn1,qn0,qn0,qn1,qn0]))
        tr[9]=qnv.anyv(np.array([qn0,qn1,qn0,qn0,qn0,qn1]))
        tr[10]=qnv.anyv(np.array([qn0,qn0,qn1,qn1,qn0,qn0]))
        tr[11]=qnv.anyv(np.array([qn0,qn0,qn1,qn0,qn1,qn0]))
        tr[12]=qnv.anyv(np.array([qn0,qn0,qn1,qn0,qn0,qn1]))
        tr[13]=qnv.anyv(np.array([qn0,qn0,qn0,qn1,qn1,qn0]))
        tr[14]=qnv.anyv(np.array([qn0,qn0,qn0,qn1,qn0,qn1]))
        tr[15]=qnv.anyv(np.array([qn0,qn0,qn0,qn0,qn1,qn1]))
        tr[16]=qnv.anyv(np.array([qn1,qn1,qn1,qn1,qn0,qn0]))
        tr[17]=qnv.anyv(np.array([qn1,qn1,qn1,qn0,qn1,qn0]))
        tr[18]=qnv.anyv(np.array([qn1,qn0,qn0,qn1,qn1,qn0]))
        tr[19]=qnv.anyv(np.array([qn1,qn0,qn1,qn1,qn1,qn0]))
        tr[20]=qnv.anyv(np.array([qn0,qn1,qn1,qn1,qn1,qn0]))
        tr[21]=qnv.anyv(np.array([qn1,qn1,qn1,qn0,qn0,qn1]))
        tr[22]=qnv.anyv(np.array([qn1,qn1,qn0,qn1,qn0,qn1]))
        tr[23]=qnv.anyv(np.array([qn1,qn0,qn1,qn1,qn0,qn1]))
        tr[24]=qnv.anyv(np.array([qn0,qn1,qn1,qn1,qn0,qn1]))
        tr[25]=qnv.anyv(np.array([qn1,qn1,qn0,qn0,qn1,qn1]))
        tr[26]=qnv.anyv(np.array([qn1,qn0,qn1,qn0,qn1,qn1]))
        tr[27]=qnv.anyv(np.array([qn0,qn1,qn1,qn0,qn1,qn1]))
        tr[28]=qnv.anyv(np.array([qn1,qn0,qn0,qn1,qn1,qn1]))
        tr[29]=qnv.anyv(np.array([qn0,qn1,qn0,qn1,qn1,qn1]))
        tr[30]=qnv.anyv(np.array([qn0,qn0,qn1,qn1,qn1,qn1]))
        tr[31]=qnv.anyv(np.array([qn1,qn1,qn1,qn1,qn1,qn1]))
    return tr
        
      