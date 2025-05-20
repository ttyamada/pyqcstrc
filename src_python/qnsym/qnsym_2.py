import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnndarray as qna
import qnvec as qnv
import qnmat as qnm
import prjop as prj
import qnmath as qmt
from numpy.typing import(NDArray)

## for dodecagonal QCs
class Qnsym_Dode(qna.QnNdarray):
    def __new__(cls):
        global nr,shape
        nr=24  #nr=48
        shape=(nr,n,n)
        return super().__new__(cls,shape)
    
    def __init__(self):
        ng=3         # three generating elements
        rg= np.zeros((ng,n,n),dtype=np.int64)  # integer nxn matrices
        gord=(12,2) #gord=(12,2,2)  # order of generators

        rg[0][0][1]=1; rg[0][1][2]=1; rg[0][2][3]=1; rg[0][3][0]=-1; rg[0][3][2]=1;rg[0][4][4]=1 # R12 
        rg[1][0][3]=1; rg[1][1][2]=1; rg[1][2][1]=1; rg[1][3][0]=1;rg[1][4][4]=1 # M
        rg[2][0][0]=-1;rg[2][1][1]=-1;rg[2][2][2]=-1;rg[2][3][3]=-1;rg[2][4][4]=-1 # I
        print_r(rg)  # for test
        
        r=np.zeros(shape,dtype=np.int64)
        rt=np.zeros(shape,dtype=np.int64)
        set_r(rg,gord,r,rt)  # set all integer rotation matrices
        #print_r(r)  # for test
        self.r=r
        self.rt=rt
        self.r_qn0=get_r_qn0(r,nr) 
        self.r_qn=qna.copy(rtor_qn(r))
        self.r_qn_e=qna.copy(rtor_qn_e(r))
        self.r_qn_i=qna.copy(rtor_qn_i(r))
        self.nr=nr
        self.n=n
        self.N=N
        self.shape=shape
        mpltbl=np.zeros((nr,nr),dtype=np.int64)
        set_mpltbl(mpltbl,r)
        self.mpltbl=mpltbl
 
## for icosahedral QCs
class Qnsym_Icos(qna.QnNdarray):
    def __new__(cls):
        global nr,shape
        nr=120
        shape=(nr,n,n)
        return super().__new__(cls,shape)
    
    def __init__(self):
        ng=5 # five generators R5 R3 R2_x R2_y I
        rg= np.zeros((ng,n,n),dtype=np.int64)  # integer nxn matrices
        gord=(5,2,2,3,2)  # order of generators
        # following data not correct
        rg[0][0][0]=1; rg[0][1][2]=1; rg[0][2][3]=1; rg[0][3][4]=1; rg[0][4][5]=1;rg[0][5][1]=1 # R5 
        rg[1][0][0]=-1;rg[1][1][1]=-1; rg[1][2][5]=-1;rg[1][3][4]=-1;rg[1][4][3]=-1; rg[1][5][2]=-1# 2
        rg[2][0][5]=-1;rg[2][1][1]=-1; rg[2][2][3]=1; rg[2][3][2]=1;rg[2][4][4]=-1; rg[2][5][0]=-1# 2
        rg[3][0][1]=1;rg[3][1][2]=1; rg[3][2][0]=1; rg[3][3][5]=1;rg[3][4][3]=-1; rg[3][5][4]=-1# 3
        rg[4][0][0]=-1;rg[4][1][1]=-1; rg[4][2][2]=-1; rg[4][3][3]=-1;rg[4][4][4]=-1; rg[4][5][5]=-1# I
        #shape=(nr,n,n) # for nr nxn -rotation matrices
        print_r(rg)  # for test
        
        r=np.zeros(shape,dtype=np.int64)  # integer nxn matrices
        rt=np.zeros(shape,dtype=np.int64) # integer nxn matrices
        set_r(rg,gord,r,rt)  # set all integer rotation matrices
        print_r(r)  # for test
        self.r=r
        self.rt=rt
        self.r_qn0=get_r_qn0(r,nr)
        self.r_qn=qna.copy(rtor_qn(r))
        self.r_qn_e=qna.copy(rtor_qn_e(r))
        self.r_qn_i=qna.copy(rtor_qn_i(r))
        self.nr=nr
        self.n=n
        self.N=N
        self.shape=shape
        mpltbl=np.zeros((nr,nr),dtype=np.int64)
        set_mpltbl(mpltbl,r)
        self.mpltbl=mpltbl
 