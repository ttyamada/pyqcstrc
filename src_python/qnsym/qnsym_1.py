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

class Qnsym_Octa(qna.QnNdarray):
    def __new__(cls):
        global nr,shape
        nr=16 #nr=32
        shape=(nr,n,n)
        #print("shape in __new__",shape) # for test
        return super().__new__(cls,shape)
        
    def __init__(self):
        ng=3         # three generating elements
        rg=np.zeros((ng,n,n),dtype=np.int64)
        #for i in range(ng):
        #    rg[i]= np.zeros((n, n),dtype=np.int64)
        gord=(8,2)   #gord=(8,2,2)
        rg[0][0][1]=1; rg[0][1][2]=1; rg[0][2][3]=1; rg[0][3][0]=-1; rg[0][4][4]=1 # R8 
        rg[1][1][2]=1; rg[1][2][1]=1; rg[1][0][3]=1; rg[1][3][0]=1;rg[1][4][4]=1   # M
        #rg[2][0][0]=-1;rg[2][1][1]=-1;rg[2][2][2]=-1;rg[2][3][3]=-1;rg[2][4][4]=-1 # I
        #print_r(rg)  # for test
        
        #print("shape in __init__",shape) # for test
        r=np.zeros(shape,dtype=np.int64) # nD int array
        rt=np.zeros(shape,dtype=np.int64) # nD int array
        #print("r.shape",r.shape) # fpr test
        set_r(rg,gord,r,rt) # set all integer symmetry operators r
        #print_r(r)  # for test

        self.r=r  # integer rotation matrix
        self.rt=rt
        self.r_qn0=get_r_qn0(r,nr)  # symmetry operator for qnv r_qn0@qnv
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
 
# for decagonal QCs
class Qnsym_Deca(qna.QnNdarray):
    def __new__(cls):
        global nr,shape
        nr=20  #nr=40
        shape=(nr,n,n)
        return super().__new__(cls,shape)
        
    def __init__(self):
        ng=3         # three generating elements
        rg=np.zeros((ng,n,n),dtype=np.int64)  # integer nxn matrices
        gord=(5,2,2)  #gord=(10,2,2)  # order of generators
        
        #rg[0][0][3]=-1;
        #rg[0][1][0]=1;rg[0][1][1]=1;rg[0][1][2]=1;rg[0][1][3]=1  # R10
        #rg[0][2][0]=-1;rg[0][3][1]=-1;rg[0][4][4]=1 
        rg[0][0][1]=1;rg[0][1][2]=1;rg[0][2][3]=1;  #R5
        rg[0][3][0]=-1;rg[0][3][1]=-1;rg[0][3][2]=-1;rg[0][3][3]=-1
        rg[0][4][4]=1
        rg[1][0][3]=1; rg[1][3][0]=1; rg[1][1][2]=1; rg[1][2][1]=1;rg[1][4][4]=1 # M
        rg[2][0][0]=-1;rg[2][1][1]=-1;rg[2][2][2]=-1;rg[2][3][3]=-1;rg[2][4][4]=-1 # I
        #print_r(rg)  # for test
        
        #r=np.zeros((nr, n, n))
        print("shape",shape) # for test
        r=np.zeros(shape,dtype=np.int64)
        rt=np.zeros(shape,dtype=np.int64)
        set_r(rg,gord,r,rt)  # set all integer rotation matrices
        print_r(rg)  # for test

        self.r=r
        self.rt=rt
        self.r_qn0=get_r_qn0(r,nr)
        self.r_qn=qnm.copyms(rtor_qn(r))
        self.r_qn_e=qnm.copyms(rtor_qn_e(r))
        self.r_qn_i=qnm.copyms(rtor_qn_i(r))
        self.nr=nr
        self.n=n
        self.N=N
        self.shape=shape
        mpltbl=np.zeros((nr,nr),dtype=np.int64)       
        set_mpltbl(mpltbl,r)
        self.mpltbl=mpltbl
 