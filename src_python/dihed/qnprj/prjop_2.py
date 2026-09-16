import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnmath as qmt
import qnndarray as qna


class Qnprj_Dode(qnm.Qnmat):
    def __new__(cls):
        global n
        n=crs.n
        shape=(n,n)
        return super().__new__(cls,shape)

    def __init__(self):
        M0=qnn.any([ 0, 0, 1]) #0
        M1=qnn.any([ 1, 0, 1]) # 1
        M2=-M1                 #-1
        M3=qnn.any([ 1, 0, 2]) # 1/2
        M4=-M3                 #-1/2
        M5=qnn.any([ 0, 1, 2]) #  sqrt(3)/2
        M6=-M5                 # -sqrt(3)/2

        
        prj0=np.array([\
           [M1,M0,M1,M0,M0],\
           [M5,M3,M6,M3,M0],\
           [M3,M5,M3,M6,M0],\
           [M0,M1,M0,M1,M0],\
           [M0,M0,M0,M0,M1]\
        ],dtype=qnn.Qnnum)
        
        #qnm.printqnm("Qnprj_Dode prj",prj0) # for test
        self.prj0=prj0
        self.prji=qmt.qnmatinv(prj0,n) # get inversion matrix of prj
        self.n=n
        self.N=crs.N
        self.shape=(n,n)
        self.scl=2.0/np.sqrt(6.0)  # for vesta or qnn2flt
        self.scly=1.0
        #prj=self
        #print("self.ndim",self.ndim) # fpr test
        #print("self.shape",self.shape) # fpr test
        #self.prj0f=qnm.qnm2flt(prj0)
        #qnm.printfm("prj0f"self.prj0f)  # for test
        prj0f=qnm.qnm2flt(prj0)  # for float number
        prjif=qmt.matinv_f(prj0f,n)
        self.prj0f=prj0f
        self.prjif=prjif

# for icosahedral QCs
#class Qnprj_Icos(npndarray):
#class Qnprj_Icos(qna.QnNdarray):
class Qnprj_Icos(qnm.Qnmat):
    def __new__(cls):
        global n
        n=crs.n
        shape=(n,n)
        return super().__new__(cls,shape)
    
    def __init__(self):
        M0=qnn.any([ 0, 0, 1]) #  0 
        M1=qnn.any([ 1, 0, 1]) #  1
        M2=-M1                 # -1
        M3=qnn.any([ 1, 1, 2]) #  tau=(1+sqrt(5))/2
        M4=-M4                 # -tau

        prj0=np.array([\
           [M1,M3,M0,M3,M2,M0],\
           [M3,M0,M1,M2,M0,M3],\
           [M3,M0,M2,M2,M0,M4],\
           [M0,M1,M4,M0,M3,M1],\
           [M2,M3,M0,M4,M2,M0],\
           [M0,M1,M3,M0,M3,M2]\
        ],dtype=qnn.Qnnum)
        
        
        #for i in range(n):
        #    for j in range(n):
        #        prj0[i][j]=mt[i][j]
        #qnm.printqnm("Qnprj_Icos prj0",prj0) # for test
        self.prj0=prj0
        self.prji=qmt.qnmatinv(prj0,n) # get inversion matrix of prj
        self.n=n
        self.N=crs.N
        self.shape=(n,n)
        tau=(1.0+np.sqrt(5.0))/2.0
        self.scl=1.0/np.sqrt(2.0+tau)
        self.scly=1.0
        #print("self.ndim",self.ndim) # fpr test
        #print("self.shape",self.shape) # fpr test
        #self.prj0f=qnm.qnm2flt(prj0)
        #qnm.printfm("prj0f",self.prj0f)  # for test
        prj0f=qnm.qnm2flt(prj0)  # for float number
        prjif=qmt.matinv_f(prj0f,n)
        self.prj0f=prj0f
        self.prjif=prjif
      