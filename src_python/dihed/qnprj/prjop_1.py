import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnmath as qmt
import qnndarray as qna

class Qnprj_Octa(qnm.Qnmat):
    def __new__(cls) : 
        global n
        n=crs.n
        #print("n in __new__",n) # for test
        shape=(n,n)
        return super().__new__(cls,shape)
 
    def __init__(self):
        M0=qnn.any([ 0, 0, 1]) #  0
        M1=qnn.any([ 1, 0, 1]) #  1
        M2=qnn.any([-1, 0, 1]) # -1
        M3=qnn.any([ 0, 1, 2]) #  sqrt(2)/2 t1
        M4=qnn.any([ 0,-1, 2]) # -sqrt(2)/2 t2=-t1
        #self=qnm.Qnmat(n,N)
        
        prj0=np.array([\
           [M1,M0,M1,M0,M0],\
           [M3,M3,M4,M3,M0],\
           [M0,M1,M0,M2,M0],\
           [M4,M3,M3,M3,M0],\
           [M0,M0,M0,M0,M1]\
        ],dtype=qnn.Qnnum)
 
        #qnm.printqnm("Qnprj_Octa prj",prj0) # for 
        self.prj0=prj0
        self.prji=qmt.qnmatinv(prj0,n) # get inversion matrix of prj
        self.n=n
        self.N=crs.N
        self.shape=(n,n)
        self.scl=1.0
        self.scly=1.0
        #prj=self
        prj0f=qnm.qnm2flt(prj0)  # for float number
        prjif=qmt.matinv_f(prj0f,n)
        self.prj0f=prj0f
        self.prjif=prjif
        #qnm.printfm("prj0f",self.prj0f)  # for test


class Qnprj_Deca(qnm.Qnmat):
    def __new__(cls):
        global n
        n=crs.n
        shape=(n,n)
        return super().__new__(cls,shape)
 
    # note that this use orthorhombic coordinate system
    def __init__(self):
        M0=qnn.any([ 0,0,1]  ) #  0
        M1=qnn.any([1,1,4])    # tau/2=c2
        M2=qnn.any([-1,1,4])   # tau^{-1}/2
        M3=qnn.any([1,0,1])    # 1
        tau=qnn.any([1,1,2])
        #c1=cos(2pi/5),c2=cos(4pi/5) s1=sin(2pi/5) s2=sim(4pi/5)
        M5=M2       #    tau^{-1}/2=c1
        M6=-M1      #   -tau/2=c2
        M7=M2*2     #    s2/(s1)=tau^{-1}
        M8=-M7      #   -s2/(s1)=-tau^{-1}
        M9=M3       #    s1/(s1)=1
        M10=-M9     #   -s1/(s1))=-1
        # y axis in external and internal spaces should be scaled by sin(2pi/5)        
        prj0=np.array([\
           [M5,M9,M6,M7,M0],\
           [M6,M7,M5,M10,M0],\
           [M6,M8,M5,M9,M0],\
           [M5,M10,M6,M8,M0],\
           [M0,M0,M0,M0,M3]\
        ],dtype=qnn.Qnnum)
        
        #qnm.printqnm("Qnprj_Deca prj",prj0) # for test
        self.prj0=prj0
        #(D^q)^{-1}
        self.prji=qmt.qnmatinv(prj0,n) # get inversion matrix of prj
        self.n=n
        self.N=crs.N
        self.shape=(n,n)
        self.scl=2.0/np.sqrt(5.0)
        self.scly=crs.scly             #s1
        self.scly2=(M3-M5**2)          #(1-c1^2)
        #self.scly2=4*(M3-M5**2)       #(1-c1^2)
        #prj=self
        #print("self.ndim",self.ndim) # for test
        #print("self.shape",self.shape) # for test
        #self.prj0f=qnm.qnm2flt(prj0)
        #qnm.printfm("prj0f",self.prj0f)  # for test
        prj0f=qnm.qnm2flt(prj0)  # for float number
        prjif=qmt.matinv_f(prj0f,n)
        self.prj0f=prj0f
        self.prjif=prjif
