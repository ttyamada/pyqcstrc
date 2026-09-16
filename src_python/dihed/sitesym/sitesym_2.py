import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import prjop as prj
import qnmath as qmt
import qnndarray as qna
import qnsym as qns
import lattice as lt
from numpy.typing import(NDArray)

def coset(irs) -> np.array: # return coset representativ indices in symop
    """
    irs: iste symmetry operator index in r_qn
    isk: index for coset representatives
    """
    # number of site symmetry operators    
    ns0=nr        # order of point group
    ns1=len(irs)  # order of site symmetry group
    
    print("irs",irs)
    idxt=np.zeros(ns0,dtype=np.int64)
    isk=np.zeros(0,dtype=np.int64)  # coset representative indices
    ics=np.zeros(0,dtype=np.int64)  # all coset indices
    nc=(np.int64)(ns0/ns1) # number of cosets
    print("nunber of cosets",nc) # number of 
    for k in range(nc):
        #print("k",k) # for test
        if k==0:
            isk=np.append(0,isk)  # identity operator
            for j in range(ns1):
                ics=np.append(ics,irs[j])
            #print("len(ics)",len(ics)) # for test
            #print("ics",ics) # for test
        else:
            i=newl(ics,ns0) #new element not included in ics
            #print("i",i) # for test
            isk=np.append(isk,i)
            lics=len(ics)
            #print("lics",lics) # for test
            for j in irs:
                m=qns.mpltbl[i,j]
                #print("i",i,"ics[j]",ics[j],"m",m) # for test
                ics=np.append(ics,m)
    #print("len(isk)",len(isk))
    print("isk",isk) # for test
    return isk

def equivalent_positions(x: qnv.Qnvec, brv: str, isk: np.ndarray, rt: NDArray[np.int64]) -> qnv.Qnvec:
    qnv.printqnv('x',x) # site : lattice coordinates
    print('equivalent_positions()')
    neq=len(isk)
    #print("neq",neq)  # for test
    n=len(x)
    #xs=np.zeros((neq,n),dtype=qnn.Qnnum)
    xs=qnv.zerovs((neq,n))
    #print("type(xs)",type(xs),"type(xs[0])",type(xs[0]))
    for i in range(neq):
        xs[i]=x@rt[i]  #r0[isk[i]]@x
        #qnv.printqnv("x",xs[i])  # for test
    return xs

def equivalent_positions_reduced(x:qnv.Qnvec,brv:str,isk:np.ndarray,r0:NDArray[np.int64]) -> qnv.Qnvec:
    xs=equivalent_positions(x,brv,isk,r0)
    reduce_x(xs,brv) # -1/2 <=x[i] < 2/1
    return xs
    
def reduce_x(xs:qnv.Qnvec,brv:str):
    print("type(xs)",type(xs),"type(xs[0])",type(xs[0]))  # for test
    nv=len(xs) # number of points
    print("nv,n in reduce_x",nv,n)  # for test
    qn1=qnn.any([1,0,2])  # 1/2
    qn2=qnn.any([-1,0,2]) # -/2
    qn3=qnn.any([1,0,1])  # 1
    qn4=qnn.any([-1,0,1]) # -1
    print("xs.shape",xs.shape)  # for test
    #qnn.printqnn("qn1",qn1) # for test
    #qnn.printqnn("qn2",qn2) # for test
    #qnn.printqnn("qn3",qn3) # for test
    #qnn.printqnn("qn4",qn4) # for test
    # how to implement mod for qnnum
    # brv='p' assumed
    for i in range(nv):
        for j in range(n):
            for k in range(2):
                if xs[i][j]<=qn2:
                    #qnn.printqnn("before",xs[i][j])
                    xs[i][j]+=qn3
                    #qnn.printqnn("after",xs[i][j])
                elif xs[i][j]>qn1:
                    #qnn.printqnn("before",xs[i][j])
                    xs[i][j]+=qn4
                    #qnn.printqnn("after",xs[i][j])
        qnv.printqnv("xs",xs[i])
    return

def symop_vec(symop:qnm.Qnmat,vt:qnv.Qnvec,centre:qnv.Qnvec):
    """ 
    Apply a symmetric operation on a vector around given centre. in TAU-style
    """
    #vt=sub_vectors(vt,centre)
    vt=qnv.sub(vt,centre)
    vt=qnm.mul(symop,vt)
    return qnv.add(vt,centre)

