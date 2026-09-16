import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnmath as qmt
import qnndarray as qna 

def prjop_init():
    global isys,n,N
    isys=crs.isys
    n=crs.n
    N=crs.N
    prj=Prjop()

    
def Prjop():
    global prj0,prji,prj0t,prjit
    global prj0f,prjif,scl
    #print("isys in Prjop",isys)  # fpr test
    if(isys==2): # projection operator for icosahedral
        prj=Qnprj_Icos()
    elif(isys==3): # projection operator for decagonal
        prj=Qnprj_Deca()
    elif(isys==4): # projection operator for octagonal
        prj=Qnprj_Octa()
    elif(isys==5): # projection operator dodecagonal
        prj=Qnprj_Dode()
    prj0=prj.prj0
    prji=prj.prji
    prj0f=prj.prj0f
    prjif=prj.prjif
    prj0t=qmt.matrixtr(prj0) # transposed prj matrix
    prjit=qmt.matrixtr(prji) # transposed prji matrix 
    scl=prj.scl
    #scly=prj.scly
    return prj

def tstwt_prjop():
    qnm.printqnm("prj0",prj0)
    qnm.printqnm("prji",prji)
    qnm.printqnm("prj0t",prj0t)
    qnm.printqnm("prjit",prjit)
    unitm=qnm.zerom((n,n))
    unitm=prji@prj0
    qnm.printqnm("unitm",unitm)
    printfm("prj0f",prj0f,n)
    prjif=qmt.matinv_f(prj0f,n)
    printfm("prjif",prjif,n)
    unitmf=prjif@prj0f
    printfm("unitmf",unitmf,n)
    
def copy(qna1: qnm.Qnmat):
    #return np.copy(qna1,dtype=qnn.Qnnum)
    return qnm.copy(qna1)
    #shape=qna1.shape
    #N=qna1[0][0].N
    #qna2=qnm.zerom(shape,N)
    #for i in range(shape[0]):
    #    for j in range(shape[1]):
    #        qna2[i][j]=qnn.copy(qna1[i][j])
    #return qna2
            
# for class cls cls should be icos octa, deca or dode
def prjvec(v: qnv.Qnvec) -> qnv.Qnvec:
    #qnm.printqnm("prj",prj0)
    #qnv.printqnv("v",v)
    n=crs.n
    vei=qnv.zerov(n)
    vei[0:3]=prjvec_e(v)
    if isys>2:
        vei[3:5]=prjvec_i(v)
    else:
        vei[3:6]=prjvec_i(v)
    #qnv.printqnv("vei",vei)  # for test
    return vei

# projection into external space for class cls
def prjvec_e(v:qnv.Qnvec) -> qnv.Qnvec:
    #n=crs.n
    #vei=qnv.zerov(n)
    vei=v@prj0  # v assumed to be qnvec
    ve=qnv.zerov(3)
    if isys>2: # dihedral
        ve[0:2]=vei[0:2]
        ve[2]=vei[4]
    elif isys==2: # icosahedral
        ve[0:3]=vei[0:3]
    #qnv.printqnv("ve",ve)  # for test
    return ve

# projection into internal space for class cls
def prjvec_i(v: qnv.Qnvec) -> qnv.Qnvec:
    #n=crs.n
    #vei=qnv.zerov(n)
    #print("v.shape",v.shape)  # for test
    #qnm.printqnm("prj0",prj0)  # for test
    vei=v@prj0
    #qnv.printqnv("v",v)  # for test
    #qnv.printqnv("vei",vei)  # for test
    isys=crs.isys
    #print("isys in prjvec_i",isys)  # for test
    if isys>2: # dihedral
        ni=2
        vi=qnv.zerov(ni)
        vi=vei[2:4]
    elif isys==2: # icosahedral
        ni=3
        vi=qnv.zerov(ni)
        vi=vei[3:6]
    #print("vi.shape",vi.shape)  # for test
    #qnv.printqnv("vi",vi)  # for test
    return vi
