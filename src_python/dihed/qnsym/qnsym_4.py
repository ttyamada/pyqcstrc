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

def get_r_qn(prj1,prj2,r,nr):
    shape=r.shape
    r_qn=np.zeros(shape,dtype=qnn.Qnnum)
    for i in range(nr):
        rqn=qnm.intm2qnm(r[i],n) # int matrix to qn matrix
        r_qn[i]=qnm.copy(prj1@rqn@prj2)
        #str="# "+format(i+1) # for test
        #qnm.printqnm(str,r_qn[i]) # for test
    return r_qn

def get_r_qn0(r,nr):
    r_qn0=qna.QnNdarray((nr,n,n))
    for i in range(nr):
        r_qn0[i]=qnm.intm2qnm(r[i],n)
        #str="# "+format(i+1) # for test
        #qnm.printqnm(str,r_qn[i]) # for test
    return r_qn0

# gemerate all rotation matrices from
def mpso(r1,m1,r2,m2,m3):
    r2[m3]=r1[m1]@r2[m2]

def set_r(rg,gorf,r,rt):
    ng=len(gorf)
    #n=rg.shape[1] # rg nxn matrix
    r[0]=np.identity(n,dtype=np.int64) # this should be a unit matrix
    impt=1
    for ns in range(ng):
        imp=gorf[ns]
        for i in range(1,imp):
            for j in range(impt):
                mp1=j+(i-1)*impt
                mp2=j+i*impt
                mpso(rg,ns,r,mp1,mp2)
        impt=impt*imp
    nsymo=impt

    # rt:transpose matrix of r
    for ns in range(nsymo):
        for i in range(n):
            for j in range(n):
                rt[ns][i][j]=r[ns][j][i]

    
def set_r0(rg,gorf,r):
    #print("r.shape",r.shape)  # for test
    nr=r.shape[0]
    #n=r.shape[1]
    ndim=len(gorf)
    print("gorf",gorf,"gorf[0]",gorf[0],"gorf[1]",gorf[1],"ndim",ndim)
    #for k in range(ndim):
    #    print(rg[k])
    r[0]=np.identity(n,dtype=np.int64) # this should be a unit matrix

    for i in range(gorf[0]-1):
        r[i+1]=get_r(rg[0],r[i],n)
    if ndim==1:
        return
    nt=1
    for k in range(1,ndim):
        nt=nt*gorf[k-1]
        for j in range(nt):
            for l in range(gorf[k]-1):
                r[j+nt]=get_r(rg[k],r[j],n)

        
# matrix multiple
# this can be replaced by r1*r2
# when r1 and r2 are qnmatrices
def get_r(r1,r2):
    r=np.zeros((n, n),dtype=np.int64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                r[i][j]+=r1[i][k]*r2[k][j]
        
    return r

def print_r(r):
    shape=r.shape
    ndim=r.ndim
    #print("r.shape",r.shape)
    #print("r.ndim",ndim)
    nr=shape[0]
    #n=shape[1]
    for i in range(nr):
        print("#",i+1)
        for j in range(n):
            print("[",end="")
            for k in range(n):
                print("{:3d}".format(r[i][j][k]),end="")
                #print(r[i][j][k],end=" ")
            print("]")
        print(" ")
