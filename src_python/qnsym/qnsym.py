import sys
import numpy as np
import cython

import crsys
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import prjop
import qnmath as qmt
import qnndarray as qna

class Qnsym_Octa(qna.QnNdarray):
    def __new__(cls):
        global nr,shape
        nr=32
        shape=(nr,n,n)
        #print("shape in __new__",shape) # for test
        return super().__new__(cls,shape)
        
    def __init__(self):
        ng=3         # three generating elements
        rg=np.zeros((ng,n,n),dtype=np.int64)
        for i in range(ng):
            rg[i]= np.zeros((n, n),dtype=np.int64)
        gorf=(8,2,2)

        rg[0][0][1]=1; rg[0][1][2]=1; rg[0][2][3]=1; rg[0][3][0]=-1; rg[0][4][4]=1 # R8 
        rg[1][1][2]=1; rg[1][2][1]=1; rg[1][0][3]=1; rg[1][3][0]=1;rg[1][4][4]=1 # M
        rg[2][0][0]=-1;rg[2][1][1]=-1;rg[2][2][2]=-1;rg[2][3][3]=-1;rg[2][4][4]=-1 # I
        #print_r(rg)  # for test
        
        #print("shape in __init__",shape) # for test
        r=np.zeros(shape,dtype=np.int64) # nD int array
        #print("r.shape",r.shape) # fpr test
        set_r(rg,gorf,r) # set all integer symmetry operators r
        #print_r(r)  # for test
        self.r=r
        self.qnr=qna.copy(rtoqnr(r))
        self.qnr_e=qna.copy(rtoqnr_e(r))
        self.qnr_i=qna.copy(rtoqnr_i(r))
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
        nr=40
        shape=(nr,n,n)
        return super().__new__(cls,shape)
        
    def __init__(self):
        ng=3         # three generating elements
        rg=np.zeros((ng,n,n),dtype=np.int64)
        gorf=(10,2,2)
        
        rg[0][0][3]=-1;
        rg[0][1][0]=1;rg[0][1][1]=1;rg[0][1][2]=1;rg[0][1][3]=1  # R8
        rg[0][2][0]=-1;rg[0][3][1]=-1;rg[0][4][4]=1 
        rg[1][0][3]=1; rg[1][3][0]=1; rg[1][1][2]=1; rg[1][2][1]=1;rg[1][4][4]=1 # M
        rg[2][0][0]=-1;rg[2][1][1]=-1;rg[2][2][2]=-1;rg[2][3][3]=-1;rg[2][4][4]=-1 # I
        #print_r(rg)  # for test
        
        #r=np.zeros((nr, n, n))
        print("shape",shape) # for test
        r=np.zeros(shape,dtype=np.int64)
        set_r(rg,gorf,r)  # set all integer rotation matrices
        #print_r(r)  # for test
        self.r=r
        self.qnr=qnm.copyms(rtoqnr(r))
        self.qnr_e=qnm.copyms(rtoqnr_e(r))
        self.qnr_i=qnm.copyms(rtoqnr_i(r))
        self.nr=nr
        self.n=n
        self.N=N
        self.shape=shape
        mpltbl=np.zeros((nr,nr),dtype=np.int64)       
        set_mpltbl(mpltbl,r)
        self.mpltbl=mpltbl
 

## for dodecagonal QCs
class Qnsym_Dode(qna.QnNdarray):
    def __new__(cls):
        global nr,shape
        nr=48
        shape=(nr,n,n)
        return super().__new__(cls,shape)
    
    def __init__(self):
        ng=3         # three generating elements
        rg= np.zeros((ng,n,n),dtype=np.int64)
        gorf=(12,2,2)

        rg[0][0][1]=1; rg[0][1][2]=1; rg[0][2][3]=1; rg[0][3][0]=-1; rg[0][3][2]=1;rg[0][4][4]=1 # R12 
        rg[1][0][3]=1; rg[1][1][2]=1; rg[1][2][1]=1; rg[1][3][0]=1;rg[1][4][4]=1 # M
        rg[2][0][0]=-1;rg[2][1][1]=-1;rg[2][2][2]=-1;rg[2][3][3]=-1;rg[2][4][4]=-1 # I
        #print_r(rg)  # for test
        
        r=np.zeros(shape,dtype=np.int64)
        set_r(rg,gorf,r)  # set all integer rotation matrices
        #print_r(r)  # for test
        self.r=r
        self.qnr=qna.copy(rtoqnr(r))
        self.qnr_e=qna.copy(rtoqnr_e(r))
        self.qnr_i=qna.copy(rtoqnr_i(r))
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
        rg= np.zeros((ng,n,n),dtype=np.int64)
        gorf=(5,2,2,3,2)
        # following data not correct
        rg[0][0][0]=1; rg[0][1][2]=1; rg[0][2][3]=1; rg[0][3][4]=1; rg[0][4][5]=1;rg[0][5][1]=1 # R5 
        rg[1][0][0]=-1;rg[1][1][1]=-1; rg[1][2][5]=-1;rg[1][3][4]=-1;rg[1][4][3]=-1; rg[1][5][2]=-1# 2
        rg[2][0][5]=-1;rg[2][1][1]=-1; rg[2][2][3]=1; rg[2][3][2]=1;rg[2][4][4]=-1; rg[2][5][0]=-1# 2
        rg[3][0][1]=1;rg[3][1][2]=1; rg[3][2][0]=1; rg[3][3][5]=1;rg[3][4][3]=-1; rg[3][5][4]=-1# 3
        rg[4][0][0]=-1;rg[4][1][1]=-1; rg[4][2][2]=-1; rg[4][3][3]=-1;rg[4][4][4]=-1; rg[4][5][5]=-1# I
        #shape=(nr,n,n) # for nr nxn -rotation matrices
        #print_r(rg)  # for test
        
        r=np.zeros(shape,dtype=np.int64)
        set_r(rg,gorf,r)  # set all integer rotation matrices
        #print_r(r)  # for test
        self.r=r
        self.qnr=qna.copy(rtoqnr(r))
        self.qnr_e=qna.copy(rtoqnr_e(r))
        self.qnr_i=qna.copy(rtoqnr_i(r))
        self.nr=nr
        self.n=n
        self.N=N
        self.shape=shape
        mpltbl=np.zeros((nr,nr),dtype=np.int64)
        set_mpltbl(mpltbl,r)
        self.mpltbl=mpltbl
 

        
def qnsym_init():
    global n,N,isys
    isys=crsys.isys
    n=crsys.n
    N=crsys.N
    print("qnsym_init isys",isys,"n",n,"N",N)  # for test

def Qnsym():
    global qnr,qnr_e,qnr_i,mpltbl
    if isys==2:
        qns=Qnsym_Icos() #Pn35
    elif isys==3:
        qns=Qnsym_Deca() #P10mm
    elif isys==4:
        qns=Qnsym_Octa() #P8mm
    elif isys==5:
        qns=Qnsym_Dode() #P12mm
    else:
        print("isys should be 2,3,4 or 5 but",isys)
        exit()
    
    qnr=qns.qnr
    qnr_e=qns.qnr_e
    qnr_i=qns.qnr_i
    mpltbl=qns.mpltbl
    return qns
    
def rtoqnr(r):
    shape=r.shape # (nr,n,n)
    nr=shape[0]
    prj=prjop.Prjop()
    prj0t=prjop.prj0t
    prjit=prjop.prjit
    return get_qnr(prj0t,prjit,r,nr)

def rtoqnr_e(r):
    qr=rtoqnr(r)
    if isys==2:
        return qr[:,0:3,0:3] #3x3 diagonal block 
    else:
        return qr[:,0:2,0:2] #2x2 diagonal block not correct at the moment

def rtoqnr_i(r):
    qr=rtoqnr(r)
    #n_=r.shape[0]
    if isys==2:
        return qr[:,3:6,3:6] # 3x3 second diagonal block
    else:
        return qr[:,2:4,2:4] # 2x2 or 3x3 second diagonal block
    
def is_equal(r1,r2):
    for i in range(n):
        for j in range(n):
            if r1[i][j]!=r2[i][j]:
                return False
    return True
        
def set_mpltbl(mpltbl:np.ndarray,r:np.ndarray): # r: integer rotation matrices in nD lattice
    rt=np.zeros((n,n),dtype=np.int64)
    for i in range(nr):
        for j in range(nr):
            for k in range(nr):
                rt=r[i]@r[j]
                if is_equal(rt,r[k]):  # ???
                    mpltbl[i][j]=k
    wt_mpltbl(mpltbl) # for test

def wt_mpltbl(mpltbl: np.ndarray):
    n_=(int)(shape[0]/2) # when centrosymmetric
    print("mpltbl 1st block")
    for i in range(n_):
        for j in range(n_):
            print("","{:2d}".format(mpltbl[i][j]),end="")
        print("")
    
    print("mpltbl 2nd block")
    for i in range(n_):
        for j in range(n_):
            print("",mpltbl[i][j+n_],end="")
        print("")
    
def get_qnr(prj0t,prjit,r,nr):
    shape=r.shape
    qnr=np.zeros(shape,dtype=qnn.Qnnum)
    for i in range(nr):
        rqn=qnm.intm2qnm(r[i],n)
        #qnr[i]=qnm.copy(prjt@rqn@prjit)  # qnmat x intmat nesessary
        qnr[i]=prj0t@rqn@prjit  # qnmat x intmat nesessary
        #str="# "+format(i+1) # for test
        #qnm.printqnm(str,qnr[i]) # for test
    return qnr

def get_qnr0(r,nr):
    qnr0=qna.QnNdarray((nr,n,n))
    for i in range(nr):
        qnr0[i]=qnm.intm2qnm(r[i],n)
        #str="# "+format(i+1) # for test
        #qnm.printqnm(str,qnr[i]) # for test
    return qnr0

# gemerate all rotation matrices from
def mpso(r1,m1,r2,m2,m3):
    r2[m3]=r1[m1]@r2[m2]

def set_r(rg,gorf,r):
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
            print("[ ",end=" ")
            for k in range(n):
                print(r[i][j][k],end=" ")
            print("]")
        print(" ")

def print_r0(r0):
    shape=r0.shape
    ndim=r0.ndim
    #print("r.shape",r.shape)
    #print("r.ndim",ndim)
    nr=shape[0]
    #n=shape[1]
    for i in range(nr):
        print("#",i+1)
        qnm.printqnm("",r0[i])

# integer matrix to qnnumber matrix transformation
def intr2qnmr(r,nr):
    #qnmr=qna.QnNdarray((nr,n,n)) #[qm0]*nr
    qnmr=qnm.Qnmat[nr]
    for i in range(nr):
        qnmr[i]=qnm.intm2qnm(r[i],n) # qnmat for i-th rotation operator r[i]
    return qnmr

def test_wt_qnr(str:str,qns:qna.QnNdarray):
    nr=qns.nr
    print("nr",nr)
    print(str)
    for i in range(nr):
        str="qnr["+format(i)+"]"
        qnm.printqnm(str,qns.qnr[i])
     
def test_wt_qnr_e(str:str,qns:qna.QnNdarray):
    nr=qns.nr
    print("nr",nr)
    print(str)
    for i in range(nr):
        str="qnr["+format(i)+"]"
        qnm.printqnm(str,qns.qnr_e[i])   

def test_wt_qnr_i(str:str,qns:qna.QnNdarray):
    nr=qns.nr
    print("nr",nr)
    print(str)
    for i in range(nr):
        str="qnr["+format(i)+"]"
        qnm.printqnm(str,qns.qnr_i[i])   
        