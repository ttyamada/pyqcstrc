import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnmat.qnmat as qnm
import pyqcstrc.prjop.prjop as prj
import pyqcstrc.qnmath.qnmath as qmt
import pyqcstrc.qnndarray.qnndarray as qna

class Qnsym_Octa(qna.QnNdarray):
    def __new__(cls):
        global nr,n,N,shape
        nr=32
        n=5
        N=2
        shape=(nr,n,n)
        #print("shape in __new__",shape) # for test
        return super().__new__(cls,shape,N)
        
    def __init__(self):
 
        ng=3
        rg=np.zeros((ng,n,n),dtype=np.int64)
        # two generating elements
        for i in range(ng):
            rg[i]= np.zeros((n, n),dtype=np.int64)
        ord=(8,2,2)

        rg[0][0][1]=1; rg[0][1][2]=1; rg[0][2][3]=1; rg[0][3][0]=-1; rg[0][4][4]=1 # R8 
        rg[1][1][2]=1; rg[1][2][1]=1; rg[1][0][3]=1; rg[1][3][0]=1;rg[1][4][4]=1 # M
        rg[2][0][0]=-1;rg[2][1][1]=-1;rg[2][2][2]=-1;rg[2][3][3]=-1;rg[2][4][4]=-1 # I
        #print_r(rg)  # for test
        
        #print("shape in __init__",shape) # for test
        r=np.zeros(shape,dtype=np.int64) # nD int array
        #print("r.shape",r.shape) # fpr test
        set_r(rg,ord,r) # set all integer symmetry operators r
        #print_r(r)  # for test
        self.r=r
        self.qnr=qna.copy(rtoqnr(r))
        self.nr=nr
        self.n=n
        self.N=N
        self.shape=shape
        
        #prj0=prj.prj0    
        #prji=prj.prji
        # r : int array
        #self=qna.copy(get_qnr(prj0,prji,r,nr,n))  # block diagonakl symmetry operator for ext and int comp.
        #qnm.printqnm("Octa self.shape",self.shape)
    
# for decagonal QCs
class Qnsym_Deca(qna.QnNdarray):
    def __new__(cls):
        global nr,n,N,shape
        nr=40
        n=5
        N=5
        ng=3
        shape=(nr,n,n)
        return super().__new__(cls,shape,N)
        
    def __init__(self):
        ng=3
        # three generating elements
        rg=np.zeros((ng,n,n),dtype=np.int64)
        ord=(10,2,2)
        
        rg[0][0][3]=-1;
        rg[0][1][0]=1;rg[0][1][1]=1;rg[0][1][2]=1;rg[0][1][3]=1  # R8
        rg[0][2][0]=-1;rg[0][3][1]=-1;rg[0][4][4]=1 
        rg[1][0][3]=1; rg[1][3][0]=1; rg[1][1][2]=1; rg[1][2][1]=1;rg[1][4][4]=1 # M
        rg[2][0][0]=-1;rg[2][1][1]=-1;rg[2][2][2]=-1;rg[2][3][3]=-1;rg[2][4][4]=-1 # I
        #print_r(rg)  # for test
        
        #r=np.zeros((nr, n, n))
        print("shape",shape) # for test
        r=np.zeros(shape,dtype=np.int64)
        set_r(rg,ord,r)  # set all integer rotation matrices
        #print_r(r)  # for test
        self.r=r
        self.qnr=qna.copy(rtoqnr(r))
        self.nr=nr
        self.n=n
        self.N=N
        self.shape=shape
        
        #prj0=prj.prj0    
        #prji=prj.prji
        #self=qna.copy(get_qnr(prj0,prji,r,nr,n))  # block diagonakl symmetry operator for ext and int comp.
        #qnm.printqnm("self.shape",self.shape) # for test
        #self.order=nr
    

## for dodecagonal QCs
class Qnsym_Dode(qna.QnNdarray):
    def __new__(cls):
        global nr,n,N,shape
        nr=48
        n=5
        N=3
        shape=(nr,n,n)
        return super().__new__(cls,shape,N)
    
    def __init__(self):
        ng=3
        # two generating elements
        rg= np.zeros((ng,n,n),dtype=np.int64)
        ord=(12,2,2)

        rg[0][0][1]=1; rg[0][1][2]=1; rg[0][2][3]=1; rg[0][3][0]=-1; rg[0][3][2]=1;rg[0][4][4]=1 # R12 
        rg[1][0][3]=1; rg[1][1][2]=1; rg[1][2][1]=1; rg[1][3][0]=1;rg[1][4][4]=1 # M
        rg[2][0][0]=-1;rg[2][1][1]=-1;rg[2][2][2]=-1;rg[2][3][3]=-1;rg[2][4][4]=-1 # I
        #print_r(rg)  # for test
        
        r=np.zeros(shape,dtype=np.int64)
        set_r(rg,ord,r)  # set all integer rotation matrices
        #print_r(r)  # for test
        self.r=r
        self.qnr=qna.copy(rtoqnr(r))
        self.nr=nr
        self.n=n
        self.N=N
        self.shape=shape
        

        #prj0=prj.prj0    
        #prji=prj.prji
        #qnr=get_qnr(prj0,prji,r,nr,n)  # block diagonakl symmetry operator for ext and int comp.
        #prji=prj.Qnprj_Dode()
        #qmt.qnmatinv(prji,n)
        
        #self=qna.copy(get_qnr(prj0,prji,r,nr,n))
        #self.order=nr
        
## for icosahedral QCs
class Qnsym_Icos(qna.QnNdarray):
    def __new__(cls):
        global nr,n,N,shape
        nr=120
        n=6
        N=5
        shape=(nr,n,n)
        return super().__new__(cls,shape,N)
    
    def __init__(self):
        ng=5 # five generators R5 R3 R2_x R2_y I
        # two generating elements
        rg= np.zeros((ng,n,n),dtype=np.int64)
        ord=(5,2,2,3,2)
        # following data not correct
        rg[0][0][0]=1; rg[0][1][2]=1; rg[0][2][3]=1; rg[0][3][4]=1; rg[0][4][5]=1;rg[0][5][1]=1 # R5 
        rg[1][0][0]=-1;rg[1][1][1]=-1; rg[1][2][5]=-1;rg[1][3][4]=-1;rg[1][4][3]=-1; rg[1][5][2]=-1# 2
        rg[2][0][5]=-1;rg[2][1][1]=-1; rg[2][2][3]=1; rg[2][3][2]=1;rg[2][4][4]=-1; rg[2][5][0]=-1# 2
        rg[3][0][1]=1;rg[3][1][2]=1; rg[3][2][0]=1; rg[3][3][5]=1;rg[3][4][3]=-1; rg[3][5][4]=-1# 3
        rg[4][0][0]=-1;rg[4][1][1]=-1; rg[4][2][2]=-1; rg[4][3][3]=-1;rg[4][4][4]=-1; rg[4][5][5]=-1# I
        #shape=(nr,n,n) # for nr nxn -rotation matrices
        #print_r(rg)  # for test
        
        r=np.zeros(shape,dtype=np.int64)
        set_r(rg,ord,r)  # set all integer rotation matrices
        #print_r(r)  # for test
        self.r=r
        self.qnr=qna.copy(rtoqnr(r))
        self.nr=nr
        self.n=n
        self.N=N
        self.shape=shape
        # r : int array
        #qnr=get_qnr(prj0,prji,r,nr,n)  # block diagonakl symmetry operator for ext and int comp.
        #prji=prj.Qnprj_Icos()
        #qmt.qnmatinv(prji,n)
        #self=qna.copy(get_qnr(prj0,prji,r,nr,n))
        #self.order=nr
    
def rtoqnr(r):
    shape=r.shape # (nr,n,n)
    nr=shape[0]
    n=shape[1]
    return get_qnr(prj0,prji,r,nr,n)
    
def qnsym_init(isys):
    global prj0,prji,nr,n,N,shape
    #global qns  # symmetry operators for external and internal space comp. of nD vector
    prjt=prj.prjop_init(isys)
    prj0=prjt.prj0
    prji=prjt.prji
    if isys==2:
        qns=Qnsym_Icos() #Pn35
    elif isys==3:
        qns=Qnsym_Deca() #P10mm
    elif isys==4:
        qns=Qnsym_Octa() #P8mm
    elif isys==5:
        qns=Qnsym_Dode() #P12mm
    shape=qns.shape
    set_mpltbl(qns.r) # 
    qns.mpltbl=mpltbl
    return qns

def is_equal(r1,r2):
    for i in range(n):
        for j in range(n):
            if r1[i][j]!=r2[i][j]:
                return False
    return True
        
def set_mpltbl(r:np.ndarray): # r: integer rotation matrices in nD lattice
    global mpltbl
    mpltbl=np.zeros((nr,nr),dtype=np.int64)
    rt=np.zeros((n,n),dtype=np.int64)
    for i in range(nr):
        for j in range(nr):
            for k in range(nr):
                rt=r[i]@r[j]
                if is_equal(rt,r[k]):  # ???
                    mpltbl[i][j]=k
    wt_mpltbl() # for test

def wt_mpltbl():
    shape=mpltbl.shape
    n=(int)(shape[0]/2)
    print("mpltbl 1st block")
    for i in range(n):
        print("",mpltbl[i][0:n])
    print("mpltbl 2nd block")
    for i in range(n):
        print("",mpltbl[i][n:n*2])
    
def get_qnr(prj,prji,r,nr,n):
    N=prj[0][0].N
    qnr=qna.QnNdarray((nr,n,n),N)
    prjt=qmt.matrixtr(prj)  # transposed prj matrix
    prjit=qmt.matrixtr(prji) # transposed prji matrix 
    for i in range(nr):
        rqn=qnm.intm2qnm(r[i],n,N)
        #qnr[i]=qnm.copy(prjt@rqn@prjit)  # qnmat x intmat nesessary
        qnr[i]=prjt@rqn@prjit  # qnmat x intmat nesessary
        #str="# "+format(i+1) # for test
        #qnm.printqnm(str,qnr[i]) # for test
    return qnr

def get_qnr0(r,nr,n,N):
    qnr0=qna.QnNdarray((nr,n,n),N)
    for i in range(nr):
        qnr0[i]=qnm.intm2qnm(r[i],n,N)
        #str="# "+format(i+1) # for test
        #qnm.printqnm(str,qnr[i]) # for test
    return qnr0

# gemerate all rotation matrices from
def mpso(r1,m1,r2,m2,m3,n):
    r2[m3]=r1[m1]@r2[m2]

def set_r(rg,ord,r):
    ng=len(ord)
    n=rg.shape[1] # rg nxn matrix
    r[0]=np.identity(n,dtype=np.int64) # this should be a unit matrix
    impt=1
    for ns in range(ng):
        imp=ord[ns]
        for i in range(1,imp):
            for j in range(impt):
                mp1=j+(i-1)*impt
                mp2=j+i*impt
                mpso(rg,ns,r,mp1,mp2,n)
        impt=impt*imp
    nsymo=impt

def set_r0(rg,ord,r):
    #print("r.shape",r.shape)  # for test
    nr=r.shape[0]
    n=r.shape[1]
    ndim=len(ord)
    print("ord",ord,"ord[0]",ord[0],"ord[1]",ord[1],"ndim",ndim)
    #for k in range(ndim):
    #    print(rg[k])
    r[0]=np.identity(n,dtype=np.int64) # this should be a unit matrix

    for i in range(ord[0]-1):
        r[i+1]=get_r(rg[0],r[i],n)
    if ndim==1:
        return
    nt=1
    for k in range(1,ndim):
        nt=nt*ord[k-1]
        for j in range(nt):
            for l in range(ord[k]-1):
                r[j+nt]=get_r(rg[k],r[j],n)

        
# matrix multiple
# this can be replaced by r1*r2
# when r1 and r2 are qnmatrices
def get_r(r1,r2,n):
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
    n=shape[1]
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
    n=shape[1]
    for i in range(nr):
        print("#",i+1)
        qnm.printqnm("",r0[i])

# integer matrix to qnnumber matrix transformation
def intr2qnmr(r,n,N,nr):
    qnmr=qna.QnNdarray((nr,n,n),N) #[qm0]*nr
    for i in range(nr):
        qnmr[i]=qnm.intm2qnm(r[i],n,N) # qnmat for i-th rotation operator r[i]
    return qnmr

def test_wt(str:str,qns:qna.QnNdarray):
    nr=qns.nr
    print("nr",nr)
    print(str)
    for i in range(nr):
        qnm.printqnm("qnr[i]",qns.qnr[i])
        
# for test
if __name__ == '__main__':
    # test for qnnum projection operators
    isys=4
    prj4=prj.prjop_init(isys)
    qns4=qnsym_init(isys)
    test_wt("Octa",qns4)

    isys=3
    prj3=prj.prjop_init(isys)
    qns3=qnsym_init(isys)
    test_wt("Deca",qns3)

    isys=5
    prj5=prj.prjop_init(isys)
    qns5=qnsym_init(isys)
    test_wt("Dode",qns5)
    
    isys=2
    prj2=prj.prjop_init(isys)
    qns2=qnsym_init(isys)
    test_wt("Icos",qns2)
    

        