import sys
import numpy as np
from numpy.typing import NDArray

class Qnnum:
    def __init__(self, n: np.int64, N: np.int64):
        self.n=n
        #self.N=N # 2 5 3 for octagonal, decagonal and dodecagonal Qnnumber   
        self.N=N
    
    def __add__(a, b):
        return add(a,b)
    
    def __sub__(a, b):
        return sub(a,b)
    
    def __iadd__(self, b):
        return iadd(self,b)
    
    def __isub__(self, b):
        return isub(self,b)
    
    def __mul__(a, b):
        return mul(a,b)
    
    def __truediv__(a, b):
        return div(a,b)
    
    def __eq__(a, b):
        return eq(a,b)
    
    def __lt__(a,b):
        return lt(a,b)
    
    def __gt__(a,b):
        return gt(a,b)
    
def add(a, b):
    #print("a1",a.n[0],"a2",a.n[1],"a3",a.n[2])
    #print("b1",b.n[0],"b2",b.n[1],"b3",b.n[2])
    c1=a.n[0]*b.n[2]+b.n[0]*a.n[2]
    c2=a.n[1]*b.n[2]+b.n[1]*a.n[2]
    c3=a.n[2]*b.n[2]
    x=np.array([c1,c2,c3],dtype=np.int64)
    g=np.gcd.reduce(x)
    #print("c1",c1,"c2",c2,"c3",c3,"g",g)
    c1=int(c1/g)
    c2=int(c2/g)
    c3=int(c3/g)
    #print("c1",c1,"c2",c2,"c3",c3)
    if c3<0:
        return Qnnum(np.array([-c1,-c2,-c3]),a.N)
    else:
        return Qnnum(np.array([c1,c2,c3]),a.N)

def iadd(self, b):
    self=add(self,b)
    return self

def sub(a, b):
    c1=a.n[0]*b.n[2]-b.n[0]*a.n[2]
    c2=a.n[1]*b.n[2]-b.n[1]*a.n[2]
    c3=a.n[2]*b.n[2]
    x=np.array([c1,c2,c3],dtype=np.int64)
    g=np.gcd.reduce(x)
    c1=int(c1/g)
    c2=int(c2/g)
    c3=int(c3/g)
    if c3<0:
        return Qnnum(np.array([-c1,-c2,-c3]),a.N)
    else:
        return Qnnum(np.array([c1,c2,c3]),a.N)

def isub(self, b):
    self=sub(self,b)
    return self

def mul(a, b):
    c1=a.n[0]*b.n[0]+a.N*a.n[1]*b.n[1]
    c2=a.n[0]*b.n[1]+a.n[1]*b.n[0]
    c3=a.n[2]*b.n[2]
    #print("a.n0",a.n[0],"a.n1",a.n[1],"a.n2",a.n[2],"a.N",a.N)
    x=np.array([c1,c2,c3],dtype=np.int64)
    g=np.gcd.reduce(x)
    c1=int(c1/g)
    c2=int(c2/g)
    c3=int(c3/g)
    if c3<0:
        return Qnnum(np.array([-c1,-c2,-c3]),a.N)
    else:
        return Qnnum(np.array([c1,c2,c3]),a.N)

def div(a, b):
    c1=b.n[0]*b.n[2]
    c2=-b.n[1]*b.n[2]
    c3=b.n[0]*b.n[0]-a.N*b.n[1]*b.n[1]
    #print("n1**2",b.n[0]*b.n[0],"n2**2",b.n[1]*b.n[1],"a.N",a.N)
    if c3==0:
        print('ERROR_1:division error')
        return
    c=Qnnum(np.array([c1,c2,c3]),a.N)
    return mul(a,c)

def eq(a, b):
    c=a-b
    if(c.n[0]==0 and c.n[1]==0):
        return True
    else:
        return False

def gt(a, b):
    c=a-b
    if(np.sign(c.n[0])*c.n[0]**2+np.sign(c.n[1])*c.n[1]**2*a.N > 0):
        return True
    else:
        return False

def lt(a, b):
    c=a-b
    if(np.sign(c.n[0])*c.n[0]**2+np.sign(c.n[1])*c.n[1]**2*a.N < 0):
        return True
    else:
        return False
    
# Qnnumber to np.array converter
def qn2npa(a):
    return np.array([a.n[0],a.n[1],a.n[2]])

def qn2flt(a):
    return (a.n[0]+a.n[1]*np.sqrt(a.N))/a.n[2]

def int2qnn(i:np.int64,N:np.int64):
    #N=self.N
    return Qnnum([i,0,1],N)

def flt2qn(qr:float,N:np.int64) -> Qnnum:
    xm=abs(qr)
    isg=np.array([1,-1])
    sqrtn=np.sqrt(float(N))
    #print("N",N,"sqrtn",sqrtn) # for test
    eps=0.000001
    n1m=200; n2m=200; n3m=200
    xn=Qnnum([0,0,1],N)
    for k in range(n3m):
        n3=k
        for i in range(n1m):
            for j in range(n2m):
                for ic in range(2):
                    n1=isg[ic]*i #+-i
                    for jc in range(2):
                        n2=isg[jc]*j #+-j
                        xt=(n1+n2*sqrtn)/n3
                        #print("xt",xt,"qr",qr)
                        xd=(xt-qr)
                        if(np.abs(xd) < xm):
                            xm=np.abs(xd)
                            xn.n[0]=n1
                            xn.n[1]=n2
                            xn.n[2]=n3
                        if(np.abs(xd)<eps):
                            #print("xt,n1,n2,n3.sqrtnr",xt,n1,n2,n3,sqrtn)
                            #printqnn("xn",xn)
                            return xn
    print("cannt convert float to qnnum")

def printqnn(str:str,a:Qnnum):
    print(str,"[",a.n[0],a.n[1],a.n[2],"]")
 