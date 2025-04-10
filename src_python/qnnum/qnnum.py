import sys
import numpy as np
import cython

#from extended_int import int_inf, ExtendedIntegral
import crsys
from numpy.typing import NDArray
from typing import Self

import crsys

class Qnnum:
    def __init__(self, n: NDArray[np.int64]):
        self.n=crsys.n
        #N :2 5 3 for octagonal, decagonal and dodecagonal Qnnumber   
        n_=np.array([n[0],n[1],n[2]])
        #print("n in Qnnum__init__",n)
        #print("n_ in Qnnum__init__",n_)
        self.n=n_
        self.N=crsys.N
    
    def __add__(a:Self, b:Self):
        return add(a,b)
    
    def __sub__(a:Self, b:Self):
        return sub(a,b)
    
    def __iadd__(self, b:Self):
        return iadd(self,b)
    
    def __isub__(self, b:Self):
        return isub(self,b)
    
    def __mul__(a:Self, b:Self):
        if isinstance(b, Qnnum):
            return mul(a,b)
        elif isinstance(b, int):
            return mul_i(a,b)
        elif isinstance(a, int):
            return i_mul(a,b)
        
    def __pow__(a:Self, b:int ):
        return pow(a,b)
    
    def __truediv__(a:Self, b:Self):
        if isinstance(b, Qnnum):
            return div(a,b)
        elif isinstance(b, int):
            return div_i(a,b)
    
    def __eq__(a:Self, b:Self):
        return eq(a,b)
    
    def __lt__(a:Self,b:Self):
        return lt(a,b)
    
    def __le__(a:Self,b:Self):
        return leq(a,b)
    
    def __gt__(a:Self,b:Self):
        return gt(a,b)
    
    def __ge__(a:Self,b:Self):
        return geq(a,b)
    
    def __neg__(self):
        return neg(self)
    
def qnnum_init():
    global n,N
    isys=crsys.isys
    n=crsys.n
    N=crsys.N
    
def zero():
    return Qnnum([0,0,1])

def inf():
    return Qnnum([int_inf,0,1])

def zeros(shape): # qnnum 1D darray
    n=shape[0]
    qnns = [zero() for i in range(n)]
    return qnns
    #return np.zeros(shape,dtype=Qnnum)
    
def copy(a:Qnnum):
    self=Qnnum([0,0,1])
    self.n[0]=np.copy(a.n[0])
    self.n[1]=np.copy(a.n[1])
    self.n[2]=np.copy(a.n[2])
    #self.N=N
    return self

def zero():
    return Qnnum([0,0,1])

def one():
    return Qnnum([1,0,1])

def any_i(n_: NDArray[np.int64]):
    return Qnnum([n_,0,1])

def any(n_:NDArray[np.int64]):
    return Qnnum([n_[0],n_[1],n_[2]])

def add(a:Qnnum, b:Qnnum):
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
        return Qnnum(np.array([-c1,-c2,-c3]))
    else:
        return Qnnum(np.array([c1,c2,c3]))

def iadd(self:Qnnum, b:Qnnum):
    self=add(self,b)
    return self

def sub(a:Qnnum, b:Qnnum):
    c1=a.n[0]*b.n[2]-b.n[0]*a.n[2]
    c2=a.n[1]*b.n[2]-b.n[1]*a.n[2]
    c3=a.n[2]*b.n[2]
    x=np.array([c1,c2,c3],dtype=np.int64)
    g=np.gcd.reduce(x)
    #print("c1",c1,"c2",c2,"c3",c3,"g",g)
    c1=int(c1/g)
    c2=int(c2/g)
    c3=int(c3/g)
    if c3<0:
        return Qnnum(np.array([-c1,-c2,-c3]))
    else:
        return Qnnum(np.array([c1,c2,c3]))

def isub(self:Qnnum, b:Qnnum):
    self=sub(self,b)
    return self

def mul(a:Qnnum, b:Qnnum):
    #N =(int)(a.N)
    c1=a.n[0]*b.n[0]+a.n[1]*b.n[1]*N
    c2=a.n[0]*b.n[1]+a.n[1]*b.n[0]
    c3=a.n[2]*b.n[2]
    #print("a.n0",a.n[0],"a.n1",a.n[1],"a.n2",a.n[2],"N",N)
    x=np.array([c1,c2,c3],dtype=np.int64)
    g=np.gcd.reduce(x)
    c1=int(c1/g)
    c2=int(c2/g)
    c3=int(c3/g)
    if c3<0:
        return Qnnum(np.array([-c1,-c2,-c3]))
    else:
        return Qnnum(np.array([c1,c2,c3]))
    
def mul_i(a:Qnnum, b:np.int64): # b should be int
    c1=a.n[0]*b
    c2=a.n[1]*b
    c3=a.n[2]
    return Qnnum(np.array([c1,c2,c3]))

def i_mul(a:np.int64, b:Qnnum): # a should be int this does not work?
    #s=Qnnum([a,a,1])
    c1=a*b.n[0]
    c2=a*b.n[1]
    c3=b.n[2]
    return Qnnum(np.array([c1,c2,c3]))

def div(a:Qnnum, b:Qnnum):
    #N = (int)(a.N)
    c1=b.n[0]*b.n[2]
    c2=-b.n[1]*b.n[2]
    c3=b.n[0]*b.n[0]-b.n[1]*b.n[1]*N
    #print("n1**2",b.n[0]*b.n[0],"n2**2",b.n[1]*b.n[1],"N",N)
    if c3==0:
        print('ERROR_1:division error')
        return
    c=Qnnum(np.array([c1,c2,c3]))
    return mul(a,c)

def div_i(a:Qnnum, b:np.int64): # b should be int
    c1=a.n[0]
    c2=a.n[1]
    c3=a.n[2]*b
    c=Qnnum(np.array([c1,c2,c3]))
    return c

def pow(a:Qnnum, b:np.int64):
    if b==0:
        return Qnnum([1,1,1])
    elif b>0:
        c=one()
        for i in range(b):
            c=mul(c,a)
        return c
    elif b<0:
        c=one()
        ai=c/a
        for i in range(b):
            c=mul(c,ai)
        return c
        
def eq(a:Qnnum, b:Qnnum):
    #if (a.n[0]==int_inf and b.n[0]==int_inf):
    #    return Qnnum([0,0,1])
    c=a-b
    if(c.n[0]==0 and c.n[1]==0):
        return True
    else:
        return False

def gt(a:Qnnum, b:Qnnum):
    c=a-b
    if(np.sign(c.n[0])*c.n[0]**2+np.sign(c.n[1])*c.n[1]**2*N > 0):
        return True
    else:
        return False
    
def geq(a:Qnnum, b:Qnnum):
    c=a-b
    if(np.sign(c.n[0])*c.n[0]**2+np.sign(c.n[1])*c.n[1]**2*N >= 0):
        return True
    else:
        return False


def lt(a:Qnnum, b:Qnnum):
    c=a-b
    if(np.sign(c.n[0])*c.n[0]**2+np.sign(c.n[1])*c.n[1]**2*N < 0):
        return True
    else:
        return False
    
def leq(a:Qnnum, b:Qnnum):
    c=a-b
    if(np.sign(c.n[0])*c.n[0]**2+np.sign(c.n[1])*c.n[1]**2*N <= 0):
        return True
    else:
        return False
    
def neg(a:Qnnum):
    #print("a.n[0]",a.n[0],"a.n[1]",a.n[1],"a.n[2]",a.n[2],"N",N) # for test
    b=Qnnum([-a.n[0],-a.n[1],a.n[2]]) # -self
    #printqnn("b",b)  # for test
    return b

def abs(a:Qnnum):
    if a<0:
        return neg(a)
    return a

# Qnnumber to np.array converter
def qn2npa(a:Qnnum) -> np.ndarray:
    return np.array([a.n[0],a.n[1],a.n[2]])

def qn2flt(a:Qnnum) -> float:
    return (a.n[0]+a.n[1]*np.sqrt(N))/a.n[2]

def int2qn(i:np.int64,N:np.int64):
    #N=self.N
    return Qnnum([i,0,1])

def flt2qn(qr:float) -> Qnnum:
    xm=np.abs(qr)
    isg=np.array([1,-1])
    sqrtn=np.sqrt(float(N))
    #print("N",N,"sqrtn",sqrtn) # for test
    eps=0.000001
    n1m=200; n2m=200; n3m=200
    xn=Qnnum([0,0,1])
    for k in range(n3m):
        n3=k+1
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
    
def printqnns(str:str,a:np.ndarray):
    for i in range(a.shape[0]):
        print(str,"[",a[i].n[0],a[i].n[1],a[i].n[2],"]")
    
    
 