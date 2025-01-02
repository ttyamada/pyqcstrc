import sys
import numpy as np
from numpy.typing import NDArray

class Qnnum:
    def __init__(self, n: NDArray[np.int64], N: np.int64):
        self.n=n
        self.N=N # 2 5 3 for octagonal, decagonal and dodecagonal Qnnumber   
    
    def __add__(a, b):
        return add(a,b)
    
    def __sub__(a, b):
        return sub(a,b)
    
    def __mul__(a, b):
        return mul(a,b)
    
    def __truediv__(a, b):
        return div(a,b)

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
    
# Qnnumber to np.array converter
def qn2npa(a):
    return np.array([a.n[0],a.n[1],a.n[2]])

def qn2flt(a):
    return (a.n[0]+a.n[1]*np.sqrt(a.N))/a.n[2]

 