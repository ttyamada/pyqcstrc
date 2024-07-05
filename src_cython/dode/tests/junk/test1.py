#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import os
import sys
import numpy as np
sys.path.append('..')
import math2
import symmetry2

SIN=np.sqrt(3)/2.0

p1,p2,p3=1,2,1
q1,q2,q3=1,1,1

#p1,p2,p3=1,2,3
#q1,q2,q3=4,5,6


def mul(p1,p2,p3,q1,q2,q3):
    c1=4*p1*q1+3*p2*q2
    c2=4*(p1*q2+p2*q1)
    c3=4*p3*q3
    x=np.array([c1,c2,c3])
    gcd=np.gcd.reduce(x)
    if c3/gcd<0:
        return [int(-c1/gcd),int(-c2/gcd),int(-c3/gcd)]
    else:
        return [int(c1/gcd),int(c2/gcd),int(c3/gcd)]

def div(p1,p2,p3,q1,q2,q3):
    if q1==0 and q2==0:
        print('ERROR_1:division error')
        return 1
    else:
        if p1==0 and p2==0:
            return [0,0,1]
        else:
            if q1!=0 and q2!=0:
                if 4*q1**2-3*q2**2!=0:
                    c1=q3*(4*p1*q1-3*p2*q2)
                    c2=-4*q3*(p1*q2-p2*q1)
                    c3=p3*(4*q1**2-3*q2**2)
                else:
                    c1=3*p2*q3
                    c2=4*p1*q3
                    c3=6*p3*q2
            elif q1==0 and q2!=0:
                c1=3*p2*q3
                c2=4*p1*q3
                c3=3*p3*q2
            else:
            #elif q1!=0 and q2==0:
                c1=p1*q3
                c2=p2*q3
                c3=p3*q1
            x=np.array([c1,c2,c3])
            gcd=np.gcd.reduce(x)
            #print('c1,c2,c3,gcd = %d %d %d %d'%(c1,c2,c3,gcd))
            if gcd!=0:
                if c3/gcd<0:
                    return [int(-c1/gcd),int(-c2/gcd),int(-c3/gcd)]
                else:
                    return [int(c1/gcd),int(c2/gcd),int(c3/gcd)]
            else:
                print('ERROR_2:division error',c1,c2,c3,p1,p2,p3,q1,q2,q3)
                return 1
    
print('math2.add')
a,b,c=math2.add(p1,p2,p3,q1,q2,q3)
print('%d %d %d'%(a,b,c))
print('%8.6f'%((a+b*SIN)/c))
print('%8.6f'%((p1+p2*SIN)/p3 + (q1+q2*SIN)/q3))

print('math2.sub')
a,b,c=math2.sub(p1,p2,p3,q1,q2,q3)
print('%d %d %d'%(a,b,c))
print('%8.6f'%((a+b*SIN)/c))
print('%8.6f'%((p1+p2*SIN)/p3 - (q1+q2*SIN)/q3))

print('math2.mul')
a,b,c=math2.mul(p1,p2,p3,q1,q2,q3)
print('%d %d %d'%(a,b,c))
print('%8.6f'%((a+b*SIN)/c))
print('%8.6f'%((p1+p2*SIN)/p3 * (q1+q2*SIN)/q3))

print('mul')
a,b,c=mul(p1,p2,p3,q1,q2,q3)
print('%d %d %d'%(a,b,c))
print('%8.6f'%((a+b*SIN)/c))
print('%8.6f'%((p1+p2*SIN)/p3 * (q1+q2*SIN)/q3))

print('math2.div')
a,b,c=math2.div(p1,p2,p3,q1,q2,q3)
print('%d %d %d'%(a,b,c))
print('%8.6f'%((a+b*SIN)/c))
print('%8.6f'%((p1+p2*SIN)/p3 / (q1+q2*SIN)*q3))

print('div')
a,b,c=div(p1,p2,p3,q1,q2,q3)
print('%d %d %d'%(a,b,c))
print('%8.6f'%((a+b*SIN)/c))
print('%8.6f'%((p1+p2*SIN)/p3 / (q1+q2*SIN)*q3))
#a=math2.gcd(1,2,3)
#print(a)

d1=np.array([[1,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
d2=np.array([[0,0,1],[1,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
d3=np.array([[0,0,1],[0,0,1],[1,0,1],[0,0,1],[0,0,1],[0,0,1]])
d4=np.array([[0,0,1],[0,0,1],[0,0,1],[1,0,1],[0,0,1],[0,0,1]])
i=0
for di in [d1,d2,d3,d4]:
    print('--------')
    print('  %d'%i)
    print('--------')
    print(' math2.projection')
    v1e,v2e,v3e,v1i,v2i,v3i=math2.projection(di[0],di[1],di[2],di[3],di[4],di[5])
    print(' external:')
    print('  %8.6f %8.6f %8.6f'%(((v1e[0]+v1e[1]*SIN)/v1e[2],(v2e[0]+v2e[1]*SIN)/v2e[2],(v3e[0]+v3e[1]*SIN)/v3e[2])))
    print(' internal:')
    print('  %8.6f %8.6f %8.6f'%(((v1i[0]+v1i[1]*SIN)/v1i[2],(v2i[0]+v2i[1]*SIN)/v2i[2],(v3i[0]+v3i[1]*SIN)/v3i[2])))
    print(' math2.projection3')
    v1i,v2i,v3i=math2.projection3(di[0],di[1],di[2],di[3],di[4],di[5])
    print(' internal:')
    print('  %8.6f %8.6f %8.6f'%(((v1i[0]+v1i[1]*SIN)/v1i[2],(v2i[0]+v2i[1]*SIN)/v2i[2],(v3i[0]+v3i[1]*SIN)/v3i[2])))
    i+=1
    
v0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
v1=np.array([[1,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
v2=np.array([[0,0,1],[1,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
vectors=np.vstack([v1]).reshape(1,6,3)
#vectors=np.vstack([v0,v1,v2]).reshape(3,6,3)
centre=v0
vsym=symmetry2.generator_obj_symmetric_vec(vectors,centre)
for i in range(len(vsym)):
    vi=vsym[i]
    for j in range(len(vi)):
        vj=vi[j]
        v1i,v2i,v3i=math2.projection3(vj[0],vj[1],vj[2],vj[3],vj[4],vj[5])
        print('%8.6f %8.6f %8.6f'%(((v1i[0]+v1i[1]*SIN)/v1i[2],(v2i[0]+v2i[1]*SIN)/v2i[2],(v3i[0]+v3i[1]*SIN)/v3i[2])))