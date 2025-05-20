import numpy as np
import cython
from numpy.typing import NDArray
import time
import itertools

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import numeric as num
import qnndarray as qna
import prjop as prj

# edges in the triangle
def get_edge(tri: qna.QnNdarray):
    edge=qna.zeros((3,2,2))
    #print("triangle.shape in get_edge",tri.shape)
    for i in range(3): # 0 1 2
        j=(i+1)%3      # 1 2 0
        #print("i,j",i,j)  # for test
        for k in range(2):
            edge[i][0][k]=tri[i][k] # a,b,c
            edge[i][1][k]=tri[j][k] # b,c,a
        #print("i",i,end=" ")  # for test
        #qnv.printqnvs("edge[i]",edge[i])  # for test
    return edge

def det_vecabc(a:qnv.Qnvec,b:qnv.Qnvec,c:qnv.Qnvec)-> qnn.Qnnum:
    da=a-c
    db=b-c
    return da[0]*db[1]-db[0]*da[1]

def counter_clockwise(tri:qna.QnNdarray) ->qna.QnNdarray:
    if det_vecabc(tri[0],tri[1],tri[2])>qnn.zero():
        return tri
    else: # swap tri[0] and tri[1]
        tmp=qnv.copy(tri[0])
        tri[0]=tri[1]
        tri[1]=qnv.copy(tmp)
        return tri
    
# common vertices of tri_1 and tri_2 vertices in their intersection
# (tri_1 : triangle_1 tri_2 : triangle_2)
def common_points(tri_1:qna.QnNdarray,tri_2:qna.QnNdarray)->qna.QnNdarray:
    det1=qnv.zerov(3)
    det2=qnv.zerov(3)
    comx=qnv.zerovs((6,2))
    n=0
    for k in range(3):
        for i in range(3): # 0 1 2
            j=(i+1)%3 # 1 2 0
            det1[i]=det_vecabc(tri_1[i],tri_1[j],tri_2[k])
        if det1[0]>=qnn.zero() and det1[1]>=qnn.zero() and det1[2]>=qnn.zero():
            # tri_2[k] is in tri_1 or on the border of tri_1
            comx[n]=tri_2[k]
            n+=1
        for i in range(3): # 0 1 2
            j=(i+1)%3 # 1 2 0
            det2[i]=det_vecabc(tri_2[i],tri_2[j],tri_1[k])
        if det2[0]>=qnn.zero() and det2[1]>=qnn.zero() and det2[2]>=qnn.zero():
            # tri_1[k] is in tri_2 or on the border of tri_2
            comx[n]=tri_1[k]
            n+=1
    return n,comx[0:n]
        
def rmv_overlapedx(x,n0):
    # x : 2D points
    #print("x.shape",x.shape)  # for test
    n=0
    for i in range(n0):
        if i==0:
            n+=1
            continue
        iskp=0
        for j in range(n):
            if x[i][0] == x[j][0] and x[i][1] == x[j][1]:
                iskp=1
                break
        if iskp==0:
            x[n] = x[i]
            n+=1
    return n,x[0:n]

