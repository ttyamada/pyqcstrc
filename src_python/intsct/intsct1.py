import numpy as np
import cython
from numpy.typing import NDArray
import time # in object_subtraction_dev1, tetrahedron_not_obj
import itertools
#from extended_int import int_inf, ExtendedIntegral

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

# new version
# this version calculates all intersection points of two triangle edges
# and return all cross points on the edges
def intersection_two_triangles(triangle_1: qna.QnNdarray, triangle_2: qna.QnNdarray) -> qna.QnNdarray:
    #print("triangle_1.shape",triangle_1.shape)  # for test
    #print("triangle_2.shape",triangle_2.shape)  # for test
    #qnv.printqnvs("triangle_1",triangle_1)  # for test
    edge_1=get_edge(triangle_1)
    #print("edge_1.shape",edge_1.shape)  # for test
    #qnv.printqnvs("triangle_2",triangle_2)  # for test
    edge_2=get_edge(triangle_2)
    #print("edge_2.shape",edge_2.shape)  # for test
    # cross points of edge_1 and edge_2
    t=qna.zeros((3,3,2))
    x=qna.zeros((9,2))   # cross points
    n=0
    for i,e1 in enumerate(edge_1):  # line segment e1
        for j,e2 in enumerate(edge_2):  # line segment e2
            den=(e1[0][0]-e1[1][0])*(e2[0][1]-e2[1][1])\
               -(e1[0][1]-e1[1][1])*(e2[0][0]-e2[1][0])
            #print("i,j",i,j,end=" ")  # for test
            #qnn.printqnn("den",den)  # for test
            if den==qnn.zero():  # e1 // e2
                continue
            de1=e1[1]-e1[0]
            de2=e2[1]-e2[0]
            if den==qnn.zero():  # no cross point (lines are parallel)
                continue
            nux=(e1[0][0]-e2[0][0])*(e2[0][1]-e2[1][1])\
               -(e1[0][1]-e2[0][1])*(e2[0][0]-e2[1][0])
            
            nuy=(e1[0][0]-e1[1][0])*(e1[0][1]-e2[0][1])\
               -(e1[0][1]-e1[1][1])*(e1[0][0]-e2[0][0])
            t[i][j][0]=nux/den # t
            t[i][j][1]=-nuy/den # u
            #qnv.printqnv("t[i][j]",t[i][j])  # for test
            # if 0<=t[i][j][:]<=1 lines have intersection on i and j-th edges of triangles 1 and 2
            # then calculate cross point
            if t[i][j][0] >=qnn.zero() and t[i][j][0] <=qnn.one() and \
            t[i][j][1] >=qnn.zero() and t[i][j][1] <=qnn.one():
            #if t[i][j][0] >=qnn.zero() and t[i][j][0] <=qnn.one(): 
                #print("i,j",i,j,end=" ")  # for test
                #qnn.printqnn("t[i][j][0]",t[i][j][0])  # for test
                x[n]=e1[0]+de1*t[i][j][0]
                #n+=1
            #if t[i][j][1] >=qnn.zero() and t[i][j][1] <=qnn.one():
                #print("i,j",i,j,end=" ")  # for test
                #qnn.printqnn("t[i][j][1]",t[i][j][1])  # for test
                x[n]=e2[0]+de2*t[i][j][1]
                n+=1
    n,x=rmv_overlapedx(x,n)
    return n,x[0:n]  # n cross point coordinates

def common_part(nx:np.int64, x:qna.QnNdarray, ny:np.int64, y:qna.QnNdarray) -> qna.QnNdarray:
    print("nx",nx,"ny",ny)
    z=qnv.zerovs((nx+ny,2))
    n=0
    for i in range(nx):
        z[n]=x[i]
        n+=1
    for i in range(ny):
        z[n]=y[i]
        n+=1
    return n,z

def ball_radius_obj(obj: qnv.Qnvec, centroid: qnv.Qnvec) -> qnn.Qnnum: #float:
    """estimate maximum distance between verices of given OBJ and its centroid.
    
    Parameters
    ----------
    obj: array (ndim=4)
        in TAU-style
    centroid: array, (ndim=2)
        a n-dimensional coordinates in TAU-style
    
    Returns
    -------
    length: float
    
    """
    vertices=remove_doubling_in_perp_space(obj)
    qn0=qnn.Qnnum([0,0,1])
    dd=qn0  #0
    for v in vertices:
        a=v-centroid
        a=projection3(a)
        dd1=qnv.dot(a,a)  #length_numerical(a)
        if dd1>dd:
            dd=dd1
        else:
            pass
    return dd

def ball_radius(triangle: qna.QnNdarray, centroid: qnv.Qnvec) -> qnn.Qnnum: # float:
    #  this transforms a tetrahedron to a boll which covers the triangle
    #  the centre of the boll is the centroid of the triangle.
    return ball_radius_obj(triangle,centroid)

def distance_in_perp_space(vt1: qnv.Qnvec, vt2: qnv.Qnvec) -> qnn.Qnnum:  #float:
    a=vt1-vt2
    a=projection3(a)
    return length_numerical(a)

