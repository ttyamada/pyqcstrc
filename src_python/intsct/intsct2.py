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

# calculates all intersection points of two triangle edges and return all cross points on the edges
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
    obj: array (ndim=4)
        in TAU-style
    centroid: array, (ndim=2)
        a n-dimensional coordinates in TAU-style
    Returns
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
