#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
import cython
from numpy.typing import NDArray

#from scipy.spatial import Delaunay
import pyDelaunay2D as dln
import itertools
import time

import crsys as crs
import qnnum as qnn
import qnndarray as qna
import qnvec as qnv
import qnmat as qnm
import qnmath as qmt
import numeric as num
import prjop as prj

def shift_object(obj: qna.QnNdarray, shift: qnv.Qnvec) -> qnv.Qnvec:
    """shift an object
    """
    shape=obj.shape
    print("obj.shape",obj.shape)  # for test
    print("shit.shape",shift.shape)  # for test
    if obj.ndim==3:
        obj_new=qna.zeros(shape) 
        for i1,tri in enumerate(obj):
            for i2,vertex in enumerate(tri):
                obj_new[i1][i2]=vertex+shift
                i2+=1
            i1+=1
        return obj_new
    else:
        print('object has an incorrect shape!')
        return 

#----------------------------
# Volume, area
#----------------------------
def obj_area_nd(obj: qnv.Qnvec) -> qnn.Qnnum:
    """Calculate volume of an object (set of triangles) in TAU style.
    
    Parameters
    ----------
    obj: array
    
    Returns
    -------
    area: array
        area in TAU-style.
    """
    #N=crs.N
    qn0=qnn.Qnnum([0,0,1])
    shape=obj.shape
    ndim=len(shape) # ndim qnndvector
    #print("shape",shape,"ndim",ndim)  # for test
    w=qn0
    if ndim==3:
        for tri in obj:
            v=triangle_area_nd(tri)
            w+=v
        return w
    #elif ndim==4:
    #    for tset in obj:
    #        for tri in tset:
    #            v=triangle_area_nd(tri)
    #            w=w+v
    #    return w
    elif ndim==2:
        return triangle_area_nd(obj)
    else:
        print('object has an incorrect shape!')
        return 

def triangle_area_2d(tri: qnv.Qnvec) -> qnn.Qnnum:
    shape=tri.shape
    ni=2
    qnmt=qnm.zerom(2,ni)
    for i in range(2):
        qnmt[i][0]=tri[i+1][0]-tri[0][0]
        qnmt[i][1]=tri[i+1][1]-tri[1][0]
    det=qnm.det_matrix_2d(qnmt)
    return qnn.abs(det)/2
    
def triangle_sqarea_3d(tri: qnv.Qnvec) -> qnn.Qnnum:
    ni=3
    qnvt=qnv.zerovs(3)
    qnvt[0]=tri[1]-tri[0]
    qnvt[1]=tri[2]-tri[0]
    qnvt[2]=qnv.cros(qnv[0],qnv[1])
    det=qnv.dot(qnvt[2],qnvt[2]) 
    return det/4  # squared area of a triangle

def tetrahedron_volume(tet: NDArray[np.float64]) -> qnn.Qnnum:
    """This function returns volume of a tetrahedron
        
    Parameters
    ----------
    tet: array
        vertex coordinates of the tetrahedron, xyz0,xyz1,xyz2,xyz3
    """
    #N=tet[0].N
    n=3
    qnmt=qnm.zeros((n))
    for i in range(n):
        qnmt[i][0]=tet[i+1][0]-tet[0][0]
        qnmt[i][1]=tet[i+1][1]-tet[0][1]
        qnmt[i][2]=tet[i+1][2]-tet[0][2]
    det = qmt.det_matrix_3d(qnmt)
    return qnn.abs(det)/6

def triangle_area_nd(tri: qnv.Qnvec) -> qnn.Qnnum:
    """Calculate volume of triangle in TAU style.
    
    Parameters
    ----------
    tri: array
        nd vectors of triangle vertices in TAU-style.
    
    Returns
    -------
    area: array
        Area in TAU-style.
    """

    shape=tri.shape
    ndim=len(shape)
    #print("shape",shape,"ndim",ndim)  # for test
    qn0=qnn.Qnnum([0,0,1])
    if ndim==2:
        return triangle_area(tri)
    else:
        print('object has an incorrect shape!')
        return 

#######################
###  To be checked  ###
#######################
# this can be replaced by triangle_sqarea_3d
def triangle_area(vts: qna) -> qnn.Qnnum:
    """Calculate area of a triangle in TAU style.
    
    Parameters
    ----------
    obj: array
        vertex coordinates of the triangle (x0,y0,z0),(x1,y1,z1),(x2,y2,z2) in TAU-style.
    
    Returns
    -------
    volume: array
        Volume in TAU-style.
    """
    isys=crs.isys
    v1=vts[1]-vts[0]
    v2=vts[2]-vts[0]

    if isys>2:
        m=np.stack((v1,v2))
        #print("m.shape",m.shape)
        det=qmt.det_matrix_2d(m)
    else:
        v3=vts[3]-vts[0]
        m=np.stack((v1,v2,v3))
        #print("m.shape",m.shape)
        det=qmt.det_matrix_3d(m)
        #v=qnv.cros(v1,v2)
        #det=qnv.dot(v,v)
    #qnn.printqnn("det",det)  # for test
    qn0=qnn.Qnnum([0,0,1])    
    if det < qn0:  #a1+a2*TAU<0.0: # to avoid negative volume...
        return -det/4 #mul(v[2],np.array([-1,0,2]))
    else:
        return det/4  #mul(v[2],np.array([1,0,2]))

#----------------------------
# Remove doubling
#----------------------------
def remove_doubling(vts: qnv.Qnvec) -> qnv.Qnvec:
    """Remove doubling nd coordinates
    
    Parameters
    ----------
    obj: array
        set of 6-dimensional vectors in TAU-style
    
    Returns
    -------
    obj: array
        set of 6-dimensional vectors in TAU-style
    """
    
    ndim=vts.ndim
    shape=vts.shape
    dtype=vts.dtype
    print("ndim",ndim,"shape",shape,"dtype",dtype)
    vt0=qnv.zerov(shape[0])
    print("vt0.shape",vt0.shape)
    ni=0
    for i in range(shape[0]):
        vtsi=vts[i]
        if i==0:
            vt0[ni]=vts[i]
            ni=ni+1
        else:
            isk=False
            for j in range(ni):
                if vts[i]==vt0[j]:
                    isk=True
            if not isk:
                vt0[ni]=vts[i]
                #qnv.printqnv("vt0[ni]",vt0[ni])
                ni=ni+1
    return vt0[0:ni]

def remove_doubling_in_perp_space(vts: qnv.Qnvec) -> qnv.Qnvec:
    """Remove 2D or 3d coordinates which is doubled in Eperp.
    
    Parameters
    ----------
    vts: array
        set of 6-dimensional vectors in TAU-style
    
    Returns
    -------
    vts: array
        set of 2-or 3-dimensional vectors in TAU-style
    """
    return remove_doubling(vts)
    
    #return np.unique(vts)

    #for i in range(num):
    #    pass

#----------------------------
# Edges
#
# Comment：Need to be reorganised.
#----------------------------

#### WIP ###
def get_common_edges(trianges: qnv.Qnvec) -> qnv.Qnvec:
    """Get common edges in trianges
    """
    return 

def generator_all_edges(obj: qnv.Qnvec) -> qnv.Qnvec:
    """Generate all egdes in Object
    
    Parameters
    ----------
    obj: array
        set of triangles or tetrahedra
    
    Returns
    -------
    edges: array
        set of edges in TAU-style.
    
    """
    # (1) preparing a list of edges
    n1,n2=obj.shape
    print("n1",n1,"n2",n2)
    if n2==3:  # triangles
        #edges=np.zeros((n1,3,2,6,3),dtype=np.int64)
        edges=qnv.zerovs((n1,n2))  #[qn0]*(n1,3,2,6)
        i1=0
        for i in range(n1):
            qnv.printqnvs("triangle",obj[i])
            edges[i1]=get_triangle_edge(obj[i])
            i1+=1
        return edges  #edges.reshape(n1*3,2,6)  #edges.reshape(n1*3,2,6,3)
    elif n2==4:  # tetrahedra
        print("edges of tetrahedron is not implemented yet")
        return 
    else:
        print('obj should be a set of trianges or tetrahedra')
        return 

### WIP: to be checked ###
def generator_unique_edges(obj: qnv.Qnvec) -> qnv.Qnvec:
    """Return unique egdes in Object
    
    """
    n1,n2=obj.shape
    
    # (1) preparing a list of edges
    if n2==3: # obj is set of trianges
        edges=generator_all_edges(obj)
    elif n2==2: # obj is set of edges
        edges=obj
    elif n2==4: # obj is set of tetrahedra
        edges=generator_all_edges(obj) 
    else:  
        pass
    
    # (2) 重複のないユニークな辺を得る。
    #print('number of edges:',len(edges))
    num_edges=len(edges)
    #N=obj[0][0].vt[0].N
    qn0=qnn.Qnnum([0,0,1])
    #a=np.zeros((num_edges,3),dtype=np.float64)
    a=np.array((num_edges),dtype=qnv.Qnvec) #[qn0]*(num,edges)
    for i1 in range(num_edges):
        vt=qmt.centroid(edges[i1])
        a[i1]=prj.get_internal_component_numerical(vt)
    b=np.unique(a,return_index=True,axis=0)[1]
    num=len(b)
    #print('number of unique edges:',num)
    #a=np.zeros((num,2,6,3),dtype=np.int64)
    a=np.array((num,2),dtype=qnv.Qnvec)  #[qn0]*(num,2,6)
    for i1 in range(num):
        a[i1]=edges[b[i1]]
    return a

def get_triangle_edge(tri: qnv.Qnvec) -> qnv.Qnvec:
    """Return three edges of triange.
    """
    # three edges of triange: 0-1, 0-2, 1-2
    comb=[\
    [0,1],\
    [0,2],\
    [1,2]] 
    
    # Three egdes of the triangl.
    qn0=qnn.Qnnum([0,0,1])
    #a=np.zeros((3,2,6,3),dtype=np.int64)
    n=crs.n
    a=qnv.zerovs((3,2,n))  #[qn0]*(3,2,n)
    i1=0
    for k in comb:
        i2=0
        for l in k:
            a[i1][i2]=tri[l]
            i2+=1
        i1+=1
    return a

#-------------
# Convex_hull
#-------------
def generate_convex_hull(obj: qnv.Qnvec) -> qnv.Qnvec:
    """generate convex hull from object (a set of triangles)
    
    objの凸包を得る。
    
    アルゴリズム
    1. objの表面三角形を得る。
    2. 無駄な表面三角形をなくす by surface_cleaner()。
    3. objの頂点座標を得る
    4. 三角形分割
    
    objが凸包であれば、この関数を実行することで、よりシンプルに四面体分割されたobjを得ることができる。
    
    """
    # 1
    #tri_surface=generator_surface_1(obj)
    tri_surface=obj
    #print('tri_surface.shape:',tri_surface.shape)
    # 2
    edge_surface=surface_cleaner(tri_surface)
    #print('edge_surface.shape:',edge_surface.shape)
    
    # 3
    vts=remove_doubling_in_perp_space(edge_surface)
    #print('tmp.shape',tmp.shape)
    
    # 4
    return triangulation_points(vts)

def surface_cleaner(surface: qnv.Qnvec) -> qnv.Qnvec:
    """generate border edges from a set of triangles on the objct's surface.
    
    obj表面の三角形からobjの外枠を出力。
    
    アルゴリズム：
    1) 同一平面上にある三角形ごとにグループ分けする
    2) 各グループにおいて、以下を行う．
        2-1) 三角形の３辺が、他のどの三角形とも共有していない辺を求める
        2-3) ２つの辺が１つの辺にまとめられるのであれば、まとめる
        2-3) 辺の集合をアウトプット
    """
    # 同一平面上にある三角形を求め、集合lst_setsとする
    lst_sets=get_sets_of_coplanar_triangles(surface)
    #print('num. of lst_sets:',len(lst_sets))
    
    #同一平面上にある三角形の辺のうち、どの三角形とも共有していない独立な辺を求める．
    for i in range(len(lst_sets)):
        #print('num. of coplanar triangles',len(lst_sets[i]))
        edges=gen_border_edges_of_coplanar_triangles(lst_sets[i])
        if i==0:
            edges_new=edges
        else:
            edges_new=np.vstack([edges_new,edges])
    
    # ２辺を１つの辺にまとめられるのであれば、まとめる
    #print('edges_new.shape',edges_new.shape)
    edges_new=generator_unique_edges(edges_new)
    #print('edges_new.shape',edges_new.shape)
    num=len(edges_new)
    lst0=[i for i in range(num)]
    lst=lst0
    flag=1
    while flag>0:
    #for _ in range(num_iteration):
        counter=0
        #print('lst',lst)
        n0=len(edges_new)
        #print('n0',n0)
        for comb in list(itertools.combinations(lst, 2)):
            a=two_segment_into_one(edges_new[comb[0]],edges_new[comb[1]])
            if np.any(a==None):
                pass
            else:
                counter=1
                break
        if counter==1:
            lst=list(filter(lambda x: x not in list(comb), lst))
            #print('  comb',comb)
            #print('  lst',lst)
            #print('  edges_new.shape',edges_new.shape)
            #print('  a.shape',a.shape)
            edges_new=np.vstack([edges_new,[a]])
            lst.append(num)
            num+=1
            #print('  lst',lst)
        else:
            flag=0
    #print('edges_new.shape',edges_new.shape)
    n1=len(lst)
    #N=surface[0].N
    qn0=qnn.Qnvec([0,0,1])
    #out=np.zeros((n1,2,6,3),dtype=np.int64)
    out=np.array((n1,2),dtype=qnv.Qnvec)  #[qn0]*(n1,2,6,3)
    for i1 in range(n1):
        out[i1]=edges_new[lst[i1]]
    #print('out.shape',out.shape)
    
    return out

def get_sets_of_coplanar_triangles(surface: qnv.Qnvec) -> qnv.Qnvec:
    """
    同一平面上にある三角形の集合を作る。surfaceに含まれるtriangleについて
    順に同一平面上にあるかどうかをチェックし、もし以前のどの三角形とも同一平面
    にない時はそのインデックスをlst_indx_groupに収納する。また、lst_indx_triangle
    にもそのインデックスを収納する。一方、すでにチェックした三角形と同一平面に
    にある場合はlst_indx_triangleに同一平面三角形のインデックスを収納
    """
    num=len(surface)
    lst_indx_group=[0]
    lst_indx_triangle=[0]
    for i1 in range(1,num):
        counter=0
        for i2 in lst_indx_group:
            if coplanar_check_two_triangles(surface[i1],surface[i2]): # coplanar
                counter=1
                break
            else: # non coplanar
                pass
        if counter==0:
            lst_indx_triangle.append(i1)
            lst_indx_group.append(i1)
        else:
            lst_indx_triangle.append(i2)
    lst_sets=[]
    for i1 in lst_indx_group:
        tmp=[]
        for i2 in range(num):
            if i1==lst_indx_triangle[i2]:
                tmp.append(surface[i2])
        num_triangle=len(tmp)
        a=np.zeros((num_triangle,3,6,3),dtype=np.int64)
        for i2 in range(num_triangle):
            a[i2]=tmp[i2]
        lst_sets.append(a)
    return lst_sets

def gen_border_edges_of_coplanar_triangles(coplanar_triangles: qnv.Qnvec) -> qnv.Qnvec:
    """
    同一平面上にある三角形の辺のうち、どの三角形とも共有していない独立な辺を求める．
    """
    unique_edges=generator_unique_edges(coplanar_triangles)
    edges=generator_all_edges(coplanar_triangles)
    #print(' num. of edges:',len(edges))
    #print(' num. of unique_edges:',len(unique_edges))
    lst=[]
    for edge1 in unique_edges:
        counter=0
        for edge2 in edges:
            tmp=np.vstack([edge1,edge2])
            tmp=remove_doubling_in_perp_space(tmp)
            if len(tmp)==2:
                counter+=1
                if counter==2:
                    break
        if counter==1:
            lst.append(edge1)
        else:
            pass
    return lst  #np.array(lst,dtype=np.int64)

#----------------------------
# Equivalence check
#
# Better to merge following four functions into a function "equivalent"???
#   equivalent_triangles
#   equivalent_edges
#   equivalent_vertices
#----------------------------
# WIP:
def equivalent(obj1: qnv.Qnvec, obj2: qnv.Qnvec) -> bool:
    """Checking whether obj1 and obj1 are equivalent or not. 
    """
    def check1(a,b,n):
        n1,_,_=a.shape
        n2,_,_=a.shape
        if n1==n2:
            a=np.vstack([a,b])
            a=remove_doubling_in_perp_space(a)
            if len(a)==n:
                return True # equivalent traiangle
            else:
                return False # not equivalent traiangles
        else:
            return False
    
    def check2(a,b):
        a=prj.projection3(a)
        b=prj.projection3(b)
        if np.all(a==b):
            return True # equivalent traiangle
        else:
            return False
    
    if obj1.ndim==3 and obj2.ndim==3:
        n1,_,_=obj1.shape
        n2,_,_=obj2.shape
        return check1(obj1,obj2,n1)
    elif obj1.ndim==4 and obj2.ndim==4:
        n1,n2,_,_=obj1.shape
        m1,m2,_,_=obj2.shape
        if n1==1 and m1==1:
            obj1=obj1[0]
            obj2=obj2[0]
            n1,_,_=obj1.shape
            n2,_,_=obj2.shape
            return check1(obj1,obj2,n1)
        else:
            return False
    elif obj1.ndim==2 and obj2.ndim==2:
        return check2(obj1,obj2)
    else:
        return 

def equivalent_triangles(tri1: qnv.Qnvec, tri2: qnv.Qnvec) -> bool:
    """Checking whether triangle1 and triangle2 are equivalent or not.
    """
    a=np.concatenate([tri1,tri2])  #np.vstack([tri1,tri2])
    a=remove_doubling_in_perp_space(a)
    if len(a)==3:
        return True # equivalent traiangle
    else:
        return False # not equivalent traiangles

def equivalent_edges(edge1: qnv.Qnvec, edge2: qnv.Qnvec) -> bool:
    """Checking whether edge1 and edge2 are equivalent or not.
    """
    a=np.vstack([edge1,edge2])
    a=remove_doubling_in_perp_space(a)
    if len(a)==2:
        return True # equivalent
    else:
        return False # not equivalent

def equivalent_vertices(vertex1: qnv.Qnvec, vertex2: qnv.Qnvec) -> bool:
    xyz1=prj.projection3(vertex1)
    xyz2=prj.projection3(vertex2)
    if np.all(xyz1==xyz2):
        return True # equivalent
    else:
        return False

#----------------------------
# Sort
#----------------------------
def sort_vctors(vts: qnv.Qnvec) -> qnv.Qnvec:
    """
    sort vectors in TAU-style
    sort the coordinates (xi,yi,zi) such that the xi in the order.
    """
    n1,n2,_=vts.shape
    #out=np.zeros(vts.shape,dtype=np.int64)
    #ln=len(vts)
    #N=vts[0].vt[0].N
    qn0=qnn.Qnnum([0,0,1])
    #qnv1=qnv.Qnvec(ln)
    #out=[qnv1]*ln
    vns=num.get_internal_component_sets_numerical(vts)
    
    ln=len(vns)
    ip=np.zeros(ln,dtype=np.int64)  #[0]*ln
    qmt.qsort(vns,ip,ln)  # qsort in qnmath
    #tmp=np.argsort(vns,axis=0)
    #tmp=vns[np.argsort(vns[:,0])]
    out=qnv.zeros(n1)
    for i1 in range(n1):
        out[i1]=vts[ip[i1][0]]  #vts[tmp[i1][0]]
    return out

def sort_obj(obj: qnv.Qnvec) -> qnv.Qnvec:
    """
    sort triangles in an object
    """
    #out=np.zeros(vts.shape,dtype=np.int64)
    shape=obj.shape
    out=np.array(shape,dtype=qnv.Qnvec)  #[qn0]*vts.shape
    #centroids=np.zeros(len(obj),dtype=np.float64)
    #centroids=np.array(shape,dtype=qnv.Qnvec)  #[qn0]*len(obj)
    #tmp=np.zeros((obj.shape,3),dtype=np.int64)
    #tmp=np.array(shape,dtype=qnv.Qnvec)  #[qn0]*obj.shape
    centroids=qnv.zeros(shape)
    tmp=qnv.zeros(shape)
    
    # 各triangleの頂点xyzをx順にソートすると同時に重心を求めておく。
    for i1 in range(len(obj)):
        tmp[i1]=sort_vctors(obj[i1])
        centroids[i1]=qmt.centroid(obj[i1])
    
    # 三角形の重心xyzのx順にソート
    #indx=np.argsort(centroids,axis=0)
    #indx=centroids[np.argsort(centroids[:,0])] # returns index
    ln=len(centroids)
    indx=np.zeros((ln),dtype=np.int64)
    qmt.qsort(centroids,indx,ln) # get index
    
    for i1 in range(ln):
        out[i1]=tmp[indx[i1][0]]
    return out

#----------------------------
# Triangulation
#----------------------------
def decomposition(tmp2v: qnv.Qnvec):
    try:
        tri=dln.Delaunay(tmp2v) # Delaunay triangulation
    except:
        print('error in decomposition')
        return 
    else:
        out=[]
        for tet in tri.simplices:
            out.append([tet[0],tet[1],tet[2]])
    return out

def triangulation_points(points: qnv.Qnvec):
    #tmp=np.zeros((len(points),2),dtype=np.float64)
    #N=points[0].vt[0].N
    qn0=qnn.Qnnum([0,0,1])
    tmp=qnv.zeros(len(points)) #[qm0]*(len(points),2) 2D vector
    for i1,p in enumerate(points):
        v=prj.projection3(p)  # internal space components of p
        #v=num.numerical_vector(v)
        #tmp[i1]=v[:2]
        tmp[i1]=v # copy v to tmp
        
    ltmp=decomposition(tmp)
    if np.all(ltmp==None):
        return 
    else:
        counter=0
        for i in ltmp:
            #tmp3=np.array([points[i[0]],points[i[1]],points[i[2]]]).reshape(3,6,3)
            tmp3=np.array([points[i[0]],points[i[1]],points[i[2]]]).reshape(3,6) # triangle
            vol=triangle_area_nd(tmp3) # volume (area) of a triangle
            if vol[0]==0 and vol[1]==0:
                pass
            else:
                if counter==0:
                    #tmp1=tmp3.reshape(54) # 3*6*3=54
                    tmp1=tmp3.reshape(18) # 3*6=18
                else:
                    tmp1=np.append(tmp1,tmp3)
                counter+=1
        if counter!=0:
            #return tmp1.reshape(int(len(tmp1)/54),3,6,3) # 3*6*3=54
            #return tmp1.reshape(counter,3,6,3) # 3*6*3=54
            return tmp1.reshape(counter,3,6) # 3*6*3=54
        else:
            return 

##############################
####
####
#### WIP: Removing
####
####
##############################
def remove_vectors(vts1: qnv.Qnvec, vts2: qnv.Qnvec) -> qnv.Qnvec:
    """remove nd vectors in a set vts2 from a set vts1.
    6次元ベクトルリストvts1から6次元ベクトルリストvts2にあるベクトルを抜きとる
    """
    lst=[]
    for i1 in range(len(vts1)):
        counter=0
        for i2 in range(len(vts2)):
            if np.all(vts1[i1]==vts2[i2]):
                counter+=1
                break
        if counter==0:
            lst.append(i1)
    num=len(lst)
    if num!=0:
        #N=vts1[0].N
        qn0=qnn.Qnnum([0,0,1])
        #out=np.zeros((len(lst),6,3),dtype=np.int64)
        out=qnv.zerovs(num)  #[qn0]*(len(lst),6)
        for i1 in range(len(lst)):
            out[i1]=vts1[lst[i1]]
        return out
    else:
        return vts1

def remove_vector(vts: qnv.Qnvec, vt: qnv.Qnvec) -> qnv.Qnvec:
    """ remove a nd vector(vt2) from a set of nd vectors (vts).
    6次元ベクトルリストvlst1から6次元ベクトルvt2を抜きとる
    """
    lst=[]
    for i1 in range(len(vts)):
        counter=0
        if np.all(vts[i1]==vt):
            pass
        else:
            lst.append(vts[i1])
    num=len(lst)
    if num!=0:
        #out=np.zeros((len(lst),6,3),dtype=np.int64)
        #N=vts[0].N
        #qn0=qnn.Qnvec([0,0,1])
        out=qnv.zerovs((num))  #[qn0]*(len(lst),6)
        for i1 in range(len(lst)):
            out[i1]=vts[lst[i1]]
        return out
    else:
        return vts

#################################
####
####
#### WIP: Merging objects
####
####
#################################
def merge_two_triangles_in_obj(obj: qnv.Qnvec) -> qnn.Qnnum:
    num=len(obj)
    return obj

def merge_two_triangles(tri_1: qnv.Qnvec, tri_2: qnv.Qnvec) -> qnv.Qnvec:
    """Return merged tetrahedra.
    """
    if check_connectivity_triangles(tri_1,tri_2): # triangle1とtriangle2が共通する辺を持つ場合
        vtx1=remove_vectors(tri_1,tri_2) # triangle1からtriangle1とtriangle2の共通頂点を消す --> 頂点1
        vtx2=remove_vectors(tri_2,tri_1) # triangle2からtriangle1とtriangle2の共通頂点を消す --> 頂点2
        vtx_common=get_common_edge_in_two_triangles(tri_1,tri_2) # triangle1とtriangle2の共通する辺
        line_segment=np.vstack([vtx1,vtx2])# 頂点1と頂点２を繋いだ辺
        flg=0
        for vtx in vtx_common:
            # 2つのtriangesを一つのtriangeに結合できる時、その頂点は上の辺の2つの頂点のうち1つの頂点と頂点1と頂点２。
            if num.point_on_segment(vtx,line_segment):
                tmp=remove_vector(vtx_common,vtx)
                tri_new=np.stack(tmp,line_segment)
                flg+=1
                break
        else:
            pass
        if flg!=0:
            return tri_new
        else:
            return 
    else:
        return 
    
def check_connectivity_triangles(tri_1: qnv.Qnvec, tri_2: qnv.Qnvec) -> bool:
    """Checking whether triangle_1 and _2 are sharing an edge or not.
    """
    a=np.vstack([tri_1,tri_1])
    a=remove_doubling_in_perp_space(a)
    if len(a)==4:
        return True # common edge
    else:
        return False # not commom edge

def get_common_edge_in_two_triangles(tri_1: qnv.Qnvec, tri_2: qnv.Qnvec) -> qnv.Qnvec:
    """ Return common edge of two connected triangles.
    """
    edges1=get_triangle_edge(tri_1)
    edges2=get_triangle_edge(tri_2)
    
    count=0
    for edge_1 in edges1:
        for edge_2 in edges2:
            if equivalent_edges(edge_1,edge_2): # equivalent
                count+=1
                break
            else:
                pass
        if count!=0:
            break
        else:
            pass
    if count==1:
        return edge_1
    else:
        return 

def two_segment_into_one(line_segment_1: qnv.Qnvec, line_segment_2:qnv.Qnvec) -> qnv.Qnvec:
    
    combination=[\
    [0,1,0,1],\
    [0,1,1,0],\
    [1,0,0,1],\
    [1,0,1,0]]
    
    counter=0
    for comb1 in combination:
        edge1a=line_segment_1[comb1[0]]
        edge1b=line_segment_1[comb1[1]]
        edge2a=line_segment_2[comb1[2]]
        edge2b=line_segment_2[comb1[3]]
        if equivalent_vertices(edge1a,edge2a): # equivalent
        #if equivalent(edge1a,edge2a): # equivalent
            edge_new=np.vstack([[edge1b],[edge2b]])
            #print(out.shape)
            if num.point_on_segment(edge1a,edge_new):
                counter+=1
                break
            else:
                pass
        else:
            pass
    if counter!=0:
        return edge_new
    else:
        return 

def coplanar_check_two_triangles(tri_1: qnv.Qnvec, tri_2: qnv.Qnvec) -> bool:
    """Checking whether two triangles are coplanar or not.
    
    Note
    ----
    Current implementation may return wrong judgement when the cross product of the first two vectors chosen randomly
    are very small in coplanar_check() and coplanar_check_numeric_tau().
    
    vtxはソートされており、coplanar_checkやcoplanar_check_numeric_tauでの外積計算の際に小さい値になるとcoplanar判定を間違うので注意。
    """
    vtx=np.vstack([tri_1,tri_2])
    vtx=remove_doubling_in_perp_space(vtx)
    
    #if coplanar_check(vtx): # in ico2.math1
    if num.coplanar_check_numeric_tau(vtx): # in ico2.numericalc
        return True # coplanar
    else:
        return False


# MICS
def middle_position(pos1: qnv.Qnvec,pos2 :qnv.Qnvec) -> qnv.Qnvec:
    out=pos1+pos2
    out=out/2
#    N=pos1.vt[0].N
#    for i1 in range(6):
#        v=pos1[i1]+pos2[i1]
#        #v=mul(v,np.array([1,0,2]))
#        v=v*qnn.Qnnum([1,0,2]) # middle points
#        if i1!=0:
#            out=np.vstack([out,v])
#        else:
#            out=v.reshape(1,3)
    return out

    