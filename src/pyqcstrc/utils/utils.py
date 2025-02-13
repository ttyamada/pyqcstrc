#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
from numpy.typing import NDArray
#from scipy.spatial import Delaunay
import pyqcstrc.pyDelaunay2D.pyDelaunay2D as dln
import itertools
import time

import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnmat.qnmat as qnm
import pyqcstrc.qnmath.qnmath as qmt
import pyqcstrc.numeric.numericalc as num
import pyqcstrc.prjop.prjop as prj

def shift_object(obj: qnv.Qnvec, shift: qnv.Qnvec) -> qnv.Qnvec:
    """shift an object
    """
    N=obj.N
    qn0=qnn.Qnnum([0,0,1],N)
    if obj.ndim==4:
        obj_new=np.array(obj.shape,dtype=qnv.Qnvec)  #[qn0]*obj.shape
        i1=0
        for tri in obj:
            i2=0
            for vertex in tri:
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
def obj_area_6d(obj: qnv.Qnvec) -> qnn.Qnnum:
    """Calculate volume of an object (set of triangles) in TAU style.
    
    Parameters
    ----------
    obj: array
    
    Returns
    -------
    area: array
        area in TAU-style.
    """
    N=obj.N
    qn0=qnn.Qnnum([0,0,1],N)
    ndim=obj.ndim # ndim qn vector
    w=qn0
    if ndim==4:
        for tri in obj:
            v=triangle_area_6d(tri)
            w=w+v
        return w
    elif ndim==5:
        for tset in obj:
            for tri in tset:
                v=triangle_area_6d(tri)
                w=w+v
        return w
    elif ndim==3:
        return triangle_area_6d(obj)
    else:
        print('object has an incorrect shape!')
        return 

def triangle_area_2d(tri: qnv.Qnvec) -> qnn.Qnnum:
    N=tri[0].N
    qnv=qnv.zerovs(2)
    qnm=qnm.zerom(2,N)
    for i in range(2):
        qnm[i][0]=tri[i+1][0]-tri[0][0]
        qnm[i][1]=tri[i+1][1]-tri[1][0]
    det=qmt.det_matrix_2d(qnm)
    return qnn.abs(det)/2
    
def triangle_sqarea_3d(tri: qnv.Qnvec) -> qnn.Qnnum:
    N=tri[0].N
    qnv=qnv.zerovs(3)
    qnv[0]=tri[1]-tri[0]
    qnv[1]=tri[2]-tri[0]
    qnv[2]=cros(qnv[0],qnv[1])
    det=dot(qnv[2],qnv[2]) 
    return det/4  # squared area of a triangle

def tetrahedron_volume(tet: NDArray[np.float64]) -> qnn.Qnnum:
    """This function returns volume of a tetrahedron
        
    Parameters
    ----------
    tetrahedron: array
        vertex coordinates of the tetrahedron, xyz0,xyz1,xyz2,xyz3
    """
    N=tetrahedron[0].N
    n=3
    qnm=qnm.zeros((n,N))
    for i in range(n):
        qnm[i][0]=tet[i+1][0]-tet[0][0]
        qnm[i][1]=tet[i+1][1]-tet[0][1]
        qnm[i][2]=tet[i+1][2]-tet[0][2]
    det = qtm.det_matrix_3d(qnm)
    return qnn.abs(det)/6

def triangle_area_6d(tri: qnv.Qnvec) -> qnn.Qnnum:
    """Calculate volume of triangle in TAU style.
    
    Parameters
    ----------
    tri: array
        6d vectors of triangle vertices in TAU-style.
    
    Returns
    -------
    area: array
        Area in TAU-style.
    """
    N=tri[0].vt[0].N
    ndim=tri.ndim
    qn0=qnn.Qnnum([0,0,1],N)
    if ndim==3:
        #print('triangle',triangle)
        vts=qnv.zerovs((ndim))  #[qn0]*(3,3)
        for i,vt in enumerate(tri):
            vts[i]=prj.projection3(vt)
        return triangle_area(vts)
    else:
        print('object has an incorrect shape!')
        return 

#######################
###  To be checked  ###
#######################
# this can be replaced by triangle_sqarea_3d
def triangle_area(vts: qnv.Qnvec) -> qnn.Qnnum:
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
    v1=vts[1]-vts[0]
    v2=vts[2]-vts[0]
    v=cross(v1,v2)
    det=dot(v,v)
    N=vts[0].N
    qn0=qnn.Qnnum([0,0,1],N)    
    if det < qn0:  #a1+a2*TAU<0.0: # to avoid negative volume...
        return -det/4 #mul(v[2],np.array([-1,0,2]))
    else:
        return det/4  #mul(v[2],np.array([1,0,2]))

#----------------------------
# Remove doubling
#----------------------------
def remove_doubling(vts: qnv.Qnvec) -> qnv.Qnvec:
    """Remove doubling 6d coordinates
    
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
    vt0=qnv.zerovs((shape[0]))
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
    if n2==3:
        #edges=np.zeros((n1,3,2,6,3),dtype=np.int64)
        edges=qnv.zerovs((n1,n2))  #[qn0]*(n1,3,2,6)
        i1=0
        for tri in obj:
            qnv.printqnv("triangle",tri)
            edges[i1]=get_triangle_edge(tri)
            i1+=1
        return edges  #edges.reshape(n1*3,2,6)  #edges.reshape(n1*3,2,6,3)
    elif n2==4:
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
    N=obj[0][0].vt[0].N
    qn0=qnn.Qnnum([0,0,1],N)
    #a=np.zeros((num_edges,3),dtype=np.float64)
    a=np.array((num),dtype=qnv.Qnvec) #[qn0]*(num,edges)
    for i1 in range(num_edges):
        vt=centroid(edges[i1])
        a[i1]=get_internal_component_numerical(vt)
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
    N=tri[0][0].vt[0].N
    qn0=qnn.Qnnum([0,0,1],N)
    #a=np.zeros((3,2,6,3),dtype=np.int64)
    a=np.array((3,2),dtype=qnv.Qnvec)  #[qn0]*(3,2,6)
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
    N=surface[0].N
    qn0=qnn.Qnvec([0,0,1],N)
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
    N=vts[0].vt[0].N
    qn0=qnn.Qnnum([0,0,1],N)
    #qnv1=qnv.Qnvec(ln,N)
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
        centroids[i1]=centroid(obj[i1])
    
    # 三角形の重心xyzのx順にソート
    #indx=np.argsort(centroids,axis=0)
    #indx=centroids[np.argsort(centroids[:,0])] # returns index
    ln=len(centroids)
    index=[0]*ln
    qmt.qsort(centroids,indx,ln) # get index
    
    for i1 in range(n1):
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
    N=points[0].vt[0].N
    qn0=qnn.Qnnum([0,0,1],N)
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
            vol=triangle_area_6d(tmp3) # volume (area) of a triangle
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
    """remove 6d vectors in a set vts2 from a set vts1.
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
        N=vts1[0].N
        qn0=qnn.Qnnum([0,0,1],N)
        #out=np.zeros((len(lst),6,3),dtype=np.int64)
        out=qnv.zerovs(num)  #[qn0]*(len(lst),6)
        for i1 in range(len(lst)):
            out[i1]=vts1[lst[i1]]
        return out
    else:
        return vts1

def remove_vector(vts: qnv.Qnvec, vt: qnv.Qnvec) -> qnv.Qnvec:
    """ remove a 6d vector(vt2) from a set of 6d vectors (vts).
    6次元ベクトルリストvlst1から6次元ベクトルvt2を抜きとる
    """
    lst=[]
    for i1 in range(len(vts)):
        counter=0
        if np.all(vts[i1]==vt):
            pass
        else:
            lst.append(i1)
    num=len(lst)
    if num!=0:
        #out=np.zeros((len(lst),6,3),dtype=np.int64)
        N=vts[0].N
        qn0=qnn.Qnvec([0,0,1],N)
        out=qnv.zerovs(shape)  #[qn0]*(len(lst),6)
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
            if point_on_segment(vtx,line_segment):
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
    edge1=get_triangle_edge(tri_1)
    edge2=get_triangle_edge(tri_2)
    
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
            if point_on_segment(edge1a,edge_new):
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
    if coplanar_check_numeric_tau(vtx): # in ico2.numericalc
        return True # coplanar
    else:
        return False


# MICS
def middle_position(pos1: qnv.Qnvec,pos2 :qnv.Qnvec):
    N=pos1.vt[0].N
    for i1 in range(6):
        v=pos1[i1]+pos2[i1]
        #v=mul(v,np.array([1,0,2]))
        v=v*qnn.Qnnum([1,0,2].N) #???
        if i1!=0:
            out=np.vstack([tmp2,v])
        else:
            out=v.reshape(1,3)
    return out

if __name__ == '__main__':
        
    N=2 # for octagonal
    n=3
    ns=3
    M0=qnn.Qnnum([0,0,1],N)
    M1=qnn.Qnnum([1,0,1],N)
    M2=qnn.Qnnum([0,1,1],N)
    M3=qnn.Qnnum([1,1,2],N)
    
    qnv0=np.zeros(ns,dtype=qnv.Qnvec) 
    qnv0[0]=qnv.anyv(n,N,[M2,M3,M0])
    qnv0[1]=qnv.anyv(n,N,[M0,M1,M2])
    qnv0[2]=qnv.anyv(n,N,[M1,M2,M0])
    for i in range(3):
        qnv.printqnv("qnv0["+format(i)+"]",qnv0[i])

    qnc0=qnv.cros(qnv0[0],qnv0[1])
    qnv.printqnv("qnc0",qnc0)
    for i in range(3):
        qnn.printqnn("dat(qnc0,qnv0["+format(i)+"])",qnv.dot(qnc0,qnv0[i]))
    #qnn.printqnn("dot(qnc0,qnv0[0])",qnv.dot(qnc0,qnv0[0])) # this should be zero
    #qnn.printqnn("dot(qnc0,qnv0[1])",qnv.dot(qnc0,qnv0[1])) # this should be zero
    #qnn.printqnn("dot(qnc0,qnv0[2])",qnv.dot(qnc0,qnv0[2])) # this should be non-zero

    
    
    #print(vts.shape)
    vinp=np.zeros(ns,dtype=qnn.Qnnum)
    for i in range(ns):
        vinp[i]=qnv.dot(qnv0[i],qnv0[i])
    qnn.printqnns("vinp",vinp)
    
    #qnn.printqnn("dot(qnvo[i],qnv0[i])",vinp[i])
    ip=np.zeros(ns,dtype=np.int64)
    vts1=qmt.qsort(vinp,ip,ns) # use qnmath
    print("ip",ip)
    qnn.printqnns("vinp",vinp)
    qnv.printqnv("vts1",vts1)
    
    # nD lattice vector for defining ODs
    n=5
    N=2
    # 8 corner vectors for AB tiling OD
    vts2=qnv.zerovs((8)) # for octagon for Ammann-Beenker tiling
    M0=qnn.Qnnum([0,0,1],N)
    M1=qnn.Qnnum([1,0,2],N)  # 1
    M2=qnn.Qnnum([-1,0,2],N) # -1
    M3=qnn.Qnnum([0,1,4],N)  # sqrt(2)/2
    M4=qnn.Qnnum([0,-1,4],N) # -sqrt(2)/2
    # AB OD edge vectors in qnnum
    vts2[0]=qnv.anyv(n,N,[M1,M0,M0,M1,M0]) #(1 0 0 1 0)/2
    vts2[1]=qnv.anyv(n,N,[M0,M0,M2,M1,M0]) #(0 0 -1 1 0)/2
    vts2[2]=qnv.anyv(n,N,[M0,M1,M2,M0,M0]) #(0 1 -1 0 0)/2
    vts2[3]=qnv.anyv(n,N,[M2,M1,M0,M0,M0]) #(-1 1 0 0 0)/2
    vts2[4]=qnv.anyv(n,N,[M2,M0,M0,M2,M0]) #(-1 0 0 -1 0)/2
    vts2[5]=qnv.anyv(n,N,[M0,M0,M1,M2,M0]) #(0 0 1 -1 0)/2
    vts2[6]=qnv.anyv(n,N,[M0,M2,M1,M0,M0]) #(0 -1 1 0 0)/2
    vts2[7]=qnv.anyv(n,N,[M1,M2,M0,M0,M0]) #(1 -1 0 0 0)/2
    qnv.printqnvs("vts2",vts2)
    
    isys=4
    prj.prjop_init(isys)
    
    # calculate internal space components of vts2
    print("vts2.shape",vts2.shape)
    vns2=num.get_internal_component_sets_numerical(vts2) # perp space components
    qnv.printqnvs("vts2",vts2)
    #for i in range(8):
    #    str="vts2["+format(i)+"]"
    #    qnv.printqnv(str,vts2[i])
    
    #================
    # 重複のテスト
    #================
    n=5
    #vst=generate_random_vectors(nset)
    vst=qnv.zerovs((n))
    #for i in range(nset):
    #    vst[i]=qnv.Qnvec(n,N)
    # set vt values
    vst[0]=qnv.anyv(2,N,[M0,M1])
    vst[1]=qnv.anyv(2,N,[M1,M2])
    vst[2]=qnv.anyv(2,N,[M1,M3])
    vst[3]=qnv.anyv(2,N,[M0,M3])
    vst[4]=qnv.anyv(2,N,[M2,M1])
    print("vst.shape",vst.shape)
    qnv.printqnvs("vst",vst)
    
    vst_d3=np.concatenate([vst,vst]) # doubling vst vectors
    print("vst_d3.shape",vst_d3.shape) # for test
    qnv.printqnvs("vst_d3",vst_d3)
    #vst_d4=np.stack([vst_d3,vst_d3]) # doubling vst_d3 vectors
    #qnv.printqnvs("vst_d4",vst_d4)
    
    a=remove_doubling(vst_d3)
    qnv.printqnvs("a",a)
    
    # projection operator
    qnm.printqnm("prj.prj0",prj.prj0)
    
    vst_d4=np.concatenate([vts2,vts2])  # 5D vectors
    qnv.printqnvs("vst_d4",vst_d4)
                  
    vst_d5=qnv.zerovs((16))
    for i in range(16):
        vst_d5[i]=prj.prjop(vst_d4[i]) # 
    qnv.printqnvs("vst_d5",vst_d5)
    
    vst_d6=qnv.zerovs((16))
    for i in range(16):
        vst_d6[i]=prj.prjop_e(vst_d4[i]) # 
    qnv.printqnvs("vst_d6",vst_d6)
    
    vst_d7=qnv.zerovs((16))
    for i in range(16):
        vst_d7[i]=prj.prjop_i(vst_d4[i]) # 
    qnv.printqnvs("vst_d7",vst_d7)
    
    a=remove_doubling_in_perp_space(vst_d4)
    if len(a)==n:
        print('remove_doubling_in_perp_space: pass')
    else:
        print('remove_doubling_in_perp_space: error')
    
    #================
    # 面と辺のテスト
    #================
    #triangle=generate_random_triangle()
    
    # generate triangles
    tri=qnv.zerovs(3)
    print("triangle.shape",tri.shape)
    qnv.printqnv("vst[0]",vst[0])
    qnv.printqnv("vst[1]",vst[1])
    qnv.printqnv("vst[2]",vst[2])
    tri[0]=vst[0]
    tri[1]=vst[1]
    tri[2]=vst[2] # 3 vectors define a triangle
    qnv.printqnvs("triangle",tri) # triangle
    # doubled triangle
    obj=np.concatenate([tri,tri]) # doubled triangle
    #generator_surface_1(obj)
    
    # a tetrahedon
    #obj=triangle
    
    #surface=generator_surface_1(obj.reshape(1,3,6,3))
    #surface=obj.reshape(1,3,6,3)
    print("obj.shape",obj.shape)
    #surface=obj.reshape(1,6)
    surface=obj
    #generator_edge(surface)
    generator_all_edges(surface)
    
    