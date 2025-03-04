#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import os
from octa2.projection import (projection3,
                              distance_in_perp_space,
                              )

from octa2.math1 import (add,
                        sub,
                        mul,
                        div,
                        add_vectors,
                        sub_vectors,
                        outer_product,
                        inner_product,
                        centroid,
                        #coplanar_check,
                        )
from octa2.numericalc import (numeric_value,
                            numerical_vector,
                            numerical_vectors,
                            point_on_segment,
                            coplanar_check_numeric_tau,
                            get_internal_component_numerical,
                            get_internal_component_sets_numerical,
                            )

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
import itertools
import time

TAU=np.sqrt(2)
#DTYPE_int = int
DTYPE_int =np.int64
#TYPE2D_int =cython.long[:,:]

def shift_object(obj: NDArray[DTYPE_int], shift: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """shift an object
    """
    if obj.ndim==4:
        obj_new=np.zeros(obj.shape,dtype=DTYPE_int)
        i1=0
        for triangle in obj:
            i2=0
            for vertex in triangle:
                obj_new[i1][i2]=add_vectors(vertex,shift)
                i2+=1
            i1+=1
        return obj_new
    else:
        print('object has an incorrect shape!')
        return 

#----------------------------
# Volume, area
#----------------------------
def obj_area_6d(obj: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Calculate volume of an object (set of triangles) in TAU style.
    
    Parameters
    ----------
    obj: array
    
    Returns
    -------
    area: array
        area in TAU-style.
    """
    w=np.array([0,0,1])
    if obj.ndim==4:
        for triangle in obj:
            v=triangle_area_6d(triangle)
            w=add(w,v)
        return w
    elif obj.ndim==5:
        for tset in obj:
            for triangle in tset:
                v=triangle_area_6d(triangle)
                w=add(w,v)
        return w
    elif obj.ndim==3:
        return triangle_area_6d(obj)
    else:
        print('object has an incorrect shape!')
        return 

def triangle_area_6d(triangle: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Calculate volume of triangle in TAU style.
    
    Parameters
    ----------
    obj: array
        6d vectors of triangle vertices in TAU-style.
    
    Returns
    -------
    area: array
        Area in TAU-style.
    """
    if triangle.ndim==3:
        #print('triangle:',triangle)
        vts=np.zeros((3,3,3),dtype=DTYPE_int)
        for i,vt in enumerate(triangle):
            vts[i]=projection3(vt)
        return triangle_area(vts)
    else:
        print('object has an incorrect shape!')
        return 

#######################
###  To be checked  ###
#######################
def triangle_area(vts: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
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
    v1=sub_vectors(vts[1],vts[0])
    v2=sub_vectors(vts[2],vts[0])
    
    v=outer_product(v1,v2)
    
    a1=v[2][0]
    a2=v[2][1]
    a3=v[2][2]
    
    if a1+a2*TAU<0.0: # to avoid negative volume...
        return mul(v[2],np.array([-1,0,2]))
    else:
        return mul(v[2],np.array([1,0,2]))

#----------------------------
# Remove doubling
#----------------------------
def remove_doubling(vts: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
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
    if ndim==4:
        n1,n2,_,_=vts.shape
        num=n1*n2
        vts=vts.reshape(num,6,3)
        return np.unique(vts,axis=0)
    elif ndim==3:
        return np.unique(vts,axis=0)
    else:
        print('ndim should be 3 or 4.')
        return 

def remove_doubling_in_perp_space(vts: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Remove 6d coordinates which is doubled in Eperp.
    
    Parameters
    ----------
    vts: array
        set of 6-dimensional vectors in TAU-style
    
    Returns
    -------
    vts: array
        set of 6-dimensional vectors in TAU-style
    """
    ndim=vts.ndim
    if ndim==4:
        n1,n2,_,_=vts.shape
        num=n1*n2
        vst=vts.reshape(num,6,3)
    elif ndim==3:
        #num,_,_=vts.shape
        pass
    
    # first run remove_doubling()
    vts=remove_doubling(vts)
    num=len(vts)
    
    # then, remove doubling in perp space.
    a=np.zeros((num,3,3),dtype=DTYPE_int)
    for i in range(num):
        a[i]=projection3(vts[i])
    b=np.unique(a,return_index=True,axis=0)[1]
    num=len(b)
    a=np.zeros((num,6,3),dtype=DTYPE_int)
    for i in range(num):
        a[i]=vts[b[i]]
    return a

#----------------------------
# Edges
#
# Comment：Need to be reorganised.
#----------------------------

#### WIP ###
def get_common_edges(trianges: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Get common edges in trianges
    """
    return 

def generator_all_edges(obj: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Generate all egdes in Object
    
    Parameters
    ----------
    obj: array
        set of triangles
    
    Returns
    -------
    edges: array
        set of edges in TAU-style.
    
    """
    
    # (1) preparing a list of edges
    n1,n2,_,_=obj.shape
    if n2==3:
        edges=np.zeros((n1,3,2,6,3),dtype=DTYPE_int)
        i1=0
        for triangle in obj:
            edges[i1]=get_triangle_edge(triangle)
            i1+=1
        return edges.reshape(n1*3,2,6,3)
    else:
        print('obj should be a set of trianges')
        return 

### WIP: to be checked ###
def generator_unique_edges(obj: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Return unique egdes in Object
    
    """
    n1,n2,_,_=obj.shape
    
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
    a=np.zeros((num_edges,3),dtype=np.float64)
    for i1 in range(num_edges):
        vt=centroid(edges[i1])
        a[i1]=get_internal_component_numerical(vt)
    b=np.unique(a,return_index=True,axis=0)[1]
    num=len(b)
    #print('number of unique edges:',num)
    a=np.zeros((num,2,6,3),dtype=DTYPE_int)
    for i1 in range(num):
        a[i1]=edges[b[i1]]
    return a

def get_triangle_edge(triangle: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Return three edges of triange.
    """
    # three edges of triange: 0-1, 0-2, 1-2
    comb=[\
    [0,1],\
    [0,2],\
    [1,2]] 
    
    # Three egdes of the triangl.
    a=np.zeros((3,2,6,3),dtype=DTYPE_int)
    i1=0
    for k in comb:
        i2=0
        for l in k:
            a[i1][i2]=triangle[l]
            i2+=1
        i1+=1
    return a

#-------------
# Convex_hull
#-------------
def generate_convex_hull(obj: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
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
    #triangle_surface=generator_surface_1(obj)
    triangle_surface=obj
    #print('triangle_surface.shape:',triangle_surface.shape)
    # 2
    edge_surface=surface_cleaner(triangle_surface)
    #print('edge_surface.shape:',edge_surface.shape)
    
    # 3
    vts=remove_doubling_in_perp_space(edge_surface)
    #print('tmp.shape',tmp.shape)
    
    # 4
    return triangulation_points(vts)

def surface_cleaner(surface: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
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
    out=np.zeros((n1,2,6,3),dtype=DTYPE_int)
    for i1 in range(n1):
        out[i1]=edges_new[lst[i1]]
    #print('out.shape',out.shape)
    
    return out

def get_sets_of_coplanar_triangles(surface: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
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
        a=np.zeros((num_triangle,3,6,3),dtype=DTYPE_int)
        for i2 in range(num_triangle):
            a[i2]=tmp[i2]
        lst_sets.append(a)
    return lst_sets

def gen_border_edges_of_coplanar_triangles(coplanar_triangles: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
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
    return np.array(lst,dtype=DTYPE_int)

#----------------------------
# Equivalence check
#
# Better to merge following four functions into a function "equivalent"???
#   equivalent_triangles
#   equivalent_edges
#   equivalent_vertices
#----------------------------
# WIP:
def equivalent(obj1: NDArray[DTYPE_int], obj2: NDArray[DTYPE_int]) -> bool:
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
        a=projection3(a)
        b=projection3(b)
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

def equivalent_triangles(triangle1: NDArray[DTYPE_int], triangle2: NDArray[DTYPE_int]) -> bool:
    """Checking whether triangle1 and triangle2 are equivalent or not.
    """
    a=np.vstack([triangle1,triangle2])
    a=remove_doubling_in_perp_space(a)
    if len(a)==3:
        return True # equivalent traiangle
    else:
        return False # not equivalent traiangles

def equivalent_edges(edge1: NDArray[DTYPE_int], edge2: NDArray[DTYPE_int]) -> bool:
    """Checking whether edge1 and edge2 are equivalent or not.
    """
    a=np.vstack([edge1,edge2])
    a=remove_doubling_in_perp_space(a)
    if len(a)==2:
        return True # equivalent
    else:
        return False # not equivalent

def equivalent_vertices(vertex1: NDArray[DTYPE_int], vertex2: NDArray[DTYPE_int]) -> bool:
    xyz1=projection3(vertex1)
    xyz2=projection3(vertex2)
    if np.all(xyz1==xyz2):
        return True # equivalent
    else:
        return False

#----------------------------
# Sort
#----------------------------
def sort_vctors(vts: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """
    sort vectors in TAU-style
    
    sort the coordinates (xi,yi,zi) such that the xi in the order.
    """
    n1,n2,_=vts.shape
    out=np.zeros(vts.shape,dtype=DTYPE_int)
    vns=get_internal_component_sets_numerical(vts)
    
    tmp=np.argsort(vns,axis=0)
    #tmp=vns[np.argsort(vns[:,0])]
    for i1 in range(n1):
        out[i1]=vts[tmp[i1][0]]
    return out

def sort_obj(obj: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """
    sort triangle in an object
    """
    out=np.zeros(obj.shape,dtype=DTYPE_int)
    centroids=np.zeros(len(obj),dtype=DTYPE_int)
    tmp=np.zeros((obj.shape,3),dtype=DTYPE_int)
    
    # 各triangleの頂点xyzをx順にソートすると同時に重心を求めておく。
    for i1 in range(len(obj)):
        tmp[i1]=sort_vctors(obj[i1])
        centroids[i1]=centroid(obj[i1])
    
    # 三角形の重心xyzのx順にソート
    #indx=np.argsort(centroids,axis=0)
    indx=centroids[np.argsort(centroids[:,0])]
    
    #for i1 in range(n1):
    for i1 in range(len(obj)):
        out[i1]=tmp[indx[i1][0]]
    return out

#----------------------------
# Triangulation
#----------------------------
def decomposition(tmp2v: NDArray[np.float64]):
    try:
        tri=Delaunay(tmp2v)
    except:
        print('error in decomposition')
        return 
    else:
        out=[]
        for tet in tri.simplices:
            out.append([tet[0],tet[1],tet[2]])
    return out

def triangulation_points(points: NDArray[DTYPE_int]):
    
    tmp=np.zeros((len(points),2),dtype=np.float64)
    for i1,p in enumerate(points):
        v=projection3(p)
        v=numerical_vector(v)
        tmp[i1]=v[:2]
        
    ltmp=decomposition(tmp)
    if np.all(ltmp==None):
        return 
    else:
        counter=0
        for i in ltmp:
            tmp3=np.array([points[i[0]],points[i[1]],points[i[2]]]).reshape(3,6,3)
            vol=triangle_area_6d(tmp3)
            if vol[0]==0 and vol[1]==0:
                pass
            else:
                if counter==0:
                    tmp1=tmp3.reshape(54) # 3*6*3=54
                else:
                    tmp1=np.append(tmp1,tmp3)
                counter+=1
        if counter!=0:
            #return tmp1.reshape(int(len(tmp1)/54),3,6,3) # 3*6*3=54
            return tmp1.reshape(counter,3,6,3) # 3*6*3=54
        else:
            return 

##############################
####
####
#### WIP: Removing
####
####
##############################
def remove_vectors(vts1: NDArray[DTYPE_int], vts2: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
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
        out=np.zeros((len(lst),6,3),dtype=DTYPE_int)
        for i1 in range(len(lst)):
            out[i1]=vts1[lst[i1]]
        return out
    else:
        return vts1

def remove_vector(vts: NDArray[DTYPE_int], vt: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
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
        out=np.zeros((len(lst),6,3),dtype=DTYPE_int)
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
def merge_two_triangles_in_obj(obj: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    
    num=len(obj)
    
    
    return obj

def merge_two_triangles(triangle_1: NDArray[DTYPE_int], triangle_2: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """Return merged tetrahedra.
    """
    if check_connectivity_triangles(triangle_1,triangle_2): # triangle1とtriangle2が共通する辺を持つ場合
        vtx1=remove_vectors(triangle_1,triangle_2) # triangle1からtriangle1とtriangle2の共通頂点を消す --> 頂点1
        vtx2=remove_vectors(triangle_2,triangle_1) # triangle2からtriangle1とtriangle2の共通頂点を消す --> 頂点2
        vtx_common=get_common_edge_in_two_triangles(triangle_1,triangle_2) # triangle1とtriangle2の共通する辺
        line_segment=np.vstack([vtx1,vtx2])# 頂点1と頂点２を繋いだ辺
        flg=0
        for vtx in vtx_common:
            # 2つのtriangesを一つのtriangeに結合できる時、その頂点は上の辺の2つの頂点のうち1つの頂点と頂点1と頂点２。
            if point_on_segment(vtx,line_segment):
                tmp=remove_vector(vtx_common,vtx)
                triange_new=np.stack(tmp,line_segment)
                flg+=1
                break
        else:
            pass
        if flg!=0:
            return triange_new
        else:
            return 
    else:
        return 
    
def check_connectivity_triangles(triangle_1: NDArray[DTYPE_int], triangle_2: NDArray[DTYPE_int]) -> bool:
    """Checking whether triangle_1 and _2 are sharing an edge or not.
    """
    a=np.vstack([triangle_1,triangle_1])
    a=remove_doubling_in_perp_space(a)
    if len(a)==4:
        return True # common edge
    else:
        return False # not commom edge

def get_common_edge_in_two_triangles(triangle_1: NDArray[DTYPE_int], triangle_2: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    """ Return common edge of two connected triangles.
    """
    edges1=get_triangle_edge(triangle_1)
    edges2=get_triangle_edge(triangle_2)
    
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

def two_segment_into_one(line_segment_1: NDArray[DTYPE_int], line_segment_2: NDArray[DTYPE_int]) -> NDArray[DTYPE_int]:
    
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

def coplanar_check_two_triangles(triange1: NDArray[DTYPE_int], triange2: NDArray[DTYPE_int]) -> bool:
    """Checking whether two triangles are coplanar or not.
    
    Note
    ----
    Current implementation may return wrong judgement when the cross product of the first two vectors chosen randomly
    are very small in coplanar_check() and coplanar_check_numeric_tau().
    
    vtxはソートされており、coplanar_checkやcoplanar_check_numeric_tauでの外積計算の際に小さい値になるとcoplanar判定を間違うので注意。
    """
    vtx=np.vstack([triange1,triange2])
    vtx=remove_doubling_in_perp_space(vtx)
    
    #if coplanar_check(vtx): # in ico2.math1
    if coplanar_check_numeric_tau(vtx): # in ico2.numericalc
        return True # coplanar
    else:
        return False


# MICS
def middle_position(pos1,pos2):
    tmp2=np.zeros((6),dtype=DTYPE_int)
    for i1 in range(6):
        v=add(pos1[i1],pos2[i1])
        v=mul(v,np.array([1,0,2])) #1/2
        tmp2[i1]=v
        #if i1!=0:
        #    out=np.vstack([tmp2,v])
        #else:
        #    out=v.reshape(1,3)
    return tmp2  #out

#----------------------------
# Surface, trianges, and edges
#
# Comment：Need to be reorganised.
#----------------------------
#def generator_surface_1(obj: NDArray[DTYPE_int], verbose: int=0) -> NDArray[DTYPE_int]:
#    """Generate triangles of the object's surface.
#    
#    Parameters
#    ----------
#    obj: array
#    
#    Returns
#    -------
#    surfaces: array
#        set of surface triangles in TAU-style.
#    """
#    # (1) preparing a list of triangle surfaces without doubling (tmp2)
#    n1,_,_,_=obj.shape
#    triangles=np.zeros((n1,4,3,6,3),dtype=DTYPE_int)
#    i1=0
#    
#    if verbose>0:
#        print('      generator_surface_1 part-1')
#        start=time.time()
#    #
#    #for tetrahedron in obj:
#    for tri in obj:
#        #triangles[i1]=get_tetrahedron_surface(tetrahedron)
#        triangles[i1]=get_triangle_surface(tri)
#        i1+=1
#    triangles=triangles.reshape(n1*4,3,6,3)
#    #
#    if verbose>0:
#        end=time.time()
#        time_diff=end-start
#        print('         ends in %4.3f sec'%time_diff)
#    
#    
#    if n1==1:
#        return triangles
#    else:
#        if verbose>0:
#            print('      generator_surface_1 part-2')
#            start=time.time()
#        #
#        
#        # (2) 重複のない三角形（すなはちobject表面の三角形）のみを得る。
#        # 三角形が重複していれば重心も同じことを利用する。重心が一致すれば重複しているとは限らないが、
#        # objが正しく与えられているとすれば問題ない。
#        #
#        # まず重心xyzを求める
#        xyz=np.zeros((n1*4,3),dtype=np.float64)
#        for i1 in range(n1*4):
#            vt=centroid(triangles[i1])
#            xyz[i1]=get_internal_component_numerical(vt)
#        
#        """
#        # 以下のやり方では効率悪い。
#        # triangleの数が多ければ時間がかかる(O(n^2))ので改良が必要
#        #===========ここから============
#        # xyzをxでソートし、indexを得る。
#        indx_xyz=np.argsort(xyz[:,0])
#        #print('number of trianges:',len(indx_xyz))
#        #print('indx_xyz:',indx_xyz)
#        #print(xyz[indx_xyz])
#        
#        # 重複しているtriangleはスキップ。表面のtriangleのみを選び出す。
#        lst=[]
#        for i1 in indx_xyz:
#            counter=0
#            for i2 in indx_xyz:
#                if i1==i2:
#                    pass
#                else:
#                    if np.allclose(xyz[i1],xyz[i2]): # equivalent
#                        counter+=1
#                        break
#            if counter==0:
#                lst.append(i1)
#        #===========ここまで============
#        """
#        #"""
#        # xyz座標のソートをx,y,zに対して行い、重複チェックを効率化(O(n))。
#        #===========ここから============
#        # xyzをx,y,zの順で優先的にソートし、indexを得る。
#        indx_xyz=np.lexsort((xyz[:,2],xyz[:,1],xyz[:,0]))
#        #print('number of trianges:',len(indx_xyz))
#        #print('indx_xyz:',indx_xyz)
#        
#        # 表面のtriangleのみを選び出すには、重複しているtriangleを除けば良い。
#        # 上でxyzをx,y,zの順で優先的にソートできていれば、着目している点をその前後と比べるだけで重複があるか判断できる。
#        lst=[]
#        if np.allclose(xyz[indx_xyz[0]],xyz[indx_xyz[1]]):
#            pass
#        else:
#            lst.append(indx_xyz[0])
#        for i1 in range(1,len(indx_xyz)-1):
#            counter=0
#            for i2 in [-1,1]:
#                if np.allclose(xyz[indx_xyz[i1]],xyz[indx_xyz[i1+i2]]): # equivalent
#                    counter+=1
#                    break
#            if counter==0:
#                lst.append(indx_xyz[i1])
#        if np.allclose(xyz[indx_xyz[-1]],xyz[indx_xyz[-2]]):
#            pass
#        else:
#            lst.append(indx_xyz[-1])
#        #===========ここまで============
#        #"""
#        
#        #print('lst:',lst)
#        num=len(lst)
#        #print('num:',num)
#        out=np.zeros((num,3,6,3),dtype=DTYPE_int)
#        #print('number of unique triangls:',num)
#        for i1 in range(num):
#            out[i1]=triangles[lst[i1]]
#        #print('shape:',out.shape)
#        
#        if verbose>0:
#            end=time.time()
#            time_diff=end-start
#            print('         ends in %4.3f sec'%time_diff)
#        
#        return out

def write(obj:NDArray[DTYPE_int]=None,path=None,basename=None,format=None,color='k',select=None,verbose=0):
    """
    Export occupation domains.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        path (str): Path of the output XYZ file
        basename (str): Basename of the output XYZ file
        format (str): format of output file
            format = 'xyz' (default)
            format = 'vesta'
        color (str)
            one of the characters {'k','r','b','p'}, which are short-hand notations 
            for shades of black, red, blue, and pink, in case where 'vesta' format is
            selected (default, color = 'k').
        select (str):'simple', 'normal', or 'egdes'
            'simple': Merging triangles into one single objecte
            'normal': Each triangle is set as single objecte (large file)
            'egdes':  Select this option when the obj is a set of edges.
    
    Returns:
        int: 0 (succeed), 1 (fail)
    
    """
    if os.path.exists(path)==False:
        os.makedirs(path)
    else:
        pass
    
    if np.all(obj==None):
        print('    Empty OD')
        return 0
    else:
        if format=='vesta':
            if select==None:
                select='normal'
            write_vesta(obj,path,basename,color,select,verbose)
            return 0
        elif format == 'xyz':
            if select==None:
                select='triangle'
            write_xyz(obj,path,basename,select,verbose)
            return 0
        else:
            return 1

def write_vesta(obj:NDArray[DTYPE_int]=None,path='.',basename='tmp',color='k',select='normal',verbose=0):
    """
    Export occupation domains in VESTA format.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        path (str): Path of the output XYZ file
        basename (str): Basename of the output XYZ file
        color (str)
            one of the characters {'k','r','b','p','l','y','c','s'}, which are short-hand notations 
            for shades of black, red, blue, pink, lime, yellow, cyan, and silver in case where 'vesta' format is
            selected (default, color = 'k').
        select (str):'simple', 'normal', 'egdes', or 'podatm'
            'simple': Merging triangles into one single objecte
            'normal': Each triangle is set as single objecte (large file)
            'egdes':  Select this option when the obj is a set of edges.
            'podatm': same as 'simple' but return "vertices" necessary to input 
            (default, select = 'normal')
    Returns:
        int: 0 (succeed), 1 (fail) when select = 'simple' or 'normal'.
        ndarray: vertices, when select = 'podatm'.
    """
    #print('write_vesta()')
    #print("obj.shape",obj.shape)
    
    if os.path.exists(path)==False:
        os.makedirs(path)
    else:
        pass
    
    def colors(code):
        if code=='red' or code=='r':
            a = [255,0,0]
        elif code=='blue' or code=='b':
            a = [0,0,255]
        elif code=='black' or code=='k':
            a = [127,127,127]
        elif code=='pink' or code=='p':
            a = [255,0,255]
        elif code=='lime' or code=='l':
            a = [0,255,0]
        elif code=='yellow' or code=='y':
            a = [255,255,0]
        elif code=='cyan' or code=='c':
            a = [0,255,255]
        elif code=='silver' or code=='s':
            a = [192,192,192]
        elif len(code)==3:
            a = code
        else:
            a = [127,127,127]
        return a
    
    file_name='%s/%s.vesta'%(path,basename)
    f=open('%s'%(file_name),'w')
    
    #dmax=5.0
    dmax=10.0
    
    if select=='simple' or select=='egdes':
        if np.all(obj==None):
            print('no volume obj')
            return 0
        else:
            # get independent edges
            if select=='simple':
                edges = utils.generator_obj_edge(obj,verbose)
            else:
                edges = obj
            # get independent vertices of the edges
            vertices = remove_doubling_in_perp_space(edges)
                
            # get bond pairs, [[distance, XXX, YYY],...]
            pairs = []
            for edge in edges:
                dist=distance_in_perp_space(edge[0],edge[1])
                a=[dist]
                for i2 in range(2):
                    for i3,vt in enumerate(vertices):
                        tmp=np.vstack([edge[i2],vt])
                        tmp=remove_doubling_in_perp_space(tmp.reshape(2,6,3))
                        if len(tmp)==1:
                            a.append(i3)
                            break
                        else:
                            pass
                pairs.append(a)
                
            print('#VESTA_FORMAT_VERSION 3.5.0\n', file=f)
            print('MOLECULE\
            \nTITLE',file=f)
            print('%s/%s\n'%(path,basename), file=f)
            print('GROUP\
            \n1 1 Custom\
            \nSYMOP\
            \n 0.000000  0.000000  0.000000  1  0  0    0  1  0    0  0  1    1\
            \n -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0\
            \nTRANM 0\
            \n 0.000000  0.000000  0.000000  1  0  0    0  1  0    0  0  1\
            \nLTRANSL\
            \n -1\
            \n 0.000000  0.000000  0.000000  0.000000  0.000000  0.000000\
            \nLORIENT\
            \n -1    0    0    0    0\
            \n 1.000000  0.000000  0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000  0.000000  1.000000\
            \nLMATRIX\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000  0.000000\
            \nCELLP\
            \n  1.000000    1.000000    1.000000  90.000000  90.000000  90.000000\
            \n  0.000000    0.000000    0.000000    0.000000    0.000000    0.000000\
            \nSTRUC', file=f)
            for i2,vrtx in enumerate(vertices):
                xyz = projection3(vrtx)
                xyz=numerical_vector(xyz)
                print('%4d A        A%d  1.0000    %8.6f %8.6f %8.6f        1'%\
                (i2+1,i2+1,xyz[0],xyz[1],xyz[2]), file=f)
                print('                             0.000000    0.000000    0.000000  0.00', file=f)
            print('  0 0 0 0 0 0 0\
            \nTHERI 0', file = f)
            i2=0
            for __ in vertices:
                print('  %d        A%d  1.000000'%(i2+1,i2+1), file=f)
                i2+=1
            print('  0 0 0\
            \nSHAPE\
            \n  0         0         0         0    0.000000  0    192    192    192    192\
            \nBOUND\
            \n         0          1        0          1        0          1\
            \n  0    0    0    0  0\
            \nSBOND', file = f)
            clr=colors(color)
            for i2,pair in enumerate(pairs):
                print('  %d   A%d   A%d   %8.6f   %8.6f  0  1  1  1  2  0.100  2.000 %3d %3d %3d'%(\
                i2+1, pair[1]+1, pair[2]+1, pair[0]-0.01, pair[0]+0.01, clr[0], clr[1], clr[2]), file=f)
            print('  0 0 0 0\
            \nSITET', file = f)
            for i2 in range(len(vertices)):
                print('    %d        A%d  0.050  76  76  76  76  76  76 204  0'%(i2+1,i2+1), file=f)
            print('  0 0 0 0 0 0\
            \nVECTR\
            \n 0 0 0 0 0\
            \nVECTT\
            \n 0 0 0 0 0\
            \nSPLAN\
            \n  0    0    0    0\
            \nLBLAT\
            \n -1\
            \nLBLSP\
            \n -1\
            \nDLATM\
            \n -1\
            \nDLBND\
            \n -1\
            \nDLPLY\
            \n -1\
            \nPLN2D\
            \n  0    0    0    0', file = f)
        
            print('ATOMT\
            \n  1        A  0.0100  76  76  76  76  76  76 204\
            \n  0 0 0 0 0 0\
            \nSCENE\
            \n 1.000000 -0.000000 -0.000000  0.000000\
            \n 0.000000  1.000000 -0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n  0.000    0.000\
            \n  0.000\
            \n  1.320\
            \nHBOND 0 2\
            \n\
            \nSTYLE\
            \nDISPF 37753794\
            \nMODEL    0  1  0\
            \nSURFS    0  1  1\
            \nSECTS  32  1\
            \nFORMS    0  1\
            \nATOMS    0  0  1\
            \nBONDS    2\
            \nPOLYS    1\
            \nVECTS 1.000000\
            \nFORMP\
            \n  1  1.0    0    0    0\
            \nATOMP\
            \n 24  24    0  50  2.0    0\
            \nBONDP\
            \n  3  16  0.250  2.000 127 127 127\
            \nPOLYP\
            \n 204 1  1.000 180 180 180\
            \nISURF\
            \n  0    0    0    0\
            \nTEX3P\
            \n  1  0.00000E+00  1.00000E+00\
            \nSECTP\
            \n  1  5.00000E-01  5.00000E-01  0.00000E+00  0.00000E+00  0.00000E+00  0.00000E+00\
            \nCONTR\
            \n 0.1 -1 1 1 10 -1 2 5\
            \n 2 1 2 1\
            \n    0    0    0\
            \n    0    0    0\
            \n    0    0    0\
            \n    0    0    0\
            \nHKLPP\
            \n 192 1  1.000 255    0 255\
            \nUCOLP\
            \n    0    1  1.000    0    0    0\
            \nCOMPS 0\
            \nLABEL 1     12  1.000 0\
            \nPROJT 0  0.962\
            \nBKGRC\
            \n 255 255 255\
            \nDPTHQ 1 -0.5000  3.5000\
            \nLIGHT0 1\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n  26  26  26 255\
            \n 179 179 179 255\
            \n 255 255 255 255\
            \nLIGHT1\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nLIGHT2\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nLIGHT3\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nATOMM\
            \n 204 204 204 255\
            \n  25.600\
            \nBONDM\
            \n 255 255 255 255\
            \n 128.000\
            \nPOLYM\
            \n 255 255 255 255\
            \n 128.000\
            \nSURFM\
            \n    0    0    0 255\
            \n 128.000\
            \nFORMM\
            \n 255 255 255 255\
            \n 128.000\
            \nHKLPM\
            \n 255 255 255 255\
            \n 128.000',file = f)
        
            f.close()
            if verbose>0:
                print('    written in %s'%(file_name))
            return 0
        
    elif select=='normal':
        if np.all(obj==None):
            print('no volume obj')
            return 0
        else:
            print('#VESTA_FORMAT_VERSION 3.5.0\n', file=f)
            for i1,obj1 in enumerate(obj):
                print('MOLECULE\
                \nTITLE',file=f)
                print('%s/%s_%d\n'%(path,basename,i1), file=f)
                print('GROUP\
                \n1 1 Custom\
                \nSYMOP\
                \n 0.000000  0.000000  0.000000  1  0  0    0  1  0    0  0  1    1\
                \n -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0\
                \nTRANM 0\
                \n 0.000000  0.000000  0.000000  1  0  0    0  1  0    0  0  1\
                \nLTRANSL\
                \n -1\
                \n 0.000000  0.000000  0.000000  0.000000  0.000000  0.000000\
                \nLORIENT\
                \n -1    0    0    0    0\
                \n 1.000000  0.000000  0.000000  1.000000  0.000000  0.000000\
                \n 0.000000  0.000000  1.000000  0.000000  0.000000  1.000000\
                \nLMATRIX\
                \n 1.000000  0.000000  0.000000  0.000000\
                \n 0.000000  1.000000  0.000000  0.000000\
                \n 0.000000  0.000000  1.000000  0.000000\
                \n 0.000000  0.000000  0.000000  1.000000\
                \n 0.000000  0.000000  0.000000\
                \nCELLP\
                \n  1.000000    1.000000    1.000000  90.000000  90.000000  90.000000\
                \n  0.000000    0.000000    0.000000    0.000000    0.000000    0.000000\
                \nSTRUC', file=f)
                for i2,vertx in enumerate(obj1):
                    xyz=projection3(vertx)
                    xyz=numerical_vector(xyz)
                    print('%4d Xx        Xx%d  1.0000    %8.6f %8.6f %8.6f        1'%\
                    (i2+1,i2+1,xyz[0],xyz[1],xyz[2]), file=f)
                    print('                             0.000000    0.000000    0.000000  0.00', file=f)
                print('  0 0 0 0 0 0 0\
                \nTHERI 0', file=f)
                for i2,_ in enumerate(obj1):
                    print('  %d        Xx%d  1.000000'%(i2+1,i2+1), file=f)
                print('  0 0 0\
                \nSHAPE\
                \n  0         0         0         0    0.000000  0    192    192    192    192\
                \nBOUND\
                \n         0          1        0          1        0          1\
                \n  0    0    0    0  0\
                \nSBOND', file=f)
                clr=colors(color)
                print('  1     Xx     Xx     0.00000     %3.2f  0  1  1  0  2  0.250  2.000 %3d %3d %3d'%(dmax,clr[0],clr[1],clr[2]), file=f)
                print('  0 0 0 0\
                \nSITET', file=f)
                for i2,_ in enumerate(obj1):
                    print('    %d        Xx%d  0.0100  76  76  76  76  76  76 204  0'%(i2+1,i2+1), file=f)
                print('  0 0 0 0 0 0\
                \nVECTR\
                \n 0 0 0 0 0\
                \nVECTT\
                \n 0 0 0 0 0\
                \nSPLAN\
                \n  0    0    0    0\
                \nLBLAT\
                \n -1\
                \nLBLSP\
                \n -1\
                \nDLATM\
                \n -1\
                \nDLBND\
                \n -1\
                \nDLPLY\
                \n -1\
                \nPLN2D\
                \n  0    0    0    0', file=f)
            print('ATOMT\
            \n  1        Xx  0.0100  76  76  76  76  76  76 204\
            \n  0 0 0 0 0 0\
            \nSCENE\
            \n 1.000000 -0.000000 -0.000000  0.000000\
            \n 0.000000  1.000000 -0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n  0.000    0.000\
            \n  0.000\
            \n  1.320\
            \nHBOND 0 2\
            \n\
            \nSTYLE\
            \nDISPF 37753794\
            \nMODEL    0  1  0\
            \nSURFS    0  1  1\
            \nSECTS  32  1\
            \nFORMS    0  1\
            \nATOMS    0  0  1\
            \nBONDS    2\
            \nPOLYS    1\
            \nVECTS 1.000000\
            \nFORMP\
            \n  1  1.0    0    0    0\
            \nATOMP\
            \n 24  24    0  50  2.0    0\
            \nBONDP\
            \n  1  16  0.250  2.000 127 127 127\
            \nPOLYP\
            \n 204 1  1.000 180 180 180\
            \nISURF\
            \n  0    0    0    0\
            \nTEX3P\
            \n  1  0.00000E+00  1.00000E+00\
            \nSECTP\
            \n  1  5.00000E-01  5.00000E-01  0.00000E+00  0.00000E+00  0.00000E+00  0.00000E+00\
            \nCONTR\
            \n 0.1 -1 1 1 10 -1 2 5\
            \n 2 1 2 1\
            \n    0    0    0\
            \n    0    0    0\
            \n    0    0    0\
            \n    0    0    0\
            \nHKLPP\
            \n 192 1  1.000 255    0 255\
            \nUCOLP\
            \n    0    1  1.000    0    0    0\
            \nCOMPS 0\
            \nLABEL 1     12  1.000 0\
            \nPROJT 0  0.962\
            \nBKGRC\
            \n 255 255 255\
            \nDPTHQ 1 -0.5000  3.5000\
            \nLIGHT0 1\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n  26  26  26 255\
            \n 179 179 179 255\
            \n 255 255 255 255\
            \nLIGHT1\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nLIGHT2\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nLIGHT3\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nATOMM\
            \n 204 204 204 255\
            \n  25.600\
            \nBONDM\
            \n 255 255 255 255\
            \n 128.000\
            \nPOLYM\
            \n 255 255 255 255\
            \n 128.000\
            \nSURFM\
            \n    0    0    0 255\
            \n 128.000\
            \nFORMM\
            \n 255 255 255 255\
            \n 128.000\
            \nHKLPM\
            \n 255 255 255 255\
            \n 128.000',file=f)
            f.close()
            if verbose>0:
                print('    written in %s'%(file_name))
            return 0
    
    elif select == 'podatm':
        if np.all(obj==None):
            print('no volume obj')
            return 0
        else:
            # get independent edges
            #edges = utils.generator_obj_edge(obj, verbose)
            edges = generator_unique_edges(obj)
            #print(len(edges))
            # get independent vertices of the edges
            vertices = remove_doubling_in_perp_space(edges)
            #print(len(vertices))
            # get bond pairs, [[distance, XXX, YYY],...]
            pairs = []
            for edge in edges:
                dist=distance_in_perp_space(edge[0],edge[1])
                a=[dist]
                for i2 in range(2):
                    i3=0
                    for vrtx in vertices:
                        tmp=np.vstack([edge[i2],vrtx])
                        tmp=remove_doubling_in_perp_space(tmp.reshape(2,6,3))
                        #i3+=1
                        if len(tmp)==1:
                            a.append(i3)
                            break
                        else:
                            i3+=1
                            pass
                pairs.append(a)
            #print(len(pairs))
            
            print('#VESTA_FORMAT_VERSION 3.5.0\n', file=f)
            print('MOLECULE\
            \nTITLE',file=f)
            print('%s/%s\n'%(path,basename), file=f)
            print('GROUP\
            \n1 1 Custom\
            \nSYMOP\
            \n 0.000000  0.000000  0.000000  1  0  0    0  1  0    0  0  1    1\
            \n -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0\
            \nTRANM 0\
            \n 0.000000  0.000000  0.000000  1  0  0    0  1  0    0  0  1\
            \nLTRANSL\
            \n -1\
            \n 0.000000  0.000000  0.000000  0.000000  0.000000  0.000000\
            \nLORIENT\
            \n -1    0    0    0    0\
            \n 1.000000  0.000000  0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000  0.000000  1.000000\
            \nLMATRIX\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000  0.000000\
            \nCELLP\
            \n  1.000000    1.000000    1.000000  90.000000  90.000000  90.000000\
            \n  0.000000    0.000000    0.000000    0.000000    0.000000    0.000000\
            \nSTRUC', file=f)
            i2=0
            for vrtx in vertices:
                xyz = projection3(vrtx)
                print('%4d A        A%d  1.0000    %8.6f %8.6f %8.6f        1'%\
                (i2+1,i2+1,numeric_value(xyz[0]),numeric_value(xyz[1]),numeric_value(xyz[2])), file=f)
                i2+=1
                print('                             0.000000    0.000000    0.000000  0.00', file=f)
            print('  0 0 0 0 0 0 0\
            \nTHERI 0', file = f)
            for i2 in range(len(vertices)):
                print('  %d        A%d  1.000000'%(i2+1,i2+1), file=f)
            print('  0 0 0\
            \nSHAPE\
            \n  0         0         0         0    0.000000  0    192    192    192    192\
            \nBOUND\
            \n         0          1        0          1        0          1\
            \n  0    0    0    0  0\
            \nSBOND', file = f)
            clr=colors(color)
            #print(pairs)
            #print(len(pairs))
            for pair in pairs:
                #print(pair)
                print('  %d   A%d   A%d   %6.3f   %6.3f  0  1  1  1  2  0.250  2.000 %3d %3d %3d'%(\
                i2+1, pair[1]+1, pair[2]+1, pair[0]-0.01, pair[0]+0.01, clr[0], clr[1], clr[2]), file=f)
                i2+=1
            print('  0 0 0 0\
            \nSITET', file = f)
            for i2 in range(len(vertices)):
                print('    %d        A%d  0.030  76  76  76  76  76  76 204  0'%(i2+1,i2+1), file=f)
            print('  0 0 0 0 0 0\
            \nVECTR\
            \n 0 0 0 0 0\
            \nVECTT\
            \n 0 0 0 0 0\
            \nSPLAN\
            \n  0    0    0    0\
            \nLBLAT\
            \n -1\
            \nLBLSP\
            \n -1\
            \nDLATM\
            \n -1\
            \nDLBND\
            \n -1\
            \nDLPLY\
            \n -1\
            \nPLN2D\
            \n  0    0    0    0', file = f)
        
            print('ATOMT\
            \n  1        A  0.0100  76  76  76  76  76  76 204\
            \n  0 0 0 0 0 0\
            \nSCENE\
            \n-0.538344 -0.838391  0.085359  0.000000\
            \n-0.362057  0.138632 -0.921789  0.000000\
            \n 0.760986 -0.527145 -0.378177  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n  0.000    0.000\
            \n  0.000\
            \n  1.320\
            \nHBOND 0 2\
            \n\
            \nSTYLE\
            \nDISPF 37753794\
            \nMODEL    0  1  0\
            \nSURFS    0  1  1\
            \nSECTS  32  1\
            \nFORMS    0  1\
            \nATOMS    0  0  1\
            \nBONDS    2\
            \nPOLYS    1\
            \nVECTS 1.000000\
            \nFORMP\
            \n  1  1.0    0    0    0\
            \nATOMP\
            \n 24  24    0  50  2.0    0\
            \nBONDP\
            \n  1  16  0.250  2.000 127 127 127\
            \nPOLYP\
            \n 204 1  1.000 180 180 180\
            \nISURF\
            \n  0    0    0    0\
            \nTEX3P\
            \n  1  0.00000E+00  1.00000E+00\
            \nSECTP\
            \n  1  5.00000E-01  5.00000E-01  0.00000E+00  0.00000E+00  0.00000E+00  0.00000E+00\
            \nCONTR\
            \n 0.1 -1 1 1 10 -1 2 5\
            \n 2 1 2 1\
            \n    0    0    0\
            \n    0    0    0\
            \n    0    0    0\
            \n    0    0    0\
            \nHKLPP\
            \n 192 1  1.000 255    0 255\
            \nUCOLP\
            \n    0    1  1.000    0    0    0\
            \nCOMPS 0\
            \nLABEL 1     12  1.000 0\
            \nPROJT 0  0.962\
            \nBKGRC\
            \n 255 255 255\
            \nDPTHQ 1 -0.5000  3.5000\
            \nLIGHT0 1\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n  26  26  26 255\
            \n 179 179 179 255\
            \n 255 255 255 255\
            \nLIGHT1\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nLIGHT2\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nLIGHT3\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nATOMM\
            \n 204 204 204 255\
            \n  25.600\
            \nBONDM\
            \n 255 255 255 255\
            \n 128.000\
            \nPOLYM\
            \n 255 255 255 255\
            \n 128.000\
            \nSURFM\
            \n    0    0    0 255\
            \n 128.000\
            \nFORMM\
            \n 255 255 255 255\
            \n 128.000\
            \nHKLPM\
            \n 255 255 255 255\
            \n 128.000',file = f)
        
            f.close()
            #write_vesta_separate(obj, path, basename, color, dmax)
            if verbose>0:
                print('    written in %s'%(file_name))
            return vertices
    
    else:
        return 1

def write_xyz(obj:NDArray[DTYPE_int],path='.',basename='tmp',select='triangle',verbose=0):
    """
    Export occupation domains in XYZ format.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        path (str): Path of the output XYZ file
        basename (str): Basename of the output XYZ file
        select (str)
            'triangle'   : set of triangles (default)
            'edge'       : set of edges
            'vertex'      : set of vertices
            (default, select = 'triangle')
    
    Returns:
        int: 0 (succeed), 1 (fail)
    """
    
    def generator_xyz_dim4_triangle(obj,path,filename):
        """
        Generate object (set of triangles) object in XYZ format.
    
        Args:
            obj (numpy.ndarray): the occupation domain
                The shape is (num,3,6,3), where num=numbre_of_triangle.
            filename (str): filename of the output XYZ file
        
        Returns:
            int: 0 (succeed), 1 (fail)
        
        """
        f=open('%s/%s.xyz'%(path,filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)*3))
        f.write('%s\n'%(filename))
        i1=0
        ln=len(obj)
        #for i1,triangle in enumerate(obj):
        for i1 in range(ln):
            tri=obj[i1]
            #for i2,vt in enumerate(triangle):
            for i2 in range(3):
                vt=tri[i2]
                v=projection3(vt)
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the triangle %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                (numeric_value(v[0]),\
                numeric_value(v[1]),\
                numeric_value(v[2]),\
                i1,i2,\
                vt[0][0],vt[0][1],vt[0][2],\
                vt[1][0],vt[1][1],vt[1][2],\
                vt[2][0],vt[2][1],vt[2][2],\
                vt[3][0],vt[3][1],vt[3][2],\
                vt[4][0],vt[4][1],vt[4][2],\
                vt[5][0],vt[5][1],vt[5][2]))
        v=obj_area_6d(obj)
        f.write('volume = %d %d %d (%8.6f)\n'%(v[0],v[1],v[2],numeric_value(v)))
        #for i1,triangle in enumerate(obj):
        for i1 in range(ln):
            tri=obj[i1]
            v=triangle_area_6d(tri)
            f.write('%3d-the triangle, %d %d %d (%8.6f)\n'\
                    %(i1,v[0],v[1],v[2],numeric_value(v)))
        f.closed
        return 0
    
    def generator_xyz_dim4_edge(obj:NDArray[DTYPE_int],path,filename):
        """
        Generate object (set of edges) object in XYZ format.
    
        Args:
            obj (numpy.ndarray): the occupation domain
                The shape is (num,2,6,3), where num=numbre_of_e.
            filename (str): filename of the output XYZ file
        
        Returns:
            int: 0 (succeed), 1 (fail)
        
        """
        f=open('%s/%s.xyz'%(path,filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)*2))
        f.write('%s\n'%(filename))
        ln=len(obj)
        #for i1,edge in enumerate(obj):
        for i1 in range(ln):
            edge=obj[i1]
            #for i2,vt in enumerate(edge):
            for i2 in range(2):
                vt=edge[i2]
                v=projection3(vt)
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the edge %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                (numeric_value(v[0]),\
                numeric_value(v[1]),\
                numeric_value(v[2]),\
                i1,i2,\
                vt[0][0],vt[0][1],vt[0][2],\
                vt[1][0],vt[1][1],vt[1][2],\
                vt[2][0],vt[2][1],vt[2][2],\
                vt[3][0],vt[3][1],vt[3][2],\
                vt[4][0],vt[4][1],vt[4][2],\
                vt[5][0],vt[5][1],vt[5][2]))
        f.closed
        return 0
    
    def generator_xyz_dim4_vertex(obj:NDArray[DTYPE_int],path,filename):
        """
        Generate object (set of vertexs) object in XYZ format.
    
        Args:
            obj (numpy.ndarray): the occupation domain
                The shape is (num,3,6,3), where num=numbre_of_tetrahedron.
            filename (str): filename of the output XYZ file
        
        Returns:
            int: 0 (succeed), 1 (fail)
        
        """
        f=open('%s/%s.xyz'%(path,filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)))
        f.write('%s\n'%(filename))
        counter=0
        ln=len(obj)
        #for triangle in obj:  #range(len(obj)):
        for i in range(ln):  #range(len(obj)):
            tri=obj[i]
            #for point in triangle: # range(len(triangle)):
            for j in range(3): # range(len(triangle)):
                point=tri[j]
                v=projection3(point)
                f.write('Xx %8.6f %8.6f %8.6f # %d-th vertex # # # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                (numeric_value(v[0]),\
                numeric_value(v[1]),\
                numeric_value(v[2]),\
                counter,\
                point[0][0],point[0][1],point[0][2],\
                point[1][0],point[1][1],point[1][2],\
                point[2][0],point[2][1],point[2][2],\
                point[3][0],point[3][1],point[3][2],\
                point[4][0],point[4][1],point[4][2],\
                point[5][0],point[5][1],point[5][2]))
                counter+=1
        f.closed
        return 0
    
    def generator_xyz_dim3_vertex(obj:NDArray[DTYPE_int],path,filename):
        """
        Generate object (set of vertexs) object in XYZ format.
    
        Args:
            obj (numpy.ndarray): the occupation domain
                The shape is (num,6,3), where num=numbre_of_vertices.
            filename (str): filename of the output XYZ file
        
        Returns:
            int: 0 (succeed), 1 (fail)
        
        """
        f=open('%s/%s.xyz'%(path,filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)))
        f.write('%s\n'%(filename))
        ln=len(obj)
        #for i1,point in enumerate(obj):
        for i1 in range(ln):
            point=obj[i1]
            v=projection3(point)
            f.write('Xx %8.6f %8.6f %8.6f # %d-th vertex # # # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
            (numeric_value(v[0]),\
            numeric_value(v[1]),\
            numeric_value(v[2]),\
            i1,\
            point[0][0],point[0][1],point[0][2],\
            point[1][0],point[1][1],point[1][2],\
            point[2][0],point[2][1],point[2][2],\
            point[3][0],point[3][1],point[3][2],\
            point[4][0],point[4][1],point[4][2],\
            point[5][0],point[5][1],point[5][2]))
        f.closed
        return 0
        
    if np.all(obj==None):
        print('empty obj')
        return 
    elif obj.ndim<3 or obj.ndim>4:
        print('object has an incorrect shape!')
        return 
    elif obj.ndim==3:
        if select=='vertex':
            generator_xyz_dim3_vertex(obj,path,basename)
            if verbose>0:
                print('    written in %s/%s.xyz'%(path,basename))
            return 0
        else:
            return 
    else:
        file_name='%s/%s.xyz'%(path,basename)
        if select=='triangle':
            generator_xyz_dim4_triangle(obj,path,basename)
            if verbose>0:
                print('    written in %s/%s.xyz'%(path,basename))
            return 0
        elif select=='edge':
            generator_xyz_dim4_edge(obj,path,basename)
            if verbose>0:
                print('    written in %s/%s.xyz'%(path,basename))
            return 0
        elif select=='vertex':
            generator_xyz_dim4_vertex(obj,path,basename)
            if verbose>0:
                print('    written in %s/%s.xyz'%(path,basename))
            return 0
        else:
            if verbose>0:
                print('    error')
            return 

def read_xyz(path,basename,select='triangle',verbose=0):
    """
    Load new occupation domain on input XYZ file.
    
    Args:
        path (str): Path of the input XYZ file
        basename (str): Basename of the input XYZ file
        select (str)
            'triangle'    : read as a set of triangles (default)
            'vertex'      : read as a set of vertices
            (default, select = 'triangle')
        verbose (int): verbose option
    Returns:
        Occupation domains (numpy.ndarray):
            Loaded occupation domains.
            The shape is (num,3,6,3), where num=numbre_of_triangles (select = 'triangle').
            The shape is (num,6,3), where num=numbre_of_vertices (select = 'vertex').
    """
    
    def read_file(file):
        try:
            f=open(file,'r')
        except IOError as e:
            print(e)
            sys.exit(0)
        line=[]
        while 1:
            a=f.readline()
            if not a:
                break
            line.append(a[:-1])
        return line
    
    filename='%s/%s.xyz'%(path,basename)
    
    f1=read_file(filename)
    f0=f1[0].split()
    num=int(f0[0])
    
    for i in range(2,num+2):
        fi=f1[i]
        fi=fi.split()
        a1=int(fi[10])
        b1=int(fi[11])
        c1=int(fi[12])
        a2=int(fi[13])
        b2=int(fi[14])
        c2=int(fi[15])
        a3=int(fi[16])
        b3=int(fi[17])
        c3=int(fi[18])
        a4=int(fi[19])
        b4=int(fi[20])
        c4=int(fi[21])
        a5=int(fi[22])
        b5=int(fi[23])
        c5=int(fi[24])
        a6=int(fi[25])
        b6=int(fi[26])
        c6=int(fi[27])
        if i==2:
            tmp=np.array([a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6])
        else:
            tmp=np.append(tmp,[a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6])
    if verbose>0:
        print('    read %s/%s.xyz'%(path,basename))
    
    if select == 'triangle':
        return tmp.reshape(int(num/3),3,6,3)
    elif select == 'vertex':
        return tmp.reshape(int(num),6,3)
    
def generator_obj_edge(obj: NDArray[DTYPE_int], verbose: int):
    # remove doubling edges in a OD
    # parameter: object (dim4)
    # return: independent edges OD (dim4)
    i1: int 
    i2: int
    num1: int
    counter1: int
    combi: list
    tmp1a: np.ndarray[DTYPE_int_t]
    tmp4a: np.ndarray[DTYPE_int_t]
    
    if verbose>0:
        print('      generator_obj_edge()')
    else:
        pass
    
    if verbose>0:
        print('       Number of tetrahedra: %d'%(len(obj)))
    else:
        pass
    
    combi=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
    
    # six edges of 1st tetrahedron
    tmp1a=np.append(obj[0][combi[0][0]],obj[0][combi[0][1]])
    for i1 in range(1,len(combi)):
        for i2 in range(2):
            tmp1a=np.append(tmp1a,obj[0][combi[i1][i2]])
    
    if len(obj)>0:
        for i1 in range(1,len(obj)):
            for i2 in range(6):
                tmp1a=np.append(tmp1a,obj[i1][combi[i2][0]])
                tmp1a=np.append(tmp1a,obj[i1][combi[i2][1]])
        num1=int(len(tmp1a)/36) # 2*6*3=36
        tmp4a=np.array(tmp1a).reshape(num1,2,6,3)
        if verbose>0:
            print('       Number of edges: %d'%(num1))
        else:
            pass
        tmp4b=tmp4a[0].reshape(1,2,6,3)
        for i1 in range(1,num1):
            counter1=0
            for i2 in range(len(tmp4b)):
                if equivalent_edge(tmp4a[i1],tmp4b[i2])==0: # equivalent
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                tmp4b=np.vstack([tmp4b,[tmp4a[i1]]])
            else:
                pass
        if verbose>0:
            print('       Number of unique edges: %d'%(len(tmp4b)))
        else:
            pass
        return tmp4b
    else:
        return tmp1a.reshape(6,2,6,3)

