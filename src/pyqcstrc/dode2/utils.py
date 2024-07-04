#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
#sys.path.append('.')
from pyqcstrc.dode2.math1 import (projection3,
                                add,
                                sub,
                                mul,
                                div,
                                add_vectors,
                                sub_vectors,
                                outer_product,
                                inner_product,
                                centroid,
                                coplanar_check,
                                )
from pyqcstrc.dode2.numericalc import (numeric_value,
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

def shift_object(obj: NDArray[np.int64], shift: NDArray[np.int64]) -> NDArray[np.int64]:
    """shift an object
    """
    if obj.ndim==4:
        obj_new=np.zeros(obj.shape,dtype=np.int64)
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
def obj_area_6d(obj: NDArray[np.int64]) -> NDArray[np.int64]:
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

def triangle_area_6d(triangle: NDArray[np.int64]) -> NDArray[np.int64]:
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
        vts=np.zeros((3,3,3),dtype=np.int64)
        for i in range(4):
            vts[i]=projection3(triangle[i])
        return triangle_volume(vts)
    else:
        print('object has an incorrect shape!')
        return 

#######################
###  To be checked  ###
#######################
def triangle_area(vts: NDArray[np.int64]) -> NDArray[np.int64]:
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
    
    # avoid a negative value
    val=numeric_value(v)
    if val<0.0: # to avoid negative volume
        return mul(v,np.array([-1,0,2]))
    else:
        return mul(v,np.array([1,0,2]))

#----------------------------
# Remove doubling
#----------------------------
def remove_doubling(vts: NDArray[np.int64]) -> NDArray[np.int64]:
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

def remove_doubling_in_perp_space(vts: NDArray[np.int64]) -> NDArray[np.int64]:
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
    a=np.zeros((num,3,3),dtype=np.int64)
    for i in range(num):
        a[i]=projection3(vts[i])
    b=np.unique(a,return_index=True,axis=0)[1]
    num=len(b)
    a=np.zeros((num,6,3),dtype=np.int64)
    for i in range(num):
        a[i]=vts[b[i]]
    return a

#----------------------------
# Edges
#
# Comment：Need to be reorganised.
#----------------------------

#### WIP ###
def get_common_edges(trianges: NDArray[np.int64]) -> NDArray[np.int64]:
    """Get common edges in trianges
    """
    return 

def generator_all_edges(obj: NDArray[np.int64]) -> NDArray[np.int64]:
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
        edges=np.zeros((n1,3,2,6,3),dtype=np.int64)
        i1=0
        for triangle in obj:
            edges[i1]=get_triangle_edge(triangle)
            i1+=1
        return edges.reshape(n1*3,2,6,3)
    else:
        print('obj should be a set of trianges')
        return 

### WIP: to be checked ###
def generator_unique_edges(obj: NDArray[np.int64]) -> NDArray[np.int64]:
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
    a=np.zeros((num,2,6,3),dtype=np.int64)
    for i1 in range(num):
        a[i1]=edges[b[i1]]
    return a

def get_triangle_edge(triangle: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return three edges of triange.
    """
    # three edges of triange: 0-1, 0-2, 1-2
    comb=[\
    [0,1],\
    [0,2],\
    [1,2]] 
    
    # Three egdes of the triangl.
    a=np.zeros((3,2,6,3),dtype=np.int64)
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
def generate_convex_hull(obj: NDArray[np.int64]) -> NDArray[np.int64]:
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

def surface_cleaner(surface: NDArray[np.int64]) -> NDArray[np.int64]:
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
    out=np.zeros((n1,2,6,3),dtype=np.int64)
    for i1 in range(n1):
        out[i1]=edges_new[lst[i1]]
    #print('out.shape',out.shape)
    
    return out

def gen_border_edges_of_coplanar_triangles(coplanar_triangles: NDArray[np.int64]) -> NDArray[np.int64]:
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
    return np.array(lst,dtype=np.int64)

#----------------------------
# Equivalence check
#
# Better to merge following four functions into a function "equivalent"???
#   equivalent_triangles
#   equivalent_edges
#   equivalent_vertices
#----------------------------
# WIP:
def equivalent(obj1: NDArray[np.int64], obj2: NDArray[np.int64]) -> bool:
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

def equivalent_triangles(triangle1: NDArray[np.int64], triangle2: NDArray[np.int64]) -> bool:
    """Checking whether triangle1 and triangle2 are equivalent or not.
    """
    a=np.vstack([triangle1,triangle2])
    a=remove_doubling_in_perp_space(a)
    if len(a)==3:
        return True # equivalent traiangle
    else:
        return False # not equivalent traiangles

def equivalent_edges(edge1: NDArray[np.int64], edge2: NDArray[np.int64]) -> bool:
    """Checking whether edge1 and edge2 are equivalent or not.
    """
    a=np.vstack([edge1,edge2])
    a=remove_doubling_in_perp_space(a)
    if len(a)==2:
        return True # equivalent
    else:
        return False # not equivalent

def equivalent_vertices(vertex1: NDArray[np.int64], vertex2: NDArray[np.int64]) -> bool:
    xyz1=projection3(vertex1)
    xyz2=projection3(vertex2)
    if np.all(xyz1==xyz2):
        return True # equivalent
    else:
        return False

#----------------------------
# Sort
#----------------------------
def sort_vctors(vts: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    sort vectors in TAU-style
    
    xi,yi,ziをxiでソート
    
    """
    n1,n2,_=vts.shape
    out=np.zeros(vts.shape,dtype=np.int64)
    vns=get_internal_component_sets_numerical(vts)
    
    #tmp=np.argsort(vns,axis=0)
    tmp=vns[np.argsort(vns[:,0])]
    for i1 in range(n1):
        out[i1]=vts[tmp[i1][0]]
    return out

def sort_obj(obj: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    sort triangle in an object
    
    """
    out=np.zeros(vts.shape,dtype=np.int64)
    centroids=np.zeros(len(obj),dtype=np.float64)
    tmp=np.zeros((obj.shape,3),dtype=np.int64)
    
    # 各triangleの頂点xyzをx順にソートすると同時に重心を求めておく。
    for i1 in range(len(obj)):
        tmp[i1]=sort_vctors(obj[i1])
        centroids[i1]=centroid(obj[i1])
    
    # 三角形の重心xyzのx順にソート
    #indx=np.argsort(centroids,axis=0)
    indx=centroids[np.argsort(centroids[:,0])]
    
    for i1 in range(n1):
        out[i1]=tmp[indx[i1][0]]
    return out

#----------------------------
# Triangulation
#----------------------------
def decomposition(tmp2v: NDArray[np.int64]) -> NDArray[np.int64]:
    try:
        tri=Delaunay(tmp2v)
    except:
        print('error in decomposition')
        return 
    else:
        tmp=[]
        for i in range(len(tri.simplices)):
            tet=tri.simplices[i]
            tmp.append([tet[0],tet[1],tet[2],tet[3]])
    return tmp

def triangulation_points(points: NDArray[np.int64]) -> NDArray[np.int64]:
    
    i1=0
    for p in points:
        v=projection3(p)
        v=numerical_vector(v)
        if i1==0:
            tmp=v
        else:
            tmp=np.vstack([tmp,v])
        i1+=1
        
    ltmp=decomposition(tmp)
    p=points
    if ltmp!=[0]:
        counter=0
        for i in ltmp:
            tmp3=np.array([p[i[0]],p[i[1]],p[i[2]],p[i[3]]]).reshape(4,6,3)
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
            return tmp1.reshape(int(len(tmp1)/54),3,6,3) # 3*6*3=54
        else:
            return 
    else:
        return 

##############################
####
####
#### WIP: Removing
####
####
##############################
def remove_vectors(vts1: NDArray[np.int64], vts2: NDArray[np.int64]) -> NDArray[np.int64]:
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
        out=np.zeros((len(lst),6,3),dtype=np.int64)
        for i1 in range(len(lst)):
            out[i1]=vts1[lst[i1]]
        return out
    else:
        return vts1

def remove_vector(vts: NDArray[np.int64], vt: NDArray[np.int64]) -> NDArray[np.int64]:
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
        out=np.zeros((len(lst),6,3),dtype=np.int64)
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
def merge_two_triangles_in_obj(obj: NDArray[np.int64]) -> NDArray[np.int64]:
    
    num=len(obj)
    
    
    return obj

def merge_two_triangles(triangle_1: NDArray[np.int64], triangle_2: NDArray[np.int64]) -> NDArray[np.int64]:
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
    
def check_connectivity_triangles(triangle_1: NDArray[np.int64], triangle_2: NDArray[np.int64]) -> bool:
    """Checking whether triangle_1 and _2 are sharing an edge or not.
    """
    a=np.vstack([triangle_1,triangle_1])
    a=remove_doubling_in_perp_space(a)
    if len(a)==4:
        return True # common edge
    else:
        return False # not commom edge

def get_common_edge_in_two_tetrahedra(triangle_1: NDArray[np.int64], triangle_2: NDArray[np.int64]) -> NDArray[np.int64]:
    """ Return common edge of two connected triangles.
    """
    edge1=get_triangle_edge(triangle_1)
    edge2=get_triangle_edge(triangle_2)
    
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

def two_segment_into_one(line_segment_1: NDArray[np.int64], line_segment_2: NDArray[np.int64]) -> NDArray[np.int64]:
    
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

def coplanar_check_two_triangles(triange1: NDArray[np.int64], triange2: NDArray[np.int64]) -> bool:
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
    for i1 in range(6):
        v=add(pos1[i1],pos2[i1])
        v=mul(v,np.array([1,0,2]))
        if i1!=0:
            out=np.vstack([tmp2,v])
        else:
            out=v.reshape(1,3)
    return out

if __name__ == '__main__':
    
    # test
    
    import random
    
    def generate_random_value():
        """ generate value in TAU-style
        """
        nmax=10
        v=np.zeros((3),dtype=np.int64)
        for i1 in range(2):
            v[i1]=random.randrange(-nmax,nmax) # a and b in (a+b*TAU)/c.
        v[2]=random.randrange(1,nmax) # c in (a+b*TAU)/c.
        return v
        
    def generate_random_vector(ndim=6):
        """ generate ndim vector in TAU-style
        ndim: dimension of vectors
        """
        nmax=10
        v=np.zeros((ndim,3), dtype=np.int64)
        for i1 in range(ndim):
            v[i1]=generate_random_value()
        return v
        
    def generate_random_vectors(n,ndim=6):
        """
        num: number of generated vectors.
        ndim: dimension of vectors
        """
        v=np.zeros((n,ndim,3), dtype=np.int64)
        for i1 in range(n):
            v[i1]=generate_random_vector(ndim)
        return v
    
    def generate_random_tetrahedron():
        return generate_random_vectors(4)
    
    
    
    #================
    # ソートのテスト
    #================
    nset=10
    vts=generate_random_vectors(nset)
    vns=get_internal_component_sets_numerical(vts)
    for vn in vns:
        print(vn)
    print('\n')
    vts1=sort_vctors(vts)
    vns1=get_internal_component_sets_numerical(vts1)
    for vn in vns1:
        print(vn)
    
    #================
    # 重複のテスト
    #================
    nset=10
    vst=generate_random_vectors(nset)
    vst_d3=np.concatenate([vst,vst]) # doubling dim3 vectors
    vst_d4=np.stack([vst_d3,vst_d3]) # doubling dim4 vectors
    
    a=remove_doubling(vst_d4)
    if len(a)==nset:
        print('remove_doubling: pass')
    else:
        print('remove_doubling: error')
        
    a=remove_doubling_in_perp_space(vst_d4)
    if len(a)==nset:
        print('remove_doubling_in_perp_space: pass')
    else:
        print('remove_doubling_in_perp_space: error')
    
    #================
    # 面と辺のテスト
    #================
    tetrahedron=generate_random_tetrahedron()
    
    # doubled tetrahedon
    obj=np.stack([tetrahedron,tetrahedron]) # doubled tetrahedon
    generator_surface_1(obj)
    
    # a tetrahedon
    obj=tetrahedron
    surface=generator_surface_1(obj.reshape(1,4,6,3))
    generator_edge(surface)
    
    
    