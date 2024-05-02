#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
sys.path.append('.')
from math1 import (projection3,
                    sub_vectors,
                    add_vectors,
                    outer_product,
                    inner_product,
                    centroid,
                    coplanar_check,
                    add,
                    sub,
                    mul,
                    div)
from numericalc import (numeric_value,
                        numerical_vector,
                        numerical_vectors,
                        get_internal_component_numerical,
                        get_internal_component_sets_numerical,
                        point_on_segment,
                        coplanar_check_numeric_tau)
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay

def shift_object(obj: NDArray[np.int64], shift: NDArray[np.int64]) -> NDArray[np.int64]:
    """shift an object
    """
    if obj.ndim==4:
        obj_new=np.zeros(obj.shape,dtype=np.int64)
        i1=0
        for tetrahedron in obj:
            i2=0
            for vertex in tetrahedron:
                obj_new[i1][i2]=add_vectors(vertex,shift)
                i2+=1
            i1+=1
        return obj_new
    else:
        print('object has an incorrect shape!')
        return 

##############################
# 体積・面積関連
##############################
def obj_volume_6d(obj: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    """
    w=np.array([0,0,1])
    if obj.ndim==4:
        for tetrahedron in obj:
            v=tetrahedron_volume_6d(tetrahedron)
            w=add(w,v)
        return w
    elif obj.ndim==5:
        for tset in obj:
            for tetrahedron in tset:
                v=tetrahedron_volume_6d(tetrahedron)
                w=add(w,v)
        return w
    else:
        print('object has an incorrect shape!')
        return 
    
def tetrahedron_volume_6d(tetrahedron: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    """
    if tetrahedron.ndim==3:
        vts=np.zeros((4,3,3),dtype=np.int64)
        for i in range(4):
            vts[i]=projection3(tetrahedron[i])
        return tetrahedron_volume(vts)
    else:
        print('object has an incorrect shape!')
        return 

def tetrahedron_volume(vts: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    # This function returns volume of a tetrahedron
    # input: vertex coordinates of the tetrahedron (x0,y0,z0),(x1,y1,z1),(x2,y2,z2),(x3,y3,z3) in TAU-style.
    """
    v1=sub_vectors(vts[1],vts[0])
    v2=sub_vectors(vts[2],vts[0])
    v3=sub_vectors(vts[3],vts[0])
    
    v=outer_product(v1,v2)
    v=inner_product(v,v3)
    
    # avoid a negative value
    val=numeric_value(v)
    if val<0.0: # to avoid negative volume
        return mul(v,np.array([-1,0,6]))
    else:
        return mul(v,np.array([1,0,6]))



##############################
# 重複関連
##############################
def remove_doubling(vts: NDArray[np.int64]) -> NDArray[np.int64]:
    """remove doubling 6d coordinates
    
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
    """ remove 6d coordinates which is doubled in Eperp.
    
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





##############################
# objの表面関連
##############################
def generator_surface_1(obj: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    # remove doubling surface in a set of tetrahedra in the OD (dim4)
    #
    """
    # (1) preparing a list of triangle surfaces without doubling (tmp2)
    n1,_,_,_=obj.shape
    triangles=np.zeros((n1,4,3,6,3),dtype=np.int64)
    i1=0
    for tetrahedron in obj:
        triangles[i1]=get_tetrahedron_surface(tetrahedron)
        i1+=1
    triangles=triangles.reshape(n1*4,3,6,3)
    if n1==1:
        return triangles
    else:
        # (2) 重複のない三角形（すなはちobject表面の三角形）のみを得る。
        # 三角形が重複していれば重心も同じことを利用する。重心が一致すれば重複しているとは限らないが、
        # objが正しく与えられているとすれば問題ない。
        #
        # まず重心xyzを求める
        xyz=np.zeros((n1*4,3),dtype=np.float64)
        for i1 in range(n1*4):
            vt=centroid(triangles[i1])
            xyz[i1]=get_internal_component_numerical(vt)
        #
        #
        # 以下のやり方では効率悪く、triangleの数が多ければ時間がかかる。改善が必要。
        # xyzをxでソートし、indexを得る。
        indx_xyz=np.argsort(xyz[:,0])
        #
        # 重複しているtriangleはスキップ。表面のtriangleのみを選ぶ。
        #print('number of trianges:',len(indx_xyz))
        lst=[]
        #print('indx_xyz:',indx_xyz)
        for i1 in indx_xyz:
            counter=0
            for i2 in indx_xyz:
                if i1==i2:
                    pass
                else:
                    if np.allclose(xyz[i1],xyz[i2]): # equivalent
                        counter+=1
                        break
            if counter==0:
                lst.append(i1)
        #print('lst:',lst)
        out=np.zeros((len(lst),3,6,3),dtype=np.int64)
        #print('number of unique triangls:',len(lst))
        for i1 in range(len(lst)):
            out[i1]=triangles[lst[i1]]
        #print('shape:',out.shape)
        return out

def generator_unique_triangles(obj: NDArray[np.int64]) -> NDArray[np.int64]:
    """get unique triangles in an object (dim4)
    
    Input:
    obj: set of tetrahedra
    
    """
    
    # (1) preparing a list of triangle surfaces without doubling (tmp2)
    #print('get_tetrahedron_surface() starts')
    n1,_,_,_=obj.shape
    triangles=np.zeros((n1,4,3,6,3),dtype=np.int64)
    i1=0
    for tetrahedron in obj:
        triangles[i1]=get_tetrahedron_surface(tetrahedron)
        i1+=1
    triangles=triangles.reshape(n1*4,3,6,3)
    #print('get_tetrahedron_surface() ends')
    #print('      number of triangle',len(triangles))
    if n1==1:
        return triangles
    else:
        # (2) ユニークな三角形を得る。
        # 三角形が重複していれば重心も同じことを利用する。重心が一致すれば重複しているとは限らないが、
        # objが正しく与えられているとすれば問題ない。
        #print('number of trianges:',len(triangles))
        a=np.zeros((n1*4,3),dtype=np.float64)
        for i1 in range(n1*4):
            vt=centroid(triangles[i1])
            a[i1]=get_internal_component_numerical(vt)
        b=np.unique(a,return_index=True,axis=0)[1]
        num=len(b)
        #print('number of unique trianges:',num)
        a=np.zeros((num,3,6,3),dtype=np.int64)
        for i1 in range(num):
            a[i1]=triangles[b[i1]]
        return a

def get_common_edges(trianges: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    common edges in trianges
    """
    return 

def generator_all_edges(obj: NDArray[np.int64]) -> NDArray[np.int64]:
    # (1) preparing a list of edges
    n1,n2,_,_=obj.shape
    edges=np.zeros((n1,n2,2,6,3),dtype=np.int64)
    i1=0
    for triangle in obj:
        edges[i1]=get_triangle_edge(triangle)
        i1+=1
    return edges.reshape(n1*n2,2,6,3)
    
def generator_unique_edges(obj: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    generates edges
    
    Input
    triangles, np.array with a shape=(number_of_triangles,3,6,3)
    """
    
    # (1) preparing a list of edges
    edges=generator_all_edges(obj)
    
    # (2) 重複のないユニークな辺を得る。
    #print('number of edges:',len(edges))
    n1,n2,_,_=obj.shape
    a=np.zeros((n1*n2,3),dtype=np.float64)
    for i1 in range(len(a)):
        vt=centroid(edges[i1])
        a[i1]=get_internal_component_numerical(vt)
    b=np.unique(a,return_index=True,axis=0)[1]
    num=len(b)
    #print('number of unique edges:',num)
    a=np.zeros((num,2,6,3),dtype=np.int64)
    for i1 in range(num):
        a[i1]=edges[b[i1]]
    return a

def get_tetrahedron_surface(tetrahedron: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    get four triangles of tetrahedron.
    """
    # four triangles of tetrahedron: 1-2-3, 1-2-4, 1-3-4, 2-3-4
    comb=[\
    [0,1,2],\
    [0,1,3],\
    [0,2,3],\
    [1,2,3]] 
    #
    a=np.zeros((4,3,6,3),dtype=np.int64)
    i1=0
    for k in comb:
        i2=0
        for l in k:
            a[i1][i2]=tetrahedron[l]
            i2+=1
        i1+=1
    return a

def get_triangle_edge(triangle: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    get four triangles of tetrahedron.
    """
    # three edges of triange: 0-1, 0-2, 1-2
    comb=[\
    [0,1],\
    [0,2],\
    [1,2]] 
    #
    # Four triangles the tetrahedron.
    a=np.zeros((3,2,6,3),dtype=np.int64)
    i1=0
    for k in comb:
        i2=0
        for l in k:
            a[i1][i2]=triangle[l]
            i2+=1
        i1+=1
    return a



##############################
# objの等価チェック
##############################
def equivalent_tetrahedra(tetrahedron_1: NDArray[np.int64], tetrahedron_2: NDArray[np.int64]) -> bool:
    # check whether tetrahedron_1 and _2 are sharing a triangle surface or not.
    
    # generate traiangles
    surface1=get_tetrahedron_surface(tetrahedron_1)
    surface2=get_tetrahedron_surface(tetrahedron_2)
    
    count=0
    for triangle1 in surface1:
        for triangle2 in surface2: # i2-th triangle of tetrahedron2
            if equivalent_triangles(triangle1,triangle2): # equivalent
                count+=1
                break
            else:
                pass
        if count!=0:
            break
        else:
            pass
    if count==0:
        return False
    else:
        return True

def equivalent_triangles(triangle1: NDArray[np.int64], triangle2: NDArray[np.int64]) -> bool:
    """Check whether triangle1 and triangle2 are equivalent or not.
    """
    a=np.vstack([triangle1,triangle2])
    a=remove_doubling_in_perp_space(a)
    if len(a)==3:
        return True # equivalent traiangle
    else:
        return False # not equivalent traiangles

def equivalent_edges(edge1: NDArray[np.int64], edge2: NDArray[np.int64]) -> bool:
    """Check whether edge1 and edge2 are equivalent or not.
    """
    a=np.vstack([edge1,edge2])
    a=remove_doubling_in_perp_space(a)
    if len(a)==2:
        return True # equivalent
    else:
        return False # not equivalent

def equivalent_vertices(vertex1: NDArray[np.int64], vertex2: NDArray[np.int64]) -> bool:
    xyz1=projection3(vertex1[0],vertex1[1],vertex1[2],vertex1[3],vertex1[4],vertex1[5])
    xyz2=projection3(vertex2[0],vertex2[1],vertex2[2],vertex2[3],vertex2[4],vertex2[5])
    if np.all(xyz1==xyz2):
        return True # equivalent
    else:
        return False



##############################
# objのソート関連
##############################
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
    sort tehtahedra in an object
    
    """
    out=np.zeros(vts.shape,dtype=np.int64)
    centroids=np.zeros(len(obj),dtype=np.float64)
    tmp=np.zeros((obj.shape,3),dtype=np.int64)
    
    # 各tetrahedronの頂点xyzをx順にソートすると同時に重心を求めておく。
    for i1 in range(len(obj)):
        tmp[i1]=sort_vctors(obj[i1])
        centroids[i1]=centroid(obj[i1])
    
    # 四面体の重心xyzのx順にソート
    #indx=np.argsort(centroids,axis=0)
    indx=centroids[np.argsort(centroids[:,0])]
    
    for i1 in range(n1):
        out[i1]=tmp[indx[i1][0]]
    return out














##############################
####
####
#### WIP: 削除関連
####
####
##############################
def remove_vectors(vts1: NDArray[np.int64], vts2: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    # 6次元ベクトルリストvlst1から6次元ベクトルリストvlst2にあるベクトルを抜きとる
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
    """
    # 6次元ベクトルリストvlst1から6次元ベクトルvt2を抜きとる
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
            out[i1]=vts1[lst[i1]]
        return out
    else:
        return vlst1



#################################
####
####
#### WIP: objの合体
####
####
#################################
def merge_two_tetrahedra(tetrahedron_1: NDArray[np.int64], tetrahedron_2: NDArray[np.int64]) -> NDArray[np.int64]:
    # merge two tetrahedra
    
    # volume
    #vol1=tetrahedron_volume_6d(tetrahedron_1)
    #vol2=tetrahedron_volume_6d(tetrahedron_2)
    #vol3=add(vol1,vol2)
    
    if equivalent_tetrahedra(tetrahedron_1,tetrahedron_2): # tet1とtet2が共通する三角形を持つ場合
        vtx1=remove_vectors(tetrahedron_1,tetrahedron_2) # tet1からtet1とtet2の共通頂点を消す --> 頂点1
        vtx2=remove_vectors(tetrahedron_2,tetrahedron_1) # tet2からtet1とtet2の共通頂点を消す --> 頂点2
        vtx_common=get_common_triangle_in_two_tetrahedra(tetrahedron_1,tetrahedron_2) # tet1とtet2の共通する三角形
        line_segment=np.vstack([vtx1,vtx2])# 頂点1と頂点２を繋いだ辺
        flg=0
        for vtx in vtx_common:
            # 2つの四面体を一つの四面体に結合できる時、その頂点は上の三角形の3つの頂点のうち2つの頂点と頂点1と頂点２。
            if point_on_segment(vtx,line_segment):
                tmp=remove_vector(vtx_common,vtx)
                tetrahedron_new=np.stack(tmp,line_segment)
                flg+=1
                break
        else:
            pass
        if flg!=0:
            return tetrahedron_new
        else:
            return 
    else:
        return 

def merge_two_tetrahedra_in_obj(obj: NDArray[np.int64]) -> NDArray[np.int64]:
    
    num=len(obj)
    
    
    return obj

def get_common_triangle_in_two_tetrahedra(tetrahedron_1: NDArray[np.int64], tetrahedron_2: NDArray[np.int64]) -> NDArray[np.int64]:
    # generate traiangles
    surface1=get_tetrahedron_surface(tetrahedron_1)
    surface2=get_tetrahedron_surface(tetrahedron_2)
    
    count=0
    for triangle1 in surface1:
        for triangle2 in surface2: # i2-th triangle of tetrahedron2
            if equivalent_triangles(triangle1,triangle2): # equivalent
                count+=1
                break
            else:
                pass
        if count!=0:
            break
        else:
            pass
    if count==1:
        return triangle1
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
            if point_on_segment(edge1a,edge1b,edge2b):
                out=np.vstack(edge1b,edge2b)
                counter+=1
                break
            else:
                pass
        else:
            pass
    if counter!=0:
        return out
    else:
        return 

def coplanar_check_two_triangles(triange1: NDArray[np.int64], triange2: NDArray[np.int64]) -> bool:
    
    vtx=np.vstack([triange1,triange2])
    vtx=remove_doubling_in_perp_space(vtx)
    
    # メモ
    # vtxはソートされており、coplanar_checkやcoplanar_check_numeric_tau
    # での外積計算の際に小さい値になるとcoplanar判定を間違うので注意。
    
    #if coplanar_check(vtx): # math1のバージョン
    if coplanar_check_numeric_tau(vtx): # numericalcのバージョン
        return True # coplanar
    else:
        return False


#################################
#### tetrahedralization関連
#################################

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
    
def tetrahedralization_points(points: NDArray[np.int64]) -> NDArray[np.int64]:
    
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
            vol=tetrahedron_volume_6d(tmp3)
            if vol[0]==0 and vol[1]==0:
                pass
            else:
                if counter==0:
                    tmp1=tmp3.reshape(72) # 4*6*3=72
                else:
                    tmp1=np.append(tmp1,tmp3)
                counter+=1
        if counter!=0:
            return tmp1.reshape(int(len(tmp1)/72),4,6,3) # 4*6*3=72
        else:
            return 
    else:
        return 




#################################
####
####
#### WIP: generate_convex_hull関連
####
####
#################################
## もしobjectが凸包であれば、頂点集合が得られれば簡素化できる。そのために凸包かどうかチェックする必要がある。
## 　　表面の三角形のセット by generator_surface_1(obj)
## 　　->三角形の簡素化 by surface_cleaner(surface,num_cycle)
## 　　->頂点集合
## 　　-> ドロネー分割
## 　　->処理前後で体積変化なしであれば凸包だと判断。
def generate_convex_hull(obj: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    objの凸包を得る。
    
    アルゴリズム
    1. objの表面三角形を得る。
    2. 無駄な表面三角形をなくす by surface_cleaner()。
    3. objの頂点座標を得る
    4. 四面体分割
    
    objが凸包であれば、この関数を実行することで、よりシンプルに四面体分割されたobjを得ることができる。
    """
    # 1
    triangle_surface=generator_surface_1(obj)
    
    # 2
    edge_surface=surface_cleaner(triangle_surface)
    
    # 3
    tmp=edge_surface[0]
    for i1 in range(1,len(edge_surface)):
        tmp=np.vstack([tmp,edge_surface[i1]])
    tmp=remove_doubling_in_perp_space(tmp)
    
    # 4
    return tetrahedralization_points(tmp)

def surface_cleaner(surface: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    同一平面上にある三角形ごとにグループ分けする
    
    各グループにおいて、以下を行う．
        三角形の３辺が、他のどの三角形とも共有していない辺を求める
        ２つの辺が１つの辺にまとめられるのであれば、まとめる
        辺の集合をアウトプット
    """
    # 同一平面上にある三角形を求め、集合lst_setsとする
    lst_sets=get_sets_of_coplanar_triangles(surface)
    #print('num. of lst_sets:',len(lst_sets))
    
    #同一平面上にある三角形の辺のうち、どの三角形とも共有していない独立な辺を求める．
    for i in range(len(lst_sets)):
        #print('num. of coplanar triangles',len(lst_sets[i]))
        edges=gen_unique_edges_of_coplanar_triangles(lst_sets[i])
        if i==0:
            out=edges
        else:
            out=np.vstack([out,edges])
    return out

def get_sets_of_coplanar_triangles(surface):
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
        triangle=surface[i1]
        #print('%d-th triangle'%(i1))
        for i2 in lst_indx_group:
            if coplanar_check_two_triangles(triangle,surface[i2]):
                counter+=1
                break
            else:
                pass
        if counter==0:
            #print('     non coplanar')
            lst_indx_triangle.append(i1)
            lst_indx_group.append(i1)
        else:
            #print('     coplanar with %d'%(i2))
            lst_indx_triangle.append(i2)
    #print(' num. of groups:',len(lst_indx_group))
    lst_sets=[]
    for i1 in lst_indx_group:
        #print(i1)
        tmp=[]
        for i2 in range(num):
            if i1==lst_indx_triangle[i2]:
                #print('  ',i2)
                tmp.append(surface[i2])
        num_triangle=len(tmp)
        #print('  num. of triangles',num_triangle)
        a=np.zeros((num_triangle,3,6,3),dtype=np.int64)
        for i2 in range(num_triangle):
            a[i2]=tmp[i2]
        lst_sets.append(a)
    return lst_sets

def gen_unique_edges_of_coplanar_triangles(coplanar_triangles):
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



#################################
####
####
#### WIP: B-reps
####
####
#################################
def breps(tetrahedron):
    """
    四面体の頂点を指す6d vectors (TAU-style)から
    境界表現 (Boundary Representation, B-reps)
    に変換する。
    
    四面体の重心を求める。稜線
    """
    return

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
    
    
    