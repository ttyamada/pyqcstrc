#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
from scipy.spatial import Delaunay
import time # in object_subtraction_dev1, tetrahedron_not_obj

from math1 import (centroid, 
                    centroid_obj,
                    coplanar_check,
                    projection3,
                    sub_vectors,
                    add_vectors, 
                    mul_vector,
                    det_matrix,
                    dot_product,
                    inner_product,
                    outer_product,
                    add,
                    sub,
                    mul,
                    div)
from numericalc import (length_numerical,
                        #check_intersection_segment_surface_numerical,
                        get_internal_component_sets_numerical,
                        get_internal_component_numerical,
                        check_intersection_segment_surface_numerical_6d_tau,
                        numeric_value,
                        numerical_vector,
                        check_intersection_two_segment_numerical_6d_tau,
                        check_intersection_segment_surface_numerical_6d_tau,
                        inside_outside_tetrahedron,
                        inside_outside_tetrahedron_tau,
                        on_out_surface)
from utils import (remove_doubling_in_perp_space,
                    obj_volume_6d,
                    tetrahedron_volume_6d,
                    generator_surface_1,
                    generator_unique_triangles,
                    generator_unique_edges,
                    )
                    
TAU=(1+np.sqrt(5))/2.0
EPS=1e-6

def decomposition(p):
    try:
        tri=Delaunay(p)
    except:
        print('error in decomposition()')
        tmp=[0]
    else:
        #for i in range(len(tri.simplices)):
        #    tet=tri.simplices[i]
        #    tmp.append([tet[0],tet[1],tet[2],tet[3]])
        for tet in tri.simplices:
            tmp.append([tet[0],tet[1],tet[2],tet[3]])
    return tmp

def ball_radius_obj(obj,centroid):
    """estimate maximum distance between verices of given OBJ and its centroid.
    
    Parameters
    ----------
    obj: array (ndim=4)
        in TAU-style
    centroid: array, (ndim=2)
        a 6-dimensional coordinates in TAU-style
    
    Returns
    -------
    length: float
    
    """
    #print("ball_radius_obj")
    
    vertices=remove_doubling_in_perp_space(obj)
    dd=0
    for v in vertices:
        a=sub_vectors(v,centroid)
        a=projection3(a)
        dd1=length_numerical(a)
        if dd1>dd:
            dd=dd1
        else:
            pass
    return dd

def ball_radius(tetrahedron,centroid):
    #  this transforms a tetrahedron to a boll which covers the tetrahedron
    #  the centre of the boll is the centroid of the tetrahedron.
    return ball_radius_obj(tetrahedron,centroid)

def distance_in_perp_space(vt1,vt2):
    a=sub_vectors(vt1,vt2)
    a=projection3(a)
    return length_numerical(a)

def rough_check_intersection_tetrahedron_obj(tetrahedron,cententer,distance):
    
    cen1=centroid(tetrahedron)
    dd1=ball_radius(tetrahedron,cen1)
    dd0=distance_in_perp_space(cen1,cententer)
    if dd0 <= dd1+distance: # two balls are intersecting.
        return True
    else: #
        return False

def check_intersection_two_tetrahedron_4(tetrahedron_1,tetrahedron_2):
    
    # checking whether tetrahedron_1 is fully inside tetrahedron_2 or not
    counter2=0
    for vtx in tetrahedron_1:
        if inside_outside_tetrahedron_tau(vtx,tetrahedron_2): # inside
            pass
        else:
            counter2+=1
            break
    # checking whether tetrahedron_2 is fully inside tetrahedron_3 or not
    counter3=0
    for vtx in tetrahedron_2:
        if inside_outside_tetrahedron_tau(vtx,tetrahedron_1): # inside
            pass
        else:
            counter3+=1
            break
    if counter2==0:
        return 1 # tetrahedron_1 is fully inside tetrahedron_2
    elif counter3==0:
        return 2 # tetrahedron_2 is fully inside tetrahedron_1
    else:
        #
        # -----------------
        # tetrahedron_1
        # -----------------
        # vertex 1: tetrahedron_1[0],  consist of (a1+b1*TAU)/c1, ... (a6+b6*TAU)/c6    a_i,b_i,c_i = tetrahedron_1[0][i:0~5][0],tetrahedron_1[0][i:0~5][1],tetrahedron_1[0][i:0~5][2]
        # vertex 2: tetrahedron_1[1]
        # vertex 3: tetrahedron_1[2]
        # vertex 4: tetrahedron_1[3]
        #
        # 4 surfaces of tetrahedron_1
        # surface 1: v1,v2,v3
        # surface 2: v1,v2,v4
        # surface 3: v1,v3,v4
        # surface 4: v2,v3,v4
        #
        # 6 edges of tetrahedron_1
        # edge 1: v1,v2
        # edge 2: v1,v3
        # edge 3: v1,v4
        # edge 4: v2,v3
        # edge 5: v2,v4
        # edge 6: v3,v4
        #
        # -----------------
        # tetrahedron_2
        # -----------------
        # vertex 1: tetrahedron_2[0]
        # vertex 2: tetrahedron_2[1]
        # vertex 3: tetrahedron_2[2]
        # vertex 4: tetrahedron_2[3]
        #
        # 4 surfaces of tetrahedron_2
        # surface 1: w1,w2,w3
        # surface 2: w1,w2,w4
        # surface 3: w1,w3,w4
        # surface 4: w2,w3,w4
        #
        # 6 edges of tetrahedron_2
        # edge 1: w1,w2
        # edge 2: w1,w3
        # edge 3: w1,w4
        # edge 4: w2,w3
        # edge 5: w2,w4
        # edge 6: w3,w4
        #
        # case 1: intersection between (edge of tetrahedron_1) and (surface of tetrahedron_2)
        # case 2: intersection between (edge of tetrahedron_2) and (surface of tetrahedron_1)
        #
        # combination_index
        # e.g. v1,v2,w1,w2,w3 (edge 1 and surface 1) ...
        comb=[\
        [0,1,0,1,2],\
        [0,1,0,1,3],\
        [0,1,0,2,3],\
        [0,1,1,2,3],\
        [0,2,0,1,2],\
        [0,2,0,1,3],\
        [0,2,0,2,3],\
        [0,2,1,2,3],\
        [0,3,0,1,2],\
        [0,3,0,1,3],\
        [0,3,0,2,3],\
        [0,3,1,2,3],\
        [1,2,0,1,2],\
        [1,2,0,1,3],\
        [1,2,0,2,3],\
        [1,2,1,2,3],\
        [1,3,0,1,2],\
        [1,3,0,1,3],\
        [1,3,0,2,3],\
        [1,3,1,2,3],\
        [2,3,0,1,2],\
        [2,3,0,1,3],\
        [2,3,0,2,3],\
        [2,3,1,2,3]]
    
        counter1=0
        for c in comb: # len(combination_index) = 24
            # case 1: intersection between
            # 6 edges of tetrahedron_1
            # 4 surfaces of tetrahedron_2
            segment=np.stack([tetrahedron_1[c[0]],tetrahedron_1[c[1]]])
            surface=np.stack([tetrahedron_2[c[2]],tetrahedron_2[c[3]],tetrahedron_2[c[4]]])
            #print(segment.shape)
            #print(surface.shape)
            if check_intersection_segment_surface_numerical_6d_tau(segment,surface)==0: # intersectiing
                counter1+=1
                break
            else:
                pass
            # case 2: intersection between
            # 6 edges of tetrahedron_2
            # 4 surfaces of tetrahedron_1
            segment=np.stack([tetrahedron_2[c[0]],tetrahedron_2[c[1]]])
            surface=np.stack([tetrahedron_1[c[2]],tetrahedron_1[c[3]],tetrahedron_1[c[4]]])
            if check_intersection_segment_surface_numerical_6d_tau(segment,surface): # intersectiing
                counter1+=1
                break
            else:
                pass
        if counter1>0:
            return 3 # intersecting
        else:
            return 0 # no intersection

def intersection_two_segment(segment_1,segment_2):
    """check intersection between two line segments.
    
    Parameters
    ----------
    line_segment_1 line_segment_2: array
        6-dimensional coordinates of line segment,(xyzuvw1, xyzuvw2) and (xyzuvw3, xyzuvw4), in TAU-style
   
    Returns
    -------
    
    """
    # check whether two line segments are intersecting or not by numerical calc.
    flg=check_intersection_two_segment_numerical(segment_1,segment_2)
    if flg!=3: # intersecting
        # calc in TAU-style
        vec6AB=sub_vectors(segment_1[1],segment_1[0])
        vecAB=projection3(vec6AB)                 # AB
        #
        tmp=sub_vectors(segment_2[1],segment_2[0])
        vecCD=projection3(tmp)                    # CD
        #
        tmp=sub_vectors(segment_1[0],segment_2[0])
        vecCA=projection3(tmp)                    # CA
        
        comb=[\
        [0,1,2],\
        [1,2,0],\
        [2,0,1]]
        c=comb[flg]
        #
        #bunbo=vecAB[c[0]]*vecCD[c[1]]-vecCD[c[0]]*vecAB[c[1]]
        tmp1=mul(vecAB[c[0]],vecCD[c[1]])
        tmp2=mul(vecCD[c[0]],vecAB[c[1]])
        bunbo=sub(tmp1,tmp2)
        #
        #s=(vecCD[c[0]]*vecCA[c[1]]-vecCA[c[0]]*vecCD[c[1]])/bunbo
        tmp1=mul(vecCD[c[0]],vecCA[c[1]])
        tmp2=mul(vecCA[c[0]],vecCD[c[1]])
        tmp1=sub(tmp1,tmp2)
        s=div(tmp1,bunbo)
        #
        # OP = OA + s*AB
        tmp=mul_vector(vecAB,s)
        return add_vectors(segment_1[0],tmp)
    else: # no intersection
        return 

def intersection_segment_surface(segment,surface):
    """check intersection between a line segment and a triangle.
    
    Möller–Trumbore intersection algorithm
    https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    
    Parameters
    ----------
    line_segment: array
        6-dimensional coordinates of line segment,xyzuvw1, xyzuvw2, in TAU-style
    triangle: array
        containing 6-dimensional coordinates of tree vertecies of a triangle, xyzuvw1, xyzuvw2, xyzuvw3, xyzuvw4 in TAU-style
    
    Returns
    -------
    
    """
    # check whether the line segment and the surface are intersecting or not by numerical calc.
    if check_intersection_segment_surface_numerical_6d_tau(segment,surface): # intersecting
        # calc in TAU-style
        vec6AB=sub_vectors(segment[1],segment[0])
        vecAB=projection3(vec6AB)                 # AB # R
        #
        tmp=sub_vectors(surface[1],surface[0])
        vecCD=projection3(tmp)                 # CD # E1
        #
        tmp=sub_vectors(surface[2],surface[0])
        vecCE=projection3(tmp)                 # CE # E2
        #
        tmp=sub_vectors(segment[0],surface[0])
        vecCA=projection3(tmp)                 # CA # T
        
        vecP=outer_product(vecAB,vecCE) # P
        vecQ=outer_product(vecCA,vecCD) # Q
        
        bunbo=inner_product(vecP,vecCD)
        
        bunshi=inner_product(vecQ,vecCE)
        t=div(bunshi,bunbo)
        
        # intersecting point: OA + t*AB
        tmp=mul_vector(vec6AB,t) # t*AB
        #print('   t=',numeric_value(t))
        return add_vectors(segment[0],tmp).reshape(1,6,3)
    else: # no intersection
        return 

def intersection_two_tetrahedron_4(tetrahedron_1,tetrahedron_2):
    #print('intersection_two_tetrahedron_4()')
    #
    # -----------------
    # tetrahedron_1
    # -----------------
    # vertex 1: tetrahedron_1[0],  consist of (a1+b1*TAU)/c1, ... (a6+b6*TAU)/c6    a_i,b_i,c_i = tetrahedron_1[0][i:0~5][0],tetrahedron_1[0][i:0~5][1],tetrahedron_1[0][i:0~5][2]
    # vertex 2: tetrahedron_1[1]
    # vertex 3: tetrahedron_1[2]
    # vertex 4: tetrahedron_1[3]
    #
    # 4 surfaces of tetrahedron_1
    # surface 1: v1,v2,v3
    # surface 2: v1,v2,v4
    # surface 3: v1,v3,v4
    # surface 4: v2,v3,v4
    #
    # 6 edges of tetrahedron_1
    # edge 1: v1,v2
    # edge 2: v1,v3
    # edge 3: v1,v4
    # edge 4: v2,v3
    # edge 5: v2,v4
    # edge 6: v3,v4
    #
    # -----------------
    # tetrahedron_2
    # -----------------
    # vertex 1: tetrahedron_2[0]
    # vertex 2: tetrahedron_2[1]
    # vertex 3: tetrahedron_2[2]
    # vertex 4: tetrahedron_2[3]
    #
    # 4 surfaces of tetrahedron_2
    # surface 1: w1,w2,w3
    # surface 2: w1,w2,w4
    # surface 3: w1,w3,w4
    # surface 4: w2,w3,w4
    #
    # 6 edges of tetrahedron_2
    # edge 1: w1,w2
    # edge 2: w1,w3
    # edge 3: w1,w4
    # edge 4: w2,w3
    # edge 5: w2,w4
    # edge 6: w3,w4
    #
    # case 1: intersection between (edge of tetrahedron_1) and (surface of tetrahedron_2)
    # case 2: intersection between (edge of tetrahedron_2) and (surface of tetrahedron_1)
    #
    # combination_index
    # e.g. v1,v2,w1,w2,w3 (edge 1 and surface 1) ...
    comb=[\
    [0,1,0,1,2],\
    [0,1,0,1,3],\
    [0,1,0,2,3],\
    [0,1,1,2,3],\
    [0,2,0,1,2],\
    [0,2,0,1,3],\
    [0,2,0,2,3],\
    [0,2,1,2,3],\
    [0,3,0,1,2],\
    [0,3,0,1,3],\
    [0,3,0,2,3],\
    [0,3,1,2,3],\
    [1,2,0,1,2],\
    [1,2,0,1,3],\
    [1,2,0,2,3],\
    [1,2,1,2,3],\
    [1,3,0,1,2],\
    [1,3,0,1,3],\
    [1,3,0,2,3],\
    [1,3,1,2,3],\
    [2,3,0,1,2],\
    [2,3,0,1,3],\
    [2,3,0,2,3],\
    [2,3,1,2,3]]
    
    #tmp=np.array([])
    #tmp1=np.array([])
    #array0=np.zeros((6,3),dtype=np.int64)
    
    counter=0
    for c in comb:
        # case 1: intersection between (edge of tetrahedron_1) and (surface of tetrahedron_2)
        segment=np.stack([tetrahedron_1[c[0]],tetrahedron_1[c[1]]])
        surface=np.stack([tetrahedron_2[c[2]],tetrahedron_2[c[3]],tetrahedron_2[c[4]]])
        vtx=intersection_segment_surface(segment,surface)
        if np.all(vtx==None):
            pass
        else:
            #print('tmp1',tmp1)
            if counter==0 :
                tmp=vtx # intersection points
            else:
                tmp=np.vstack([tmp,vtx]) # intersecting points
            counter+=1
            #print('tmp',tmp)
        # case 2: intersection between (edge of tetrahedron_2) and (surface of tetrahedron_1)
        segment=np.stack([tetrahedron_2[c[0]],tetrahedron_2[c[1]]])
        surface=np.stack([tetrahedron_1[c[2]],tetrahedron_1[c[3]],tetrahedron_1[c[4]]])
        vtx=intersection_segment_surface(segment,surface)
        if np.all(vtx==None):
            pass
        else:
            #print('tmp1',tmp1)
            if counter==0:
                tmp=vtx # intersection points
            else:
                #print('   tmp.shape',tmp.shape)
                #print('   tmp1.shape',tmp1.shape)
                tmp=np.vstack([tmp,vtx]) # intersecting points
            counter+=1
            #print('tmp',tmp)
    #tmp=remove_doubling_in_perp_space(tmp)
    #print('   (1) num of points=',len(tmp))
    #a=get_internal_component_sets_numerical(tmp)
    #print(a)
    
    # get vertces of tetrahedron_1 that are inside tetrahedron_2
    for i1 in range(len(tetrahedron_1)):
        vtx=tetrahedron_1[i1]
        if inside_outside_tetrahedron_tau(vtx,tetrahedron_2): # inside
            #print('tetrahedron_1[i1]',tetrahedron_1[i1])
            #a=get_internal_component_numerical(tetrahedron_1[i1])
            #print('      vertex of tet1',a)
            if counter==0:
                tmp=vtx.reshape(1,6,3)
            else:
                tmp=np.vstack([tmp,[vtx]])
            counter+=1
            #print('tmp',tmp)
    # get vertces of tetrahedron_2 that are inside tetrahedron_1
    for i1 in range(len(tetrahedron_2)):
        vtx=tetrahedron_2[i1]
        if inside_outside_tetrahedron_tau(vtx,tetrahedron_1): # inside
            #print('tetrahedron_2[i1]',tetrahedron_2[i1])
            if counter==0:
                tmp=vtx.reshape(1,6,3)
            else:
                tmp=np.vstack([tmp,[vtx]])
            counter+=1
        else:
            pass
    #print('tmp.shape',tmp.shape)
    #tmp=tmp.reshape(int(len(tmp)/6),6,3)
    #print('tmp:',tmp)
    #tmp=remove_doubling_in_perp_space(tmp)
    #print('   (2) num of points=',len(tmp))
    #a=get_internal_component_sets_numerical(tmp)
    #print(a)
    
    #tmp4=np.array([[[[0]]]])
    if counter>=4:
        tmp=remove_doubling_in_perp_space(tmp)
        #print(len(tmp))
        if len(tmp)>=4:
            # Tetrahedralization
            if coplanar_check(tmp): # coplanar
                pass
            else:
                if len(tmp)==4:
                    tmp4=tmp.reshape(1,4,6,3)
                else:
                    tmp4=tetrahedralization_points(tmp)
                v=obj_volume_6d(tmp4)
                nv=numeric_value(v)
                #print('     common vol:',v,nv)
                return tmp4
        else:
            return 
    else:
        return 

def decomposition(tmp2v):
    try:
        tri=Delaunay(tmp2v)
    except:
        print('error in decomposition()')
        tmp=[0]
    else:
        tmp=[]
        for i in range(len(tri.simplices)):
            tet=tri.simplices[i]
            tmp.append([tet[0],tet[1],tet[2],tet[3]])
    return tmp
    
def tetrahedralization_points(points):
    
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

def intersection_two_obj_1(obj1,obj2,kind=None):
    """
    Return an intersection between two objects.
    
    Parameters
    ----------
    obj1 : ndarray
        a set of tetrahedra to be intersected with obj2.
    obj2 : ndarray
        a set of tetrahedra to be intersected with obj1.
    kind : {'standard', 'simple'}, optional
        The default is 'standard'. 
    
    Returns
    -------
    intersection between obj1 and obj2 : ndarray
        Array of the same type and shape as `obj1` and `obj2`.
    
    Notes
    -----
    
    'standard' intersection ...
    
    
    
    'simple' intersection ...
    
    """
    
    cent2=centroid_obj(obj2)
    dd2=ball_radius_obj(obj2,cent2)
    
    counter0=0
    for tetrahedron1 in obj1:
        if rough_check_intersection_tetrahedron_obj(tetrahedron1,cent2,dd2):
            counter1=0
            #vol1=tetrahedron_volume_6d(tetrahedron1)
            #print('vol1',vol1,numeric_value(vol1))
            for tetrahedron2 in obj2:
                flag=check_intersection_two_tetrahedron_4(tetrahedron1,tetrahedron2)
                #
                # tetrahedron_1 is fully inside tetrahedron_2
                if flag==1:
                    if counter0==0:
                        common4=tetrahedron1
                        counter0+=1
                    else:
                        common4=np.vstack([common4,tetrahedron1])
                    break
                #
                # tetrahedron_2 is fully inside tetrahedron_1
                elif flag==2:
                    if counter1==0:
                        tmp_common4=tetrahedron2
                        counter1+=1
                    else:
                        tmp_common4=np.vstack([tmp_common4,tetrahedron2])
                #
                # tetrahedron_1 and tetrahedron_2 are intersecting
                elif flag==3:
                    tmp4=intersection_two_tetrahedron_4(tetrahedron1,tetrahedron2)
                    ###
                    ### Comment:
                    ### intersection_two_tetrahedron_4()で見つからないのに
                    ### check_intersection_two_tetrahedron_4()では交差して
                    ### いると判定される場合がある。2つの四面体の交差点が3つ以下の場合
                    ###（つまり接している場合）がこれに相当するので、以下のようにする。
                    ###
                    if np.all(tmp4==None):
                        pass
                    else:
                        #print(i)
                        #v=obj_volume_6d(tmp4)
                        #print('.    common vol:',v,numeric_value(v))
                        if counter1==0:
                            tmp_common4=tmp4
                            counter1+=1
                        else:
                            tmp_common4=np.vstack((tmp_common4,tmp4))
                    #if counter1==0:
                    #    tmp_common4=tmp4
                    #    counter1+=1
                    #else:
                    #    tmp_common4=np.vstack((tmp_common4,tmp4))
                else:
                    pass
                #i+=1
                
            if counter1!=0:
                #print('tmp_common4',tmp_common4)
                #vol2=obj_volume_6d(tmp_common4)
                #print('vol2',vol2,numeric_value(vol2))
                if kind=='simple':
                    vol1=tetrahedron_volume_6d(tetrahedron1)
                    vol2=obj_volume_6d(tmp_common4)
                    if np.all(vol1==vol2):
                        if counter0==0:
                            common4=tetrahedron1.reshape(1,4,6,3)
                            #print('common4.shape',common4.shape)
                            counter0+=1
                        else:
                            #common4=np.concatenate([common4,tetrahedron1])
                            common4=np.vstack([common4,[tetrahedron1]])
                            #print('common4.shape',common4.shape)
                    else:
                        if counter0==0:
                            common4=tmp_common4
                            counter0+=1
                            #print('tmp_common4.shape',tmp_common4.shape)
                        else:
                            #common4=np.concatenate([common4,tmp_common4])
                            common4=np.vstack([common4,tmp_common4])
                            #print('tmp_common4.shape',tmp_common4.shape)
                else:
                    if counter0==0:
                        common4=tmp_common4
                        counter0+=1
                        #print('tmp_common4.shape',tmp_common4.shape)
                    else:
                        #common4=np.concatenate([common4,tmp_common4])
                        common4=np.vstack([common4,tmp_common4])
                        #print('tmp_common4.shape',tmp_common4.shape)
                #print(common4.shape)
                #return common4
    
    if counter0>0:
        return common4
    else:
        return 

def intersection_two_obj_convex(obj1,obj2,vervose=0):
    """
    # This is very simple but work correctly only when each subdivided 
    # three ODs (i.e part part, ODA and ODB) are able to define as a
    # set of tetrahedra.
    """
    if vervose>0:
        print("       start: intersection_two_obj_convex()")
    
    obj1_surf=generator_surface_1(obj1)
    obj2_surf=generator_surface_1(obj2)
    obj1_edge=generator_unique_edges(obj1_surf)
    obj2_edge=generator_unique_edges(obj2_surf)
    
    if vervose>0:
        print("         num. of unique triangles in obj1:",len(obj1_surf))
        print("         num. of unique triangles in obj2:",len(obj2_surf))
        print("         num. of unique deges in obj1:",len(obj1_edge))
        print("         num. of unique deges in obj2:",len(obj2_edge))
    
    
    #
    # (1) Extract vertces of 2nd OD which are insede 1st OD --> point_a1
    #     Extract vertces of 2nd OD which are outsede 1st OD --> point_b2
    #
    counter1a=0
    #counter2a=0
    vertices1=remove_doubling_in_perp_space(obj1_edge) # generating vertces of 1st OD
    for vrtx in vertices1:
        counter2a=0
        for tetrahedron2 in obj2:
            if inside_outside_tetrahedron_tau(vrtx,tetrahedron2):
                counter2a+=1
                break
            else:
                pass
        if counter2a>0:
            if counter1a==0:
                point_a1=vrtx.reshape(1,6,3)
            else:
                point_a1=np.vstack([point_a1,[vrtx]])
            counter1a+=1
        #else:
        #    if counter2==0:
        #        point_b2=vrtx.reshape(1,6,3)
        #    else:
        #        point_b2=np.vstack([point_b2,[vrtx]])
        #    counter2+=1
    #
    # (2) Extract vertces of 1st OD which are insede 2nd OD --> point_b1
    #     Extract vertces of 1st OD which are outsede 2nd OD --> point_a2
    #
    counter1b=0
    #counter2b=0
    vertices2=remove_doubling_in_perp_space(obj2_edge) # generating vertces of 2nd OD
    for vrtx in vertices2:
        counter2b=0
        for tetrahedron1 in obj1:
            if inside_outside_tetrahedron_tau(vrtx,tetrahedron1):
                counter2b+=1
                break
            else:
                pass
        if counter2b>0:
            if counter1b==0:
                point_b1=vrtx.reshape(1,6,3)
            else:
                point_b1=np.vstack([point_b1,[vrtx]])
            counter1b+=1
        #else:
        #    if counter2==0:
        #        point_a2=vrtx.reshape(1,6,3)
        #    else:
        #        point_a2=np.vstack([point_a2,[vrtx]])
        #    counter2+=1
    if counter1a==len(vertices1): # obj1 is fully inside obj2
        return obj1
    elif counter1b==len(vertices2): # obj2 is fully inside obj1
        return obj2
    else:
        #
        # (3) Get intersecting points between obj1 and obj2
        #
        counter=0
        for tr1 in obj1_surf:
            for ed2 in obj2_edge:
                if check_intersection_segment_surface_numerical_6d_tau(ed2,tr1): # intersection
                    tmp=intersection_segment_surface(ed2,tr1)
                    if counter==0:
                        p=tmp
                    else:
                        p=np.vstack([p,tmp])
                    counter+=1
                else:
                    pass
        for tr2 in obj2_surf:
            for ed1 in obj1_edge:
                if check_intersection_segment_surface_numerical_6d_tau(ed1,tr2): # intersection
                    tmp=intersection_segment_surface(ed1,tr2)
                    if counter==0:
                        p=tmp
                    else:
                        p=np.vstack([p,tmp])
                    counter+=1
                else:
                    pass
        if counter==0:
            return 
        else:
            point1=remove_doubling_in_perp_space(p)
            #
            # (3) Sum point A, point B and Intersections --->>> common part
            #
            # common part = point1 + point_a1 + point_b1
            points=np.vstack([point1,point_a1,point_b1])
            points=remove_doubling_in_perp_space(points)
            common=tetrahedralization_points(points)
            if np.all(common==None):
                print('no common part')
                return 
            else:
                return common

#########
###
###   WIP
###
#########
def object_subtraction_dev1(obj1,obj2):
    """
    get A not B = A not (A and B)
    obj1: A
    obj2: B
    """
    print('     object_subtraction_dev1()')
    # surface triangles of obj2
    
    print('      generating surface_obj2')
    start = time.time()
    surface_obj2=generator_surface_1(obj2)
    end=time.time()
    time_diff=end-start
    print('         ends in %4.3f sec'%time_diff)  # 処理にかかった時間データ
    
    
    counter1=0
    for tetrahedron in obj1:
        print('      %d-th tetrahedron in obj1'%(counter1))
        a=tetrahedron_not_obj(tetrahedron.reshape(1,4,6,3),obj2,surface_obj2)
        if np.all(a==None):
            pass
        else:
            if counter1==0:
                out=a
            else:
                out=np.vstack([out,a])
        counter1+=1
    return out

def tetrahedron_not_obj(tetrahedron,obj,surface_obj):
    """
    get A not B = A not (A and B)
    tetrahedron: A
    obj: B
    
    surface_obj = surface of B
    """
    
    print('        tetrahedron_not_obj()')
        
    # surface triangles of obj
    #surface_obj=generator_surface_1(obj)
    
    # surface triangles and vertices of common
    print('         intersection_two_obj_1()')
    start=time.time()
    common=intersection_two_obj_1(tetrahedron,obj)
    end=time.time()
    time_diff=end-start
    print('          ends in %4.3f sec'%time_diff)  # 処理にかかった時間データ
    surface_common=generator_surface_1(common)
    #vertx_common=remove_doubling_in_perp_space(surface_common)
    
    vol0=obj_volume_6d(tetrahedron)
    vol1=obj_volume_6d(common)
    #print('tetrahedron volume:',vol0,numeric_value(vol0))
    #print('common volume:',vol1,numeric_value(vol1))
    vol2=sub(vol0,vol1)
    #print('tetrahedron NOT obj:',vol2,numeric_value(vol2))
    
    out=None
    
    # get surface triangles of common part which are on the surface of obj
    counter2=0
    for triangle2 in surface_common:
        counter1=0
        for vrtx2 in triangle2:
            for triangle3 in surface_obj:
                if on_out_surface(vrtx2,triangle3): # on
                    counter1+=1
                    break
                else:
                    pass
        if counter1==3:
            if counter2==0:
                tmp=triangle2.reshape(1,3,6,3)
            else:
                tmp=np.vappend([tmp,[triangle2]])
            counter2+=1
        else:
            pass
    triangle_common=tmp
    #print('triangle_common.shape',triangle_common.shape)
    
    # vertices of common part which are on the surface of obj
    vertx_common=remove_doubling_in_perp_space(triangle_common)
    vertx_common=remove_doubling_in_perp_space(vertx_common)
    
    # get vertices of tetrahedron which are NOT inside obj
    counter2=0
    tetrahedron=tetrahedron.reshape(4,6,3)
    for vrtx1 in tetrahedron:
        counter1=0
        for tet3 in obj:
            #print('vrtx1.shape',vrtx1.shape)
            #print('tet3.shape',tet3.shape)
            if inside_outside_tetrahedron_tau(vrtx1,tet3): # inside
                counter1+=1
                break
            else:
                pass
        if counter1==0:
            if counter2==0:
                tmp=vrtx1.reshape(1,6,3)
            else:
                tmp=np.vstack([tmp1a,[vrtx1]])
            counter2+=1
        else:
            pass
    vrtx1_out=tmp
    #print('vrtx1_out.shape',vrtx1_out.shape)
    #
    #
    #
    # 四面体の4つの頂点のうちobjの外側にある頂点vrtx1_outの個数Nは、1, 2, 3, 4個。
    # 以下、それぞれの場合について考える。
    ############################################################ 
    ### N=1の場合は、triangle_commonにある各三角形と頂点を結んだものが求めたいものになる。
    ############################################################
    if counter2==1:
        print('         case 1')
        counter3=0
        for triangle in triangle_common:
            #print('triangle.shape',triangle.shape)
            tet=np.vstack([vrtx1_out,triangle]).reshape(1,4,6,3)
            #print('tet.shape',tet.shape)
            if counter3==0:
                tmp=tet
            else:
                tmp=np.vappend([tmp,tet])
            counter3+=1
        #print('out.shape',out.shape)
        vol=obj_volume_6d(tmp)
        #print('obtained volume:',vol,numeric_value(vol))
        if np.all(vol==vol2):
            out=tmp
        return out
    ############################################################
    #### N=2の場合。triangle_commonに含まれる三角形の個数で場合分けする。
    ############################################################
    elif counter2==2:
        if len(triangle_common)==1:
            print('         case 2-1')
            out=tetrahedralization_points(np.vstack([vrtx1_out,triangle_common]))
        elif len(triangle_common)==2:
            print('         case 2-2')
            combination=[\
            [1,2],\
            [0,2],\
            [0,1]]
            tr1=triangle_common[0]
            tr2=triangle_common[1]
            for c1 in combination:
                counter=0
                for c2 in combination:
                    edge1=np.vstack([tr1[c1[0]],tr1[c1[1]]])
                    edge2=np.vstack([tr2[c1[0]],tr2[c1[1]]])
                    tmp=np.vstack([edge1,edge2])
                    edge_common=remove_doubling_in_perp_space(tmp)
                    if len(tmp)==2:
                        counter+=1
                        break
                if counter!=0:
                    beak
            tet1=np.vstack([vrtx1_out,edge_common])
            vola=obj_volume_6d(tet1)
            combination=[\
            [1,0],\
            [0,1]]
            for c1 in combination: 
                tet2=np.vstack([vrtx1_out[c1[0]],tr1])
                tet3=np.vstack([vrtx1_out[c1[1]],tr2])
                tet_tot=np.vstack([tet1,tet2,tet3])
                vol_tot=obj_volume_6d(tet_tot)
                if np.all(vol_tot==vol2):
                    out=tet_tot
                    break
        elif len(triangle_common)==3:
            print('         case 2-3')
        elif len(triangle_common)==4:
            print('         case 2-4')
        else:
            print('         case 2-X')
        return out
    ############################################################
    #### N=3の場合。triangle_commonに含まれる三角形の個数で場合分けする。
    ############################################################
    elif counter2==3:
        if len(triangle_common)==1:
            print('         case 3-1')
            out=tetrahedralization_points(np.vstack([vrtx1_out,triangle_common]))
        elif len(triangle_common)==2:
            print('         case 3-2')
        elif len(triangle_common)==3:
            print('         case 3-3')
        elif len(triangle_common)==4:
            print('         case 3-4')
        else:
            print('         case 3-X')
        return out
    ############################################################
    #### N=4の場合。triangle_commonに含まれる三角形の個数で場合分けする。
    ############################################################
    else:
        if len(triangle_common)==3:
            print('         case 4-3')
        elif len(triangle_common)==4:
            print('         case 4-4')
        else:
            print('         case 4-X')
        return out

def difference_tetrahedron_not_obj(tetrahedron,obj):
    """
    tetrahedron AND (NOT object)
    
    # tetrahedronからobjを引いた物体の表面にある三角形を求める（本当は四面体を求めたいが難しい）
    
    アルゴリズム
    1. tetrahedronとobjの共通部分Aを求める。
    2. A表面の三角形T1を求める
    3. objの表面の三角形T2を求める。
    4. T1のうち、T2上にあるものT1'を得る。T1'は求めたい差分tetrahedron NOT objの表面の一部になる。
    5. T1のうち、T2の外側にあるものとT1'を合わせる。
    """
    
    vol0=obj_volume_6d(tetrahedron)
    print('tetrahedron volume:',vol0,numeric_value(vol0))
    
    # 1. tetrahedronとobjの共通部分Aを求める。
    # intersection between tetrahedron and obj
    common=intersection_two_obj_1(tetrahedron,obj,kind='standard')
    vol1=obj_volume_6d(common)
    print('common volume:',vol1,numeric_value(vol1))
    
    # 2. A表面の三角形T1を求める
    surface_obj2=generator_surface_1(common)
    
    # 3. objの表面の三角形T2を求める。
    # surface of obj
    surface_obj3=generator_surface_1(obj)
    
    # 4. T1のうち、T2上にあるものT1'を得る。T1'は求めたい差分tetrahedron NOT objの表面の一部になる。
    # get triangles of common part which are on the surface of obj3
    counter2=0
    for triangle2 in surface_obj2:
        counter1=0
        for vrtx2 in triangle2:
            for triangle3 in surface_obj3:
                if on_out_surface(vrtx2,triangle3): # on
                    counter1+=1
                    break
                else:
                    pass
        if counter1==4:
            if counter2==0:
                tmp=triangle2
            else:
                tmp=np.vappend([tmp,triangle2])
            counter2+=1
        else:
            pass
    
    # 5. T1のうち、T2の外側にあるものとT1'を合わせる。
    ###
    ### WIP
    ###
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
