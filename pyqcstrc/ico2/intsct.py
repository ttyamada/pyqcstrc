#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
from scipy.spatial import Delaunay

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
                        check_intersection_segment_surface_numerical_6d_tau,
                        numeric_value,
                        numerical_vector,
                        check_intersection_two_segment_numerical_6d_tau,
                        check_intersection_segment_surface_numerical_6d_tau,
                        inside_outside_tetrahedron,
                        inside_outside_tetrahedron_tau)
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
    
    # checking whether tetrahedron_1 is fully inside tetrahedron_2 or not
    counter2=0
    for i1 in range(4):
        if inside_outside_tetrahedron_tau(tetrahedron_1[i1],tetrahedron_2): # inside
            pass
        else:
            counter2+=1
            break
        
    # checking whether tetrahedron_2 is fully inside tetrahedron_3 or not
    counter3=0
    for i1 in range(4):
        if inside_outside_tetrahedron_tau(tetrahedron_2[i1],tetrahedron_1): # inside
            pass
        else:
            counter3+=1
            break
    
    if counter2==0 :
        return 1 # tetrahedron_1 is fully inside tetrahedron_2
    elif counter3==0:
        return 2 # tetrahedron_2 is fully inside tetrahedron_1
    else:
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
        return np.zeros((6,3),dtype=np.int64)

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
        return add_vectors(segment[0],tmp)
    else: # no intersection
        return np.zeros((6,3),dtype=np.int64)

def intersection_two_tetrahedron_4(tetrahedron_1,tetrahedron_2):
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
    
    tmp=np.array([])
    tmp1=np.array([])
    
    array0=np.zeros((6,3),dtype=np.int64)
    
    counter=0
    for c in comb:
        # case 1: intersection between
        segment=np.stack([tetrahedron_1[c[0]],tetrahedron_1[c[1]]])
        surface=np.stack([tetrahedron_2[c[2]],tetrahedron_2[c[3]],tetrahedron_2[c[4]]])
        tmp1=intersection_segment_surface(segment,surface)
        if np.all(tmp1==array0):
            pass
        else:
            #print('tmp1',tmp1)
            if counter==0 :
                tmp=tmp1 # intersection points
            else:
                tmp=np.vstack([tmp,tmp1]) # intersecting points
            counter+=1
            #print('tmp',tmp)
        # case 2: intersection between
        segment=np.stack([tetrahedron_2[c[0]],tetrahedron_2[c[1]]])
        surface=np.stack([tetrahedron_1[c[2]],tetrahedron_1[c[3]],tetrahedron_1[c[4]]])
        tmp1=intersection_segment_surface(segment,surface)
        if np.all(tmp1==array0):
            pass
        else:
            #print('tmp1',tmp1)
            if counter==0:
                tmp=tmp1 # intersection points
            else:
                tmp=np.vstack([tmp,tmp1]) # intersecting points
            counter+=1
            #print('tmp',tmp)
    
    # get vertces of tetrahedron_1 that are inside tetrahedron_2
    for i1 in range(4):
        if inside_outside_tetrahedron_tau(tetrahedron_1[i1],tetrahedron_2): # inside
            #print('tetrahedron_1[i1]',tetrahedron_1[i1])
            if counter==0:
                tmp=tetrahedron_1[i1]
            else:
                tmp=np.vstack([tmp,tetrahedron_1[i1]])
            counter+=1
            #print('tmp',tmp)
    # get vertces of tetrahedron_2 that are inside tetrahedron_1
    for i1 in range(4):
        if inside_outside_tetrahedron_tau(tetrahedron_2[i1],tetrahedron_1): # inside
            #print('tetrahedron_2[i1]',tetrahedron_2[i1])
            if counter==0:
                tmp=tetrahedron_2[i1]
            else:
                tmp=np.vstack([tmp,tetrahedron_2[i1]])
            counter+=1
            #print('tmp',tmp)
        else:
            pass
    tmp=tmp.reshape(int(len(tmp)/6),6,3)
    #print('tmp:',tmp)
    
    tmp4=np.array([[[[0]]]])
    if counter>=4:
        tmp3=remove_doubling_in_perp_space(tmp)
        if len(tmp3)>=4:
            # Tetrahedralization
            if coplanar_check(tmp3): # coplanar
                pass
            else:
                if len(tmp3)==4:
                    tmp4=tmp3.reshape(1,4,6,3)
                else:
                    tmp4=tetrahedralization_points(tmp3)
        else:
            pass
    else:
        pass
    return tmp4

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
            return np.array([[[[0]]]])
    else:
        return np.array([[[[0]]]])

def intersection_two_obj_1(obj1,obj2):
    
    cent2=centroid_obj(obj2)
    dd2=ball_radius_obj(obj2,cent2)
    
    #common4=np.array([[[[0]]]])
    counter0=0
    for tetrahedron1 in obj1:
        if rough_check_intersection_tetrahedron_obj(tetrahedron1,cent2,dd2):
            counter1=0
            tmp_common4=np.array([[[[0]]]])
            vol1=tetrahedron_volume_6d(tetrahedron1)
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
                    if np.all(tmp4==np.array([[[[0]]]])):
                        pass
                    else:
                        #print('tmp4:',tmp4)
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
            if counter1>0:
                #print('tmp_common4',tmp_common4)
                vol2=obj_volume_6d(tmp_common4)
                if np.all(vol1==vol2):
                    if counter0==0:
                        common4=tetrahedron1.reshape(1,4,6,3)
                        counter0+=1
                    else:
                        common4=np.concatenate([common4,tetrahedron1])
                else:
                    if counter0==0:
                        common4=tmp_common4
                        counter0+=1
                    else:
                        common4=np.concatenate([common4,tmp_common4])
            else:
                pass
        else:
            pass
    return common4



########## WIP ##########


def intersection_two_obj_convex(obj1,obj2):
    """
    # This is very simple but work correctly only when each subdivided 
    # three ODs (i.e part part, ODA and ODB) are able to define as a
    # set of tetrahedra.
    """
    print("       start: intersection_two_obj_convex()")
    print("         num. of triangles:",len(obj1)*4)
    print("         num. of triangles:",len(obj2)*4)
    obj1_surf=generator_surface_1(obj1)
    print("       end1")
    obj2_surf=generator_surface_1(obj2)
    print("       end2")
    obj1_edge=generator_unique_edges(obj1_surf)
    print("       end3")
    obj2_edge=generator_unique_edges(obj2_surf)
    print("       end4")
    print("         num. of unique triangles in obj1:",len(obj1_surf))
    print("         num. of unique triangles in obj2:",len(obj2_surf))
    print("         num. of unique deges in obj1:",len(obj1_edge))
    print("         num. of unique deges in obj2:",len(obj2_edge))
    
    counter=0
    for tr1 in obj1_surf:
        for ed2 in obj2_edge:
            if check_intersection_segment_surface_numerical_6d_tau(ed2,tr1): # intersection
                tmp=intersection_segment_surface(ed2,tr1)
                if counter==0:
                    p=tmp
                else:
                    p=np.append(p,tmp)
                counter+=1
            else:
                pass
    print("       end5")
    for tr2 in obj2_surf:
        for ed1 in obj1_edge:
            if check_intersection_segment_surface_numerical_6d_tau(ed1,tr2): # intersection
                tmp=intersection_segment_surface(ed1,tr2)
                if counter==0:
                    p=tmp
                else:
                    p=np.append(p,tmp)
                counter+=1
            else:
                pass
    print("       end6")
    if counter==0:
        return np.array([[[[0]]]])
    else:
        point1=remove_doubling_in_perp_space(p.reshape(int(len(p)/18),6,3))
        #print(' dividing into three PODs:')
        #print('    Common   :     OD1 and     ODB')
        #print('  UnCommon 1 :     OD1 and Not OD2')
        #print('  UnCommon 2 : Not OD1 and     OD2')
        #
        # --------------------------
        # (1) Extract vertces of 2nd OD which are insede 1st OD --> point_a1
        #     Extract vertces of 2nd OD which are outsede 1st OD --> point_b2
        #
        counter1=0
        counter2=0
        tmp3=remove_doubling_in_perp_space(obj1_surf) # generating vertces of 1st OD
        for point_tmp in tmp3:
            counter=0
            for tet in obj2:
                if inside_outside_tetrahedron_tau(point_tmp,tet):
                    counter+=1
                    break
                else:
                    pass
            if counter>0:
                if counter1==0:
                    tmp1a=point_tmp.reshape(18) # 18=6*3
                else:
                    tmp1a=np.append(tmp1a,point_tmp)
                counter1+=1
            else:
                if counter2==0:
                    tmp1b=point_tmp.reshape(18)
                else:
                    tmp1b=np.append(tmp1b,point_tmp)
                counter2+=1
        point_a1=tmp1a.reshape(int(len(tmp1a)/18),6,3)
        point_b2=tmp1b.reshape(int(len(tmp1b)/18),6,3)
        print("       end7")
        #
        # (2) Extract vertces of 1st OD which are insede 2nd OD --> point_b1
        #     Extract vertces of 1st OD which are outsede 2nd OD --> point_a2
        #
        counter1=0
        counter2=0
        tmp3=remove_doubling_in_perp_space(obj2_surf) # generating vertces of 2nd OD
        for i1 in range(len(tmp3)):
            point_tmp=tmp3[i1]
            counter=0
            for tet in obj1:
                if inside_outside_tetrahedron_tau(point_tmp,tet):
                    counter+=1
                    break
                else:
                    pass
            if counter>0:
                if counter1==0:
                    tmp1a=point_tmp.reshape(18) # 18=6*3
                else:
                    tmp1a=np.append(tmp1a,point_tmp)
                counter1+=1
            else:
                if counter2==0:
                    tmp1b=point_tmp.reshape(18)
                else:
                    tmp1b=np.append(tmp1b,point_tmp)
                counter2+=1
        point_b1=tmp1a.reshape(int(len(tmp1a)/18),6,3)
        point_a2=tmp1b.reshape(int(len(tmp1b)/18),6,3)
        print("       end8")
        #
        # (3) Sum point A, point B and Intersections --->>> common part
        #
        # common part = point1 + point_a1 + point_b1
        tmp=np.append(point1,point_a1)
        tmp=np.append(tmp,point_b1)
        tmp=tmp.reshape(int(len(tmp)/18),6,3) # 18=6*3
        point_common=remove_doubling_in_perp_space(tmp)
        
        common=tetrahedralization_points(point_common)
        if common.tolist()!=[[[[0]]]]:
            return common
        else:
            print('no common part')
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
