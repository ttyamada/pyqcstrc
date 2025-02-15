#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
#import sys
import numpy as np
from numpy.typing import NDArray
import time # in object_subtraction_dev1, tetrahedron_not_obj
import itertools
from pyqcstrc.octa2.math1 import (projection3,
                                centroid, 
                                centroid_obj,
                                coplanar_check,
                                det_matrix,
                                dot_product,
                                inner_product,
                                outer_product,
                                add,
                                sub,
                                mul,
                                div,
                                sub_vectors,
                                add_vectors, 
                                mul_vector,
                                )
from pyqcstrc.octa2.numericalc import (numeric_value,
                                    numerical_vector,
                                    length_numerical,
                                    get_internal_component_sets_numerical,
                                    get_internal_component_numerical,
                                    check_intersection_two_segment_numerical_6d_tau,
                                    check_intersection_segment_surface_numerical_6d_tau,
                                    #check_intersection_segment_surface_numerical,
                                    check_intersection_two_segment_numerical,
                                    inside_outside_triangle,
                                    inside_outside_triangle_tau,
                                    on_out_surface,
                                    )
from pyqcstrc.octa2.utils import (remove_doubling_in_perp_space,
                                triangle_area_6d,
                                obj_area_6d,
                                #generator_surface_1,
                                #generator_unique_triangles,
                                generator_unique_edges,
                                triangulation_points,
                                generate_convex_hull,
                                surface_cleaner,
                                )

TAU=np.sqrt(2)
EPS=1e-6

def ball_radius_obj(obj: NDArray[np.int64], centroid: NDArray[np.int64]) -> float:
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

def ball_radius(triangle: NDArray[np.int64], centroid: NDArray[np.int64]) -> float:
    #  this transforms a tetrahedron to a boll which covers the triangle
    #  the centre of the boll is the centroid of the triangle.
    return ball_radius_obj(triangle,centroid)

def distance_in_perp_space(vt1: NDArray[np.int64], vt2: NDArray[np.int64]) -> float:
    a=sub_vectors(vt1,vt2)
    a=projection3(a)
    return length_numerical(a)

def rough_check_intersection_triangle_obj(triangle: NDArray[np.int64], cententer: NDArray[np.int64], distance: float) -> bool:
    cen1=centroid(triangle)
    dd1=ball_radius(triangle,cen1)
    dd0=distance_in_perp_space(cen1,cententer)
    if dd0 <= dd1+distance: # two balls are intersecting.
        return True
    else:
        return False

def check_intersection_two_triangles(triangle_1: NDArray[np.int64], triangle_2: NDArray[np.int64]) -> int:
    # checking whether triangle_1 is fully inside triangle_2 or not
    counter2=0
    for vtx in triangle_1:
        if inside_outside_triangle_tau(vtx,triangle_2): # inside
            pass
        else:
            counter2+=1
            break
    # checking whether triangle_2 is fully inside triangle_1 or not
    counter3=0
    for vtx in triangle_2:
        if inside_outside_triangle_tau(vtx,triangle_1): # inside
            pass
        else:
            counter3+=1
            break
    if counter2==0:
        return 1 # triangle_1 is fully inside triangle_2
    elif counter3==0:
        return 2 # triangle_2 is fully inside triangle_1
    else:
        #
        # -----------------
        # triangle_1
        # -----------------
        # vertex 1: triangle_1[0],  consist of (a1+b1*TAU)/c1, ... (a6+b6*TAU)/c6    a_i,b_i,c_i = tetrahedron_1[0][i:0~5][0],tetrahedron_1[0][i:0~5][1],tetrahedron_1[0][i:0~5][2]
        # vertex 2: triangle_1[1]
        # vertex 3: triangle_1[2]
        #
        # 1 triangle of triangle_1
        # surface 1: v1,v2,v3
        #
        # 3 edges of triangle_1
        # edge 1: v1,v2
        # edge 2: v1,v3
        # edge 3: v2,v3
        #
        # -----------------
        # triangle_2
        # -----------------
        # vertex 1: triangle_2[0]
        # vertex 2: triangle_2[1]
        # vertex 3: triangle_2[2]
        #
        # 1 surfaces of triangle_2
        # surface 1: w1,w2,w3
        #
        # 3 edges of triangle_2
        # edge 1: w1,w2
        # edge 2: w1,w3
        # edge 3: w2,w3
        #
        # case 1: intersection between (edge of triangle_1) and (surface of triangle_2)
        # case 2: intersection between (edge of triangle_2) and (surface of triangle_1)
        #
        # combination_index
        # e.g. v1,v2,w1,w2,w3 (edge 1 and surface 1) ...
        
        #comb=[\
        #[0,1,0,1,2],\
        #[0,2,0,1,2],\
        #[1,2,0,1,2]]
        comb=[\
        [0,1],\
        [0,2],\
        [1,2]]
    
        counter1=0
        for c in comb:
            # case 1: intersection between
            # 3 edges of triangle_1
            # 1 surfaces of triangle_2
            segment=np.stack([triangle_1[c[0]],triangle_1[c[1]]])
            surface=triangle_2
            if check_intersection_segment_surface_numerical_6d_tau(segment,surface): # intersectiing
                counter1+=1
                break
            else:
                pass
            # case 2: intersection between
            # 3 edges of triangle_2
            # 1 surfaces of triangle_1
            segment=np.stack([triangle_2[c[0]],triangle_2[c[1]]])
            surface=triangle_1
            if check_intersection_segment_surface_numerical_6d_tau(segment,surface): # intersectiing
                counter1+=1
                break
            else:
                pass
        if counter1>0:
            return 3 # intersecting
        else:
            return 0 # no intersection

def intersection_two_segment(segment_1: NDArray[np.int64], segment_2: NDArray[np.int64]) -> NDArray[np.int64]:
    """check intersection between two line segments.
    
    Parameters
    ----------
    line_segment_1 line_segment_2: array
        6-dimensional coordinates of line segment,(xyzuvw1, xyzuvw2) and (xyzuvw3, xyzuvw4), in TAU-style
   
    Returns
    -------
    
    """
    # check whether two line segments are intersecting or not by numerical calc.
    if check_intersection_two_segment_numerical_6d_tau(segment_1,segment_2): # intersecting
        # calc in TAU-style
        vecAB_6d=sub_vectors(segment_1[1],segment_1[0])
        #print('vecAB_6d:',vecAB_6d)
        vecAB=projection3(vecAB_6d)               # AB
        #print('vecAB:',vecAB)
        #
        tmp=sub_vectors(segment_2[1],segment_2[0])
        #print('tmp:',tmp)
        vecCD=projection3(tmp)                    # CD
        #print('vecCD:',vecCD)
        
        #
        #tmp=sub_vectors(segment_1[0],segment_2[0])
        #vecCA=projection3(tmp)                    # CA
        #
        tmp=sub_vectors(segment_2[0],segment_1[0])
        #print('tmp:',tmp)
        vecAC=projection3(tmp)                    # AC
        #print('vecAC:',vecAC)
        
        # bunbo=dot_product(vecAB,vecCD)*dot_product(vecCD,vecAB)-dot_product(vecAB,vecAB)*dot_product(vecCD,vecCD)
        tmp1=dot_product(vecAB,vecCD)
        tmp2=dot_product(vecCD,vecAB)
        tmp3=mul(tmp1,tmp2)
        #print('tmp3:',tmp3)
        
        #
        tmp1=dot_product(vecAB,vecAB)
        tmp2=dot_product(vecCD,vecCD)
        tmp4=mul(tmp1,tmp2)
        #print('tmp4:',tmp4)
        bunbo=sub(tmp3,tmp4)
        #print('bunbo:',bunbo)
        if bunbo[0]==0 and bunbo[1]==0:
            return 
        else:
            # bunshi=dot_product(vecAC,vecCD)*dot_product(vecCD,vecAB)-dot_product(vecCD,vecCD)*dot_product(vecAC,vecAB)
            tmp1=dot_product(vecAC,vecCD)
            tmp2=dot_product(vecCD,vecAB)
            tmp3=mul(tmp1,tmp2)
            #print('tmp3:',tmp3)
            #
            tmp1=dot_product(vecCD,vecCD)
            tmp2=dot_product(vecAC,vecAB)
            tmp4=mul(tmp1,tmp2)
            bunshi=sub(tmp3,tmp4)
            #print('bunshi:',bunshi)
            
            # s=bunshi/bunbo
            s=div(bunshi,bunbo)
            sn=(s[0]+s[1]*TAU)/s[2]
            #print('s:',s,sn)
            if sn<=1.0 and sn>=0.0:
            
                # OP = OA + s*AB
                tmp=mul_vector(vecAB_6d,s)
                return add_vectors(segment_1[0],tmp)
            else:
                return 
    else: # no intersection
        return 

def intersection_segment_surface(segment: NDArray[np.int64], surface: NDArray[np.int64]) -> NDArray[np.int64]:
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
        
        """
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
        """
        #  edge: 0-1,0-2,1-2
        comb=[[0,1],[0,2],[1,2]]
        counter=0
        for j in comb:
            segment1=np.vstack([surface[j[0]],surface[j[1]]])
            segment1=segment1.reshape(2,6,3)
            #print('1st segment:')
            #print(segment)
            #print('2nd segment:')
            #print(segment1)
            
            tmp1=intersection_two_segment(segment,segment1)
            #print('tmp1',tmp1)
            if np.all(tmp1==None):
                pass
            else:
                if counter==0:
                    p=tmp1
                else:
                    p=np.vstack([p,tmp1])
                counter+=1
        if counter>0: # intersection
            return p
        else: # no intersection
            return 
    else: # no intersection
        return 
    
def intersection_two_triangles(triangle_1: NDArray[np.int64], triangle_2: NDArray[np.int64]) -> NDArray[np.int64]:
    #
    # -----------------
    # triangle_1
    # -----------------
    # vertex 1: triangle_1[0],  consist of (a1+b1*TAU)/c1, ... (a6+b6*TAU)/c6    a_i,b_i,c_i = tetrahedron_1[0][i:0~5][0],tetrahedron_1[0][i:0~5][1],tetrahedron_1[0][i:0~5][2]
    # vertex 2: triangle_1[1]
    # vertex 3: triangle_1[2]
    #
    # 1 triangle of triangle_1
    # surface 1: v1,v2,v3
    #
    # 3 edges of triangle_1
    # edge 1: v1,v2
    # edge 2: v1,v3
    # edge 3: v2,v3
    #
    # -----------------
    # triangle_2
    # -----------------
    # vertex 1: triangle_2[0]
    # vertex 2: triangle_2[1]
    # vertex 3: triangle_2[2]
    #
    # 1 surfaces of triangle_2
    # surface 1: w1,w2,w3
    #
    # 3 edges of triangle_2
    # edge 1: w1,w2
    # edge 2: w1,w3
    # edge 3: w2,w3
    #
    # case 1: intersection between (edge of triangle_1) and (surface of triangle_2)
    # case 2: intersection between (edge of triangle_2) and (surface of triangle_1)
    #
    # combination_index
    # e.g. v1,v2,w1,w2,w3 (edge 1 and surface 1) ...
    comb=[\
    [0,1,0,1,2],\
    [0,2,0,1,2],\
    [1,2,0,1,2]]
    
    counter=0
    for c in comb:
        # case 1: intersection between (edge of triangle_1) and (surface of triangle_2)
        #print('case 1')
        segment=np.stack([triangle_1[c[0]],triangle_1[c[1]]])
        surface=np.stack([triangle_2[c[2]],triangle_2[c[3]],triangle_2[c[4]]])
        #print('segment:',segment)
        #print('surface:',surface)
        vtx=intersection_segment_surface(segment,surface)
        #print(vtx)
        if np.all(vtx==None):
            pass
        else:
            if counter==0 :
                tmp=vtx # intersection points
            else:
                tmp=np.vstack([tmp,vtx]) # intersecting points
            counter+=1
        # case 2: intersection between (edge of triangle_2) and (surface of triangle_1)
        #print('case 2')
        segment=np.stack([triangle_2[c[0]],triangle_2[c[1]]])
        surface=np.stack([triangle_1[c[2]],triangle_1[c[3]],triangle_1[c[4]]])
        #print('segment:',segment)
        #print('surface:',surface)
        vtx=intersection_segment_surface(segment,surface)
        #print(vtx)
        if np.all(vtx==None):
            pass
        else:
            if counter==0:
                tmp=vtx # intersection points
            else:
                tmp=np.vstack([tmp,vtx]) # intersecting points
            counter+=1
    tmp=tmp.reshape(int(len(tmp)/6),6,3)
    #print('len(tmp):',len(tmp))
    
    # get vertces of triangle_1 that are inside triangle_2
    for vtx in triangle_1:
        if inside_outside_triangle_tau(vtx,triangle_2): # inside
            if counter==0:
                tmp=vtx.reshape(1,6,3)
            else:
                tmp=np.vstack([tmp,[vtx]])
            counter+=1
    # get vertces of triangle_2 that are inside triangle_1
    for vtx in triangle_2:
        if inside_outside_triangle_tau(vtx,triangle_1): # inside
            if counter==0:
                tmp=vtx.reshape(1,6,3)
            else:
                tmp=np.vstack([tmp,[vtx]])
            counter+=1
        else:
            pass
    
    if counter>=3:
        tmp=remove_doubling_in_perp_space(tmp)
        #print('len(tmp):',len(tmp))
        if len(tmp)>3:
            tmp4=triangulation_points(tmp)
            if np.all(tmp4==None):
                return 
            else:
                return tmp4
        elif len(tmp)==3:
            return tmp.reshape(1,3,6,3)
        else:
            return 
    else:
        return 

def intersection_two_obj_1(obj1: NDArray[np.int64],obj2: NDArray[np.int64],select=None,verbose: int=0) -> NDArray[np.int64]:
    """
    Return an intersection between two objects.
    
    Parameters
    ----------
    obj1 : ndarray
        a set of triangles to be intersected with obj2.
    obj2 : ndarray
        a set of triangles to be intersected with obj1.
    select : {'standard', 'simple'}, optional
        The default is 'standard'. 
    
    Returns
    -------
    intersection between obj1 and obj2 : ndarray
        Array of the same type and shape as `obj1` and `obj2`.
    
    Notes
    -----
    
    'standard' intersection is default.
    
    Output from 'simple' intersection is simpler but may cause a problem when generating its surface triangles.
    
    """
    
    if verbose>0:
        print("       start: intersection_two_obj_1()")
    
    cent2=centroid_obj(obj2)
    dd2=ball_radius_obj(obj2,cent2)
    
    if verbose>0:
        print("         dd2:%6.4f"%(dd2))
    
    counter0=0
    for i1,triangle1 in enumerate(obj1):
        if verbose>0:
            print("         %d-th triangle in obj1"%(i1))
        if rough_check_intersection_triangle_obj(triangle1,cent2,dd2):
            if verbose>0:
                print("          Rough_check:True")
            counter1=0
            for i2,triangle2 in enumerate(obj2):
                flag=check_intersection_two_triangles(triangle1,triangle2)
                if verbose>0:
                    print("          %d-th triangle in obj2, flag:%d"%(i2,flag))
                #
                # triangle1 is fully inside triangle2
                if flag==1:
                    if counter0==0:
                        common4=triangle1.reshape(1,3,6,3)
                        counter0+=1
                    else:
                        common4=np.vstack([common4,[triangle1]])
                    break
                #
                # triangle2 is fully inside triangle1
                elif flag==2:
                    if counter1==0:
                        tmp_common4=triangle2.reshape(1,3,6,3)
                        counter1+=1
                    else:
                        tmp_common4=np.vstack([tmp_common4,[triangle2]])
                #
                # triangle1 and triangle2 are intersecting
                elif flag==3:
                    tmp4=intersection_two_triangles(triangle1,triangle2)
                    ###
                    ### Comment:
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
                #vol2=obj_area_6d(tmp_common4)
                #print('vol2',vol2,numeric_value(vol2))
                if select=='simple':
                    vol1=triangle_area_6d(triangle1)
                    vol2=obj_area_6d(tmp_common4)
                    if np.all(vol1==vol2):
                        if counter0==0:
                            common4=triangle1.reshape(1,3,6,3)
                            #print('common4.shape',common4.shape)
                            counter0+=1
                        else:
                            #common4=np.concatenate([common4,tetrahedron1])
                            common4=np.vstack([common4,[triangle1]])
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
                    #print('tmp_common4.shape:',tmp_common4.shape)
                    if counter0==0:
                        common4=tmp_common4
                        counter0+=1
                        #print('tmp_common4.shape',tmp_common4.shape)
                    else:
                        #print('common4.shape:',common4.shape)
                        #common4=np.concatenate([common4,tmp_common4])
                        common4=np.vstack([common4,tmp_common4])
                        #print('tmp_common4.shape',tmp_common4.shape)
                #print(common4.shape)
                #return common4
        else:
            if verbose>0:
                print("          Rough_check:False")
            pass
            
    if counter0>0:
        return common4
    else:
        return 

def intersection_two_obj_convex(obj1: NDArray[np.int64], obj2: NDArray[np.int64], verbose: int=0) -> NDArray[np.int64]:
    """
    Return an intersection between two objects.
    
    Parameters
    ----------
    obj1 : ndarray
        a set of triangles to be intersected with obj2.
    obj2 : ndarray
        a set of triangles to be intersected with obj1.
    kind : {'standard', 'simple'}, optional
        The default is 'standard'. 
    
    Returns
    -------
    intersection between obj1 and obj2 : ndarray
        Array of the same type and shape as `obj1` and `obj2`.
    
    Notes
    -----
    
    Both obj1 and obj2 have to be convex hull.
    """
    if verbose>0:
        print("       start: intersection_two_obj_convex()")
    
    
    obj1_surf=obj1
    obj2_surf=obj2
    #obj1_edge=generator_unique_edges(obj1_surf)
    #obj2_edge=generator_unique_edges(obj2_surf)
    obj1_edge=surface_cleaner(obj1)
    obj2_edge=surface_cleaner(obj2)
    
    if verbose>1:
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
        for triangle2 in obj2:
            if inside_outside_triangle_tau(vrtx,triangle2):
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
        for triangle1 in obj1:
            if inside_outside_triangle_tau(vrtx,triangle1):
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
                    if np.all(tmp!=None):
                        if counter==0:
                            p=tmp
                        else:
                            p=np.vstack([p,tmp])
                        counter+=1
                    else:
                        pass
                else:
                    pass
        for tr2 in obj2_surf:
            for ed1 in obj1_edge:
                if check_intersection_segment_surface_numerical_6d_tau(ed1,tr2): # intersection
                    tmp=intersection_segment_surface(ed1,tr2)
                    if np.all(tmp!=None):
                        if counter==0:
                            p=tmp
                        else:
                            p=np.vstack([p,tmp])
                        counter+=1
                    else:
                        pass
                else:
                    pass
        if counter==0:
            return 
        else:
            point1=remove_doubling_in_perp_space(p.reshape(int(len(p)/6),6,3))
            #
            # (3) Sum point A, point B and Intersections --->>> common part
            #
            # common part = point1 + point_a1 + point_b1
            points=np.vstack([point1,point_a1,point_b1])
            points=remove_doubling_in_perp_space(points)
            common=triangulation_points(points)
            if np.all(common==None):
                if verbose>0:
                    print('no common part')
                return 
            else:
                return common

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
    
    def generate_random_triangle():
        return generate_random_vectors(3)
    
    segment_1=np.array(\
    [[[1, 0, 1],\
      [1, 0, 1],\
      [0, 0, 1],\
      [0, 0, 1],\
      [0, 0, 1],\
      [0, 0, 1]],\
     [[3, 0, 2],\
      [1, 0, 2],\
      [0, 0, 1],\
      [0, 0, 1],\
      [0, 0, 1],\
      [0, 0, 1]]])
      
    segment_2=np.array(\
    [[[ 0,  0,  1],\
      [ 0,  0,  1],\
      [ 0,  0,  1],\
      [ 1,  0,  1],\
      [ 0,  0,  1],\
      [ 0,  0,  1]],\
     [[ 0,  0,  1],\
      [ 0,  0,  1],\
      [-1,  0,  2],\
      [ 1,  0,  2],\
      [ 0,  0,  1],\
      [ 0,  0,  1]]])
    
    a=check_intersection_two_segment_numerical_6d_tau(segment_1,segment_2)
    print(a)
    a=intersection_two_segment(segment_1, segment_2) 
    print(a)
    
    
    
    print('TEST1')
    segment_1=np.array(\
    [[[1, 0, 1],\
      [1, 0, 1],\
      [0, 0, 1],\
      [0, 0, 1],\
      [0, 0, 1],\
      [0, 0, 1]],\
     [[1, 0, 2],\
      [3, 0, 2],\
      [0, 0, 1],\
      [0, 0, 1],\
      [0, 0, 1],\
      [0, 0, 1]]])
    segment_2=np.array(\
    [[[ 0,  0,  1],\
      [ 1,  0,  1],\
      [ 0,  0,  1],\
      [ 0,  0,  1],\
      [ 0,  0,  1],\
      [ 0,  0,  1]],\
     [[ 0,  0,  1],\
      [ 1,  0,  2],\
      [-1,  0,  2],\
      [ 0,  0,  1],\
      [ 0,  0,  1],\
      [ 0,  0,  1]]])
    a=check_intersection_two_segment_numerical_6d_tau(segment_1,segment_2)
    print(a)
    a=intersection_two_segment(segment_1, segment_2) 
    print(a)
    
    """
    s: [ 2 -1  1] 0.5857864376269049
    tmp1 [[ 0  1  2]
     [ 4 -1  2]
     [ 0  0  1]
     [ 0  0  1]
     [ 0  0  1]
     [ 0  0  1]]
    """