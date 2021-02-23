#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
from scipy.spatial import Delaunay
cimport numpy as np
cimport cython

from pyqcstrc.icosah.math1 cimport centroid, coplanar_check, projection, det_matrix, dot_product, add, sub, mul, div
from pyqcstrc.icosah.numericalc cimport obj_volume_6d_numerical, tetrahedron_volume_6d_numerical, inside_outside_tetrahedron
from pyqcstrc.icosah.utils cimport obj_volume_6d, tetrahedron_volume_6d, remove_doubling_dim3_in_perp_space, remove_doubling_dim4_in_perp_space, generator_edge, generator_surface_1

DTYPE_double = np.float64
DTYPE_int = int

cdef np.float64_t TAU=(1+np.sqrt(5))/2.0

cdef np.ndarray intersection_line_segment_triangle(np.ndarray[DTYPE_int_t, ndim=3] line_segment,
                                                    np.ndarray[DTYPE_int_t, ndim=3] triangle,
                                                    int verbose):
    cdef int i,j,counter,num1,num2,num3
    cdef np.ndarray[DTYPE_int_t,ndim=1] p,tmp,tmp1
    #
    # -----------------
    # triangle
    # -----------------
    # vertex 1: triangle[0],  consist of (a1+b1*TAU)/c1, ... (a6+b6*TAU)/c6    a_i,b_i,c_i = tetrahedron_1[0][i:0~5][0],triangle[0][i:0~5][1],triangle[0][i:0~5][2]
    # vertex 2: triangle[1]
    # vertex 3: triangle[2]
    #
    # triangle
    # triangle 1: v1,v2,v3
    #
    # -----------------
    # line_segment
    # -----------------
    # line_segment[i][0][0],line_segment[i][0][1],line_segment[i][0][2],line_segment[i][0][3],line_segment[i][0][4],line_segment[i][0][5]
    # line_segment[i][1][0],line_segment[i][1][1],line_segment[i][1][2],line_segment[i][1][3],line_segment[i][1][4],line_segment[i][1][5]
    #
    #
    # intersection between (edge) and (triangle_1)
    #
    # combination_index
    # e.g. v1,v2,w1,w2,w3 (edge 1 and triangle 1) ...
    #
    # edges
    # triangle_1
    segment_1=line_segment[0]
    segment_2=line_segment[1]
    surface_1=triangle[0]
    surface_2=triangle[1]
    surface_3=triangle[2]
    p=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3,verbose)
    return p

cpdef list intersection_two_obj_convex(np.ndarray[DTYPE_int_t, ndim=4] obj1,
                                        np.ndarray[DTYPE_int_t, ndim=4] obj2,
                                        int verbose):
    cdef np.ndarray[DTYPE_int_t, ndim=4] obj1_surf,obj2_surf,obj1_edge,obj2_edge
    
    
    # This is very simple but work correctly only when each subdivided 
    # three ODs (i.e part part, ODA and ODB) are able to define as a
    # set of tetrahedra.
    cdef int i1,i2,i3,counter,counter1,counter2,num1,num2
    cdef np.ndarray[DTYPE_int_t,ndim=1] p,tmp,tmp1,tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t,ndim=2] point_tmp,point1_tmp
    cdef np.ndarray[DTYPE_int_t,ndim=3] tet
    cdef np.ndarray[DTYPE_int_t,ndim=3] tr1,tr2,ed1,ed2,tmp3
    cdef np.ndarray[DTYPE_int_t,ndim=3] point_a,point_a1,point_a2
    cdef np.ndarray[DTYPE_int_t,ndim=3] point_b,point_b1,point_b2
    cdef np.ndarray[DTYPE_int_t,ndim=3] point,point1,point_common
    cdef np.ndarray[DTYPE_int_t,ndim=4] obj_common,obj_1,obj_2
    cdef list comb
    
    if verbose>0:
        print('     intersection_two_obj_convex')
    else:
        pass
    
    obj1_surf=generator_surface_1(obj1,verbose-1)
    obj2_surf=generator_surface_1(obj2,verbose-1)
    obj1_edge=generator_edge(obj1_surf,verbose-1)
    obj2_edge=generator_edge(obj2_surf,verbose-1)
    
    counter=0
    for i1 in range(len(obj1_surf)):
        tr1=obj1_surf[i1]
        for i2 in range(len(obj2_edge)):
            ed2=obj2_edge[i2]
            tmp=intersection_line_segment_triangle(ed2,tr1,verbose-2)
            if len(tmp)!=1:
                if counter==0:
                    p=tmp
                else:
                    p=np.append(p,tmp)
                counter+=1
            else:
                pass
    for i1 in range(len(obj2_surf)):
        tr2=obj2_surf[i1]
        for i2 in range(len(obj1_edge)):
            ed1=obj1_edge[i2]
            tmp=intersection_line_segment_triangle(ed1,tr2,verbose-2)
            if len(tmp)!=1:
                if counter==0:
                    p=tmp
                else:
                    p=np.append(p,tmp)
                counter+=1
            else:
                pass
    if counter==0:
        return np.array([[[[0]]]])
    else:
        point=p.reshape(int(len(p)/18),6,3) # 18=6*3
        point1=remove_doubling_dim3_in_perp_space(point)
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
        tmp3=remove_doubling_dim4_in_perp_space(obj1_surf) # generating vertces of 1st OD
        for i1 in range(len(tmp3)):
            point_tmp=tmp3[i1]
            counter=0
            for i2 in range(len(obj2)):
                tet=obj2[i2]
                num1=inside_outside_tetrahedron(point_tmp,tet[0],tet[1],tet[2],tet[3])
                if num1==0:
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
        #
        # (2) Extract vertces of 1st OD which are insede 2nd OD --> point_b1
        #     Extract vertces of 1st OD which are outsede 2nd OD --> point_a2
        #
        counter1=0
        counter2=0
        tmp3=remove_doubling_dim4_in_perp_space(obj2_surf) # generating vertces of 2nd OD
        for i1 in range(len(tmp3)):
            point_tmp=tmp3[i1]
            counter=0
            for i2 in range(len(obj1)):
                tet=obj1[i2]
                num1=inside_outside_tetrahedron(point_tmp,tet[0],tet[1],tet[2],tet[3])
                if num1==0:
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
        #
        # (3) Sum point A, point B and Intersections --->>> common part
        #
        # common part = point1 + point_a1 + point_b1
        tmp=np.append(point1,point_a1)
        tmp=np.append(tmp,point_b1)
        point_common=tmp.reshape(int(len(tmp)/18),6,3) # 18=6*3
        point_common=remove_doubling_dim3_in_perp_space(point_common)
        #
        # point_a = point_a1 + point_a2 + point1
        tmp=np.append(point1,point_a1)
        tmp=np.append(tmp,point_a2)
        point_a=tmp.reshape(int(len(tmp)/18),6,3)
        point_a=remove_doubling_dim3_in_perp_space(point_a)
        #
        # point_b = point_b1 + point_b2 + point1
        tmp=np.append(point1,point_b1)
        tmp=np.append(tmp,point_b2)
        point_b=tmp.reshape(int(len(tmp)/18),6,3)
        point_b=remove_doubling_dim3_in_perp_space(point_b)
        #
        return [point_common,point_a,point_b]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray intersection_two_obj(np.ndarray[DTYPE_int_t, ndim=4] obj1,
                                    np.ndarray[DTYPE_int_t, ndim=4] obj2,
                                    int flag1,
                                    int flag2,
                                    int flag3,
                                    int verbose):
    cdef int i1,counter
    cdef double dd2
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=2] cent2
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,tmp4_common
    """
    flag1 = 0, rough check intersection of obj1 and obj2; flag1 = 1, no
    flag2 = 0, rough check intersection of tetrahedron in obj1 and obj2; flag2 = 1, no
    flag3 = 0, rough check intersection of tetrahedron in obj1 and tetrahedron in obj2; flag3 = 1, no
    """
    if verbose>0:
        print('     intersection_two_obj()')
    else:
        pass
    
    tmp4_common=np.array([[[[0]]]])
    if flag1==0: # filtering ON, rough check intersection of obj1 and obj2
        if rough_check_intersection_two_obj(obj1,obj2)==1:
            if verbose>0:
                print('      rough intersection check, no intersection, skip')
            else:
                pass
            return tmp4_common
        else:
            if flag2==0: # filtering ON, rough check intersection of tetrahedron in obj1 and obj2
                cent2=centroid_obj(obj2)
                dd2=ball_radius_obj(obj2,cent2)
                counter=0
                for i1 in range(len(obj1)):
                    if verbose>0:
                        print('      %d-th tetrahedron in obj1'%(i1))
                    else:
                        pass
                    if rough_check_intersection_tetrahedron_obj(obj1[i1],cent2,dd2)==0:
                        tmp4a=intersection_tetrahedron_obj_4(obj1[i1],obj2,flag3,verbose)
                        #if len(tmp4a)!=1:
                        if tmp4a.tolist()!=[[[[0]]]]:
                            if counter==0:
                                 tmp1a=tmp4a.reshape(len(tmp4a)*72)
                            else:
                                tmp1a=np.append(tmp1a,tmp4a)
                            counter+=1
                        else:
                            pass
                    else:
                        if verbose>0:
                            print('      rough intersection check, no intersection, skip')
                        else:
                            pass
                if counter>0:
                    tmp4_common=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
                else:
                    pass
                return tmp4_common
            else:
                counter=0
                for i1 in range(len(obj1)):
                    tmp4a=intersection_tetrahedron_obj_4(obj1[i1],obj2,flag3,verbose)
                    #if len(tmp4a)!=1:
                    if tmp4a.tolist()!=[[[[0]]]]:
                        if counter==0:
                            tmp1a=tmp4a.reshape(len(tmp4a)*72)
                        else:
                            tmp1a=np.append(tmp1a,tmp4a)
                        counter+=1
                    else:
                        pass
                if counter>0:
                    tmp4_common=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
                else:
                    pass
                return tmp4_common
    else:
        counter=0
        for i1 in range(len(obj1)):
            tmp4a=intersection_tetrahedron_obj_4(obj1[i1],obj2,flag3,verbose)
            #if len(tmp4a)!=1:
            if tmp4a.tolist()!=[[[[0]]]]:
                if counter==0:
                    tmp1a=tmp4a.reshape(len(tmp4a)*72)
                else:
                    tmp1a=np.append(tmp1a,tmp4a)
                counter+=1
            else:
                pass
        if counter>0:
            tmp4_common=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
        else:
            pass
        return tmp4_common
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int rough_check_intersection_two_tetrahedron(np.ndarray[DTYPE_int_t, ndim=2] centroid1,
                                                    double dd1,
                                                    np.ndarray[DTYPE_int_t, ndim=3] tetrahedron2):
    cdef double dd2
    cdef np.ndarray[DTYPE_int_t,ndim=2] cent2
    
    cent2=centroid(tetrahedron2)
    dd2=ball_radius(tetrahedron2,cent2)
    dd0=distance_in_perp_space(centroid1,cent2)
    if dd0<=dd1+dd2: # two balls are intersecting.
        return 0
    else: #
        return 1
"""
cdef np.ndarray intersection_two_tetrahedron(np.ndarray[DTYPE_int_t, ndim=2] centroid1,\
                                            double dd1,\
                                            np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_2,\
                                            int flag3,\
                                            int verbose):
    if flag3==0:
        if rough_check_intersection_two_tetrahedron(centroid1,dd1,tetrahedron_2)==0:
            return intersection_two_tetrahedron_4(tetrahedron_1,tetrahedron_2,verbose)
        else:
            return np.array([[[[0]]]])
    else:
        return intersection_two_tetrahedron_4(tetrahedron_1,tetrahedron_2,verbose)
"""
"""
cdef np.ndarray intersection_tetrahedron_obj(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron,\
                                            np.ndarray[DTYPE_int_t, ndim=2] cent2,\
                                            double dd2
                                            int flag3,\
                                            int verbose):
    if verbose>0:
        print(' intersection_tetrahedron_obj()')
    else:
        pass
    if rough_check_intersection_tetrahedron_obj(tetrahedron,cent2,dd2)==0:
        return intersection_tetrahedron_obj_4(tetrahedron,obj,flag3,verbose)
    else:
        return intersection_tetrahedron_obj_4(tetrahedron,obj,flag3,verbose)
"""
"""
cdef int rough_check_intersection_tetrahedron_obj(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron,\
                                                     np.ndarray[DTYPE_int_t, ndim=4] obj):
    cdef double dd1,dd2
    cdef np.ndarray[DTYPE_int_t,ndim=2] cent1,cent2
    
    cent1=centroid(tetrahedron)
    cent2=centroid_obj(obj)
    dd1=ball_radius(tetrahedron,cent1)
    dd2=ball_radius_obj(obj,cent2)
    dd0=distance_in_perp_space(cent1,cent2)
    if dd0<dd1+dd2: # two balls are intersecting.
        return 0
    else: #
        return 1
"""

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int rough_check_intersection_tetrahedron_obj(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron,
                                                     np.ndarray[DTYPE_int_t,ndim=2] cent2,
                                                     double dd2):
    cdef double dd1
    cdef np.ndarray[DTYPE_int_t,ndim=2] cent1
    
    cent1=centroid(tetrahedron)
    dd1=ball_radius(tetrahedron,cent1)
    dd0=distance_in_perp_space(cent1,cent2)
    if dd0 <= dd1+dd2: # two balls are intersecting.
        return 0
    else: #
        return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int rough_check_intersection_two_obj(np.ndarray[DTYPE_int_t, ndim=4] obj1,
                                             np.ndarray[DTYPE_int_t, ndim=4] obj2):
    cdef double dd1,dd2
    cdef np.ndarray[DTYPE_int_t,ndim=2] cent1,cent2
    
    cent1=centroid_obj(obj1)
    cent2=centroid_obj(obj2)
    dd1=ball_radius_obj(obj1,cent1)
    dd2=ball_radius_obj(obj2,cent2)
    dd0=distance_in_perp_space(cent1,cent2)
    if dd0 <= dd1+dd2: # two balls are intersecting.
        return 0
    else: #
        return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray intersection_using_tetrahedron_4(np.ndarray[DTYPE_int_t, ndim=4] obj1,
                                                    np.ndarray[DTYPE_int_t, ndim=4] obj2,
                                                    int verbose,
                                                    int dummy):
    #
    # Intersection; obj1 and obj2
    #
    cdef int i1,i2,i3,i4,counter1,counter2,counter3,counter4,counter5,q1,q2,q3,flag,num,flag1,flag2
    cdef long v1a,v1b,v1c,v2a,v2b,v2c,v3a,v3b,v3c
    cdef double vol1,vol2,vol3,vol4,vol5
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1b,tmp1c
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4_common,tmp4a,tmp4b
    
    if verbose>0:
        print('      intersection_using_tetrahedron_4()')
    else:
        pass
    
    v1a,v1b,v1c=obj_volume_6d(obj1)
    vol2=obj_volume_6d_numerical(obj1)
    if verbose>1:
        print('      obj1, volume = %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol2))
        print('                   = ',obj_volume_6d_numerical(obj1))
    else:
        pass    
    v1a,v1b,v1c=obj_volume_6d(obj2)
    vol2=obj_volume_6d_numerical(obj2)
    if verbose>1:
        print('      obj2, volume = %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol2))
    else:
        pass

    counter3=0
    tmp1b=np.array([0])
    tmp1c=np.array([0])
    tmp4_common=np.array([[[[0]]]])
    for i1 in range(len(obj1)):
        # volume check
        v1a,v1b,v1c=tetrahedron_volume_6d(obj1[i1])
        vol1=tetrahedron_volume_6d_numerical(obj1[i1])
        if verbose>1:
            print('      %2d-th tetrahedron in obj1, %d %d %d (%10.8f)'%(i1,v1a,v1b,v1c,vol1))
        else:
            pass
        counter1=0
        for i2 in range(len(obj2)):
            if verbose>1:
                v2a,v2b,v2c=tetrahedron_volume_6d(obj2[i2])
                vol2=tetrahedron_volume_6d_numerical(obj2[i2])
                print('      %2d-th tetrahedron in obj2, %d %d %d (%10.8f)'%(i2,v2a,v2b,v2c,vol2))
            else:
                pass
            tmp4_common=intersection_two_tetrahedron_4(obj1[i1],obj2[i2],verbose-1)
            
            if tmp4_common.tolist()!=[[[[0]]]]: # if common part is not empty
                if counter1==0:
                    tmp1c=tmp4_common.reshape(72*len(tmp4_common)) # 72=4*6*3
                else:
                    tmp1c=np.append(tmp1c,tmp4_common)
                counter1+=1
            else:
                pass
            
        if counter1!=0:
            tmp4a=tmp1c.reshape(int(len(tmp1c)/72),4,6,3)
            #if option==0:
                ####################
                ## simplification ##
                ####################
                #tmp4a=simplification_convex_polyhedron(tmp4a,2,verbose-1)
                #pass
            #else:
                #pass
            # volume check
            v3a,v3b,v3c=obj_volume_6d(tmp4a)
            vol3=(v3a+v3b*TAU)/(v3c)
            if v3a==v1a and v3b==v1b and v3c==v1c: # vol1==vol3 ＃ この場合、obj1[i1]全体が、obj2に含まれている
                if verbose>1:
                    print('                 common part (all_1), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol3))
                else:
                    pass
                if counter3==0:
                    tmp1b=obj1[i1].reshape(72)
                else:
                    tmp1b=np.append(tmp1b,obj1[i1])
                counter3+=1
            else: # to avoid overflow
                vol4=obj_volume_6d_numerical(tmp4a)
                if abs(vol1-vol4)<=1e-8:
                    if verbose>1:
                        print('                 common part (all_2), %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol4))
                    else:
                        pass
                    if counter3==0:
                        tmp1b=obj1[i1].reshape(72)
                    else:
                        tmp1b=np.append(tmp1b,obj1[i1])
                    counter3+=1
                elif abs(vol1-vol4)>1e-8 and vol4>=0.0:
                    if abs(vol4-vol3)<1e-8:
                        if verbose>1:
                            print('                 common part (partial_1), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4))
                        else:
                            pass
                        if counter3==0:
                            tmp1b=tmp4a.reshape(len(tmp4a)*72)
                        else:
                            tmp1b=np.append(tmp1b,tmp4a)
                        counter3+=1
                    else:
                        tmp4b=tmp4a
                        vol4=obj_volume_6d_numerical(tmp4b)
                        v3a,v3b,v3c=obj_volume_6d(tmp4b)
                        vol3=(v3a+v3b*TAU)/(v3c)
                        if abs(vol4-vol3)<1e-8:
                            if verbose>1:
                                print('                 common part (partial_2), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4))
                            else:
                                pass
                            if counter3==0:
                                tmp1b=tmp4b.reshape(len(tmp4b)*72)
                            else:
                                tmp1b=np.append(tmp1b,tmp4b)
                            counter3+=1
                        else:
                            if verbose>1:
                                print('                 common part (partial_3), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4))
                            else:
                                pass
                            if counter3==0:
                                tmp1b=tmp4a.reshape(len(tmp4a)*72)
                            else:
                                tmp1b=np.append(tmp1b,tmp4a)
                            counter3+=1
                else:
                    if verbose>1:
                        print('                 common part, %d %d %d (%10.8f) out of expected!'%(v3a,v3b,v3c,vol3))
                        print('                 numerical value, %10.8f'%(vol4))
                    else:
                        pass
                    pass
        else: # if common part (obj1_reduced and obj2) is NOT empty
            if verbose>1:
                print('                 common part, empty')
            else:
                pass
            pass

    if counter3!=0:
        tmp4_common=tmp1b.reshape(int(len(tmp1b)/72),4,6,3)
        v1a,v1b,v1c=obj_volume_6d(tmp4_common)
        vol2=obj_volume_6d_numerical(tmp4_common)
        if verbose>1:
            print('      common obj, volume = %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol2))
        else:
            pass
        return tmp4_common
    else:
        if tmp4_common.tolist()!=[[[[0]]]]:
            return tmp4_common
        else:
            return np.array([[[[0]]]])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray intersection_tetrahedron_obj_4(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron,
                                                np.ndarray[DTYPE_int_t, ndim=4] obj,
                                                int flag3,
                                                int verbose):
    #
    # Intersection; tetrahedron and obj
    #
    cdef int i1,i2,i3,i4,counter1,counter2,counter3,counter4,counter5,q1,q2,q3,flag,num,flag1,flag2
    cdef long v1a,v1b,v1c,v2a,v2b,v2c,v3a,v3b,v3c
    cdef double vol1,vol2,vol3,vol4,vol5
    cdef double dd1
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1b,tmp1c
    cdef np.ndarray[DTYPE_int_t,ndim=2] cent1
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4_common,tmp4a,tmp4b
    
    if verbose>0:
        print('      intersection_tetrahedron_obj_4()')
    else:
        pass

    tmp1c=np.array([0])
    tmp4_common=np.array([[[[0]]]])
    
    # volume check
    v1a,v1b,v1c=tetrahedron_volume_6d(tetrahedron)
    vol1=tetrahedron_volume_6d_numerical(tetrahedron)
    if verbose>0:
        print('       tetrahedron, %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol1))
    else:
        pass
    
    if flag3==0:
        cent1=centroid(tetrahedron)
        dd1=ball_radius(tetrahedron,cent1)
    else:
        pass
    
    counter1=0
    for i2 in range(len(obj)):
        if verbose>1:
            v2a,v2b,v2c=tetrahedron_volume_6d(obj[i2])
            vol2=tetrahedron_volume_6d_numerical(obj[i2])
            print('       %2d-th tetrahedron in obj2, %d %d %d (%10.8f)'%(i2,v2a,v2b,v2c,vol2))
        else:
            pass
        if flag3==0: # filter ON
            if rough_check_intersection_two_tetrahedron(cent1,dd1,obj[i2])==0:
                tmp4_common=intersection_two_tetrahedron_4(tetrahedron,obj[i2],verbose-1)
            else:
                tmp4_common=np.array([[[[0]]]])
        else: # filter OFF
            tmp4_common=intersection_two_tetrahedron_4(tetrahedron,obj[i2],verbose-1)
        
        if tmp4_common.tolist()!=[[[[0]]]]: # if common part is not empty
            if verbose>1:
                v2a,v2b,v2c=obj_volume_6d(tmp4_common)
                vol2=obj_volume_6d_numerical(tmp4_common)
                print('        common obj, %d %d %d (%10.8f)'%(v2a,v2b,v2c,vol2))
            else:
                pass
            if counter1==0:
                tmp1c=tmp4_common.reshape(72*len(tmp4_common)) # 72=4*6*3
            else:
                tmp1c=np.append(tmp1c,tmp4_common)
            counter1+=1
        else:
            pass
    
    if counter1!=0:
        tmp4a=tmp1c.reshape(int(len(tmp1c)/72),4,6,3)
        # volume check
        v3a,v3b,v3c=obj_volume_6d(tmp4a)
        vol3=(v3a+v3b*TAU)/v3c
        if v3a==v1a and v3b==v1b and v3c==v1c: # vol1==vol3 ＃ この場合、tetrahedron全体が、objに含まれている
            if verbose>0:
                print('        common part (all_1), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol3))
            else:
                pass
            #tmp1b=tetrahedron.reshape(72)
            tmp4_common=tetrahedron.reshape(1,4,6,3)
        else:
            if vol3<vol1:
                if verbose>0:
                    print('        common part (partial_1), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol3))
                else:
                    pass
                #tmp1b=tmp4a.reshape(len(tmp4a)*72)
                tmp4_common=tmp4a
            else:
                if verbose>0:
                    print('        common part, %d %d %d (%10.8f) out of expected!'%(v3a,v3b,v3c,vol3))
                else:
                    pass
                #tmp1b=np.array([0])
                tmp4_common=np.array([[[[0]]]])
        if tmp4_common.tolist()!=[[[[0]]]]:
            #tmp4_common=tmp1b.reshape(int(len(tmp1b)/72),4,6,3)
            if verbose>0:
                v1a,v1b,v1c=obj_volume_6d(tmp4_common)
                vol2=obj_volume_6d_numerical(tmp4_common)
                print('        common part, volume = %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol2))
            else:
                pass
            return tmp4_common
        else:
            if verbose>0:
                print('        common part, empty')
            else:
                pass
            return np.array([[[[0]]]])
        
    else: # if common part (obj1_reduced and obj2) is NOT empty
        if verbose>0:
            print('        common part, empty')
        else:
            pass
        return np.array([[[[0]]]])


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray intersection_two_tetrahedron_4(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_1,
                                                np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_2,
                                                int verbose):
    cdef int i1,i2,counter1,counter2,num1,num2
    cdef long a1,b1,c1,a2,b2,c2
    cdef float vol1
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c,v1,v2,v3,v4,v5,v6
    cdef np.ndarray[DTYPE_int_t,ndim=2] comb
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a,tmp2b,tmp2c
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    
    if verbose>0:
        print('        intersection_two_tetrahedron_4()')
    else:
        pass
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
    comb=np.array([\
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
    [2,3,1,2,3]])
    
    tmp1a=np.array([0])
    tmp1b=np.array([0])
    tmp1c=np.array([0])

    counter1=0
    #for i1 in range(len(comb)): # len(combination_index) = 24
    for i1 in range(24):
        # case 1: intersection between
        # 6 edges of tetrahedron_1
        # 4 surfaces of tetrahedron_2
        segment_1=tetrahedron_1[comb[i1][0]] 
        segment_2=tetrahedron_1[comb[i1][1]]
        surface_1=tetrahedron_2[comb[i1][2]]
        surface_2=tetrahedron_2[comb[i1][3]]
        surface_3=tetrahedron_2[comb[i1][4]]
        tmp1c=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3,verbose-1)
        if len(tmp1c)!=1:
            if counter1==0 :
                tmp1a=tmp1c # intersection
            else:
                tmp1a=np.append(tmp1a,tmp1c) # intersecting points
            counter1+=1
        else:
            pass
        # case 2: intersection between
        # 6 edges of tetrahedron_2
        # 4 surfaces of tetrahedron_1
        segment_1=tetrahedron_2[comb[i1][0]]
        segment_2=tetrahedron_2[comb[i1][1]]
        surface_1=tetrahedron_1[comb[i1][2]]
        surface_2=tetrahedron_1[comb[i1][3]]
        surface_3=tetrahedron_1[comb[i1][4]]
        tmp1c=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3,verbose-1)
        if len(tmp1c)!=1:
            if counter1==0:
                tmp1a=tmp1c # intersection
            else:
                tmp1a=np.append(tmp1a,tmp1c) # intersecting points
            counter1+=1
        else:
            pass

    # get vertces of tetrahedron_1 which are inside tetrahedron_2
    #for i1 in range(len(tetrahedron_1)):
    for i1 in range(4):
        flag=inside_outside_tetrahedron(tetrahedron_1[i1],tetrahedron_2[0],tetrahedron_2[1],tetrahedron_2[2],tetrahedron_2[3])
        if flag==0:
            if counter1==0:
                tmp1a=tetrahedron_1[i1].reshape(18)
            else:
                tmp1a=np.append(tmp1a,tetrahedron_1[i1])
            counter1+=1
        else:
            pass
    # get vertces of tetrahedron_2 which are inside tetrahedron_1
    #for i1 in range(len(tetrahedron_2)):
    for i1 in range(4):
        flag=inside_outside_tetrahedron(tetrahedron_2[i1],tetrahedron_1[0],tetrahedron_1[1],tetrahedron_1[2],tetrahedron_1[3])
        if flag==0:
            if counter1==0:
                tmp1a=tetrahedron_2[i1].reshape(18)
            else:
                tmp1a=np.append(tmp1a,tetrahedron_2[i1])
            counter1+=1
        else:
            pass
    
    tmp4a=np.array([[[[0]]]])
    if counter1>=4:
        # remove doubling
        tmp3a=tmp1a.reshape(int(len(tmp1a)/18),6,3)
        tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
        
        if len(tmp3a)>=4:
            if verbose>0:
                print('         number of points for tetrahedralization, %d'%(len(tmp3a)))
            else:
                pass
            # Tetrahedralization
            if coplanar_check(tmp3a)==0:
                if len(tmp3a)==4:
                    tmp4a=tmp3a.reshape(1,4,6,3)
                    if verbose>0:
                        a1,b1,c1=tetrahedron_volume_6d(tmp4a[0])
                        vol1=(a1+b1*TAU)/c1
                        print('              volume, %d %d %d (%8.6f)'%(a1,b1,c1,vol1))
                        print('              volume (numerical value)',tetrahedron_volume_6d_numerical(tmp4a[0]))
                    else:
                        pass
                else:
                    tmp4a=tetrahedralization_points(tmp3a,verbose)
                
                num1=len(tmp4a)
                if verbose>0:
                    print('         -> number of tetrahedron,  %d'%(num1))
                else:
                    pass
                return tmp4a
                """
                if num1!=1:
                    for i1 in range(num1):
                        tmp2c=centroid(tmp4a[i1])
                        # check tmp2c is inside both tetrahedron_1 and 2
                        flag1=inside_outside_tetrahedron(tmp2c,tetrahedron_1[0],tetrahedron_1[1],tetrahedron_1[2],tetrahedron_1[3])
                        flag2=inside_outside_tetrahedron(tmp2c,tetrahedron_2[0],tetrahedron_2[1],tetrahedron_2[2],tetrahedron_2[3])
                        #
                        if verbose>1:
                            print('              tetraheddron %d'%(i1))
                            
                            for i2 in range(4):
                                v1,v2,v3,v4,v5,v6=projection(tmp4a[i1][i2][0],tmp4a[i1][i2][1],tmp4a[i1][i2][2],tmp4a[i1][i2][3],tmp4a[i1][i2][4],tmp4a[i1][i2][5])
                                print('              Qq %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/(v4[2]),(v5[0]+v5[1]*TAU)/(v5[2]),(v6[0]+v6[1]*TAU)/(v6[2])))
                            
                            # volume of tetrahedron
                            a1,b1,c1=tetrahedron_volume_6d(tmp4a[i1])
                            vol1=(a1+b1*TAU)/c1
                            print('              volume, %d %d %d (%8.6f)'%(a1,b1,c1,vol1))
                            print('              volume (numerical value)',tetrahedron_volume_6d_numerical(tmp4a[i1]))
                        else:
                            pass
                        #
                        if flag1==0 and flag2==0: # inside
                            if verbose>1:
                                print('              in')
                            else:
                                pass
                            if len(tmp1b)==1:
                                tmp1b=tmp4a[i1].reshape(72)
                            else:
                                tmp1b=np.append(tmp1b,tmp4a[i1])
                        else:
                            if verbose>1:
                                print('              out (%d,%d)'%(flag1,flag2))
                            else:
                                pass
                            pass
                    num2=len(tmp1b)
                    if num2!=1:
                        if int(num2/72)==num1: # 全体が入っている場合
                            return tmp4a
                        else:  # 一部が交差している場合
                            return tmp1b.reshape(int(num2/72),4,6,3)
                    else:
                        return tmp4a
                else:
                    return tmp4a
                """
            else:
                return tmp4a
        else:
            return tmp4a
    else:
        return tmp4a

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray intersection_segment_surface(np.ndarray[DTYPE_int_t, ndim=2] segment_1,
                                                np.ndarray[DTYPE_int_t, ndim=2] segment_2,
                                                np.ndarray[DTYPE_int_t, ndim=2] surface_1,
                                                np.ndarray[DTYPE_int_t, ndim=2] surface_2,
                                                np.ndarray[DTYPE_int_t, ndim=2] surface_3,
                                                int verbose):
#def intersection_segment_surface(np.ndarray[DTYPE_int_t, ndim=2] segment_1,np.ndarray[DTYPE_int_t, ndim=2] segment_2,np.ndarray[DTYPE_int_t, ndim=2] surface_1,np.ndarray[DTYPE_int_t, ndim=2] surface_2,np.ndarray[DTYPE_int_t, ndim=2] surface_3):
    #
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1,tmp1a
    #cdef np.ndarray[DTYPE_int_t,ndim=1] seg1e1,seg1e2,seg1e3
    cdef np.ndarray[DTYPE_int_t,ndim=1] seg1i1,seg1i2,seg1i3
    #cdef np.ndarray[DTYPE_int_t,ndim=1] seg2e1,seg2e2,seg2e3
    cdef np.ndarray[DTYPE_int_t,ndim=1] seg2i1,seg2i2,seg2i3
    #cdef np.ndarray[DTYPE_int_t,ndim=1] sur1e1,sur1e2,sur1e3
    cdef np.ndarray[DTYPE_int_t,ndim=1] sur1i1,sur1i2,sur1i3
    #cdef np.ndarray[DTYPE_int_t,ndim=1] sur2e1,sur2e2,sur2e3
    cdef np.ndarray[DTYPE_int_t,ndim=1] sur2i1,sur2i2,sur2i3
    #cdef np.ndarray[DTYPE_int_t,ndim=1] sur3e1,sur3e2,sur3e3
    cdef np.ndarray[DTYPE_int_t,ndim=1] sur3i1,sur3i2,sur3i3
    cdef np.ndarray[DTYPE_int_t,ndim=2] vec1,vecBA,vecCD,vecCE,vecCA
    cdef long bx1,bx2,bx3,by1,by2,by3,bz1,bz2,bz3
    cdef long cx1,cx2,cx3,cy1,cy2,cy3,cz1,cz2,cz3
    cdef long dx1,dx2,dx3,dy1,dy2,dy3,dz1,dz2,dz3
    cdef long ex1,ex2,ex3,ey1,ey2,ey3,ez1,ez2,ez3
    cdef long f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12
    cdef long g1,g2,g3,g4,g5,g6,g7,g8,g9
    cdef long h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18
    cdef long i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18
    cdef long j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18
    cdef double val1,val2,val3,val4
    #cdef int verbose
    
    #verbose=0
    
    if verbose>0:
        print('          intersection_segment_surface()')
    else:
        pass
    #
    #seg1e1,seg1e2,seg1e3,seg1i1,seg1i2,seg1i3=projection(segment_1[0],segment_1[1],segment_1[2],segment_1[3],segment_1[4],segment_1[5])
    #seg2e1,seg2e2,seg2e3,seg2i1,seg2i2,seg2i3=projection(segment_2[0],segment_2[1],segment_2[2],segment_2[3],segment_2[4],segment_2[5])
    #sur1e1,sur1e2,sur1e3,sur1i1,sur1i2,sur1i3=projection(surface_1[0],surface_1[1],surface_1[2],surface_1[3],surface_1[4],surface_1[5])
    #sur2e1,sur2e2,sur2e3,sur2i1,sur2i2,sur2i3=projection(surface_2[0],surface_2[1],surface_2[2],surface_2[3],surface_2[4],surface_2[5])
    #sur3e1,sur3e2,sur3e3,sur3i1,sur3i2,sur3i3=projection(surface_3[0],surface_3[1],surface_3[2],surface_3[3],surface_3[4],surface_3[5])
    _,_,_,seg1i1,seg1i2,seg1i3=projection(segment_1[0],segment_1[1],segment_1[2],segment_1[3],segment_1[4],segment_1[5])
    _,_,_,seg2i1,seg2i2,seg2i3=projection(segment_2[0],segment_2[1],segment_2[2],segment_2[3],segment_2[4],segment_2[5])
    _,_,_,sur1i1,sur1i2,sur1i3=projection(surface_1[0],surface_1[1],surface_1[2],surface_1[3],surface_1[4],surface_1[5])
    _,_,_,sur2i1,sur2i2,sur2i3=projection(surface_2[0],surface_2[1],surface_2[2],surface_2[3],surface_2[4],surface_2[5])
    _,_,_,sur3i1,sur3i2,sur3i3=projection(surface_3[0],surface_3[1],surface_3[2],surface_3[3],surface_3[4],surface_3[5])

    #
    # Origin: seg1i1,seg1i2,seg1i3
    # line segment
    # segment line A-B: seg3i1,seg3i2,seg3i3
    #
    #ax1,ax2,ax3=seg1i1[0],seg1i1[1],seg1i1[2]
    #ay1,ay2,ay3=seg1i2[0],seg1i2[1],seg1i2[2]
    #az1,az2,az3=seg1i3[0],seg1i3[1],seg1i3[2]
    # AB
    #bx1,bx2,bx3=sub(seg2i1[0],seg2i1[1],seg2i1[2],seg1i1[0],seg1i1[1],seg1i1[2])
    #by1,by2,by3=sub(seg2i2[0],seg2i2[1],seg2i2[2],seg1i2[0],seg1i2[1],seg1i2[2])
    #bz1,bz2,bz3=sub(seg2i3[0],seg2i3[1],seg2i3[2],seg1i3[0],seg1i3[1],seg1i3[2])
    # BA
    [bx1,bx2,bx3]=sub(seg1i1[0],seg1i1[1],seg1i1[2],seg2i1[0],seg2i1[1],seg2i1[2])
    [by1,by2,by3]=sub(seg1i2[0],seg1i2[1],seg1i2[2],seg2i2[0],seg2i2[1],seg2i2[2])
    [bz1,bz2,bz3]=sub(seg1i3[0],seg1i3[1],seg1i3[2],seg2i3[0],seg2i3[1],seg2i3[2])
    # plane CDE
    # AC
    #cx1,cx2,cx3=sub(sur1i1[0],sur1i1[1],sur1i1[2],seg1i1[0],seg1i1[1],seg1i1[2])
    #cy1,cy2,cy3=sub(sur1i2[0],sur1i2[1],sur1i2[2],seg1i2[0],seg1i2[1],seg1i2[2])
    #cz1,cz2,cz3=sub(sur1i3[0],sur1i3[1],sur1i3[2],seg1i3[0],seg1i3[1],seg1i3[2])
    # CA
    [cx1,cx2,cx3]=sub(seg1i1[0],seg1i1[1],seg1i1[2],sur1i1[0],sur1i1[1],sur1i1[2])
    [cy1,cy2,cy3]=sub(seg1i2[0],seg1i2[1],seg1i2[2],sur1i2[0],sur1i2[1],sur1i2[2])
    [cz1,cz2,cz3]=sub(seg1i3[0],seg1i3[1],seg1i3[2],sur1i3[0],sur1i3[1],sur1i3[2])
    # CD
    [dx1,dx2,dx3]=sub(sur2i1[0],sur2i1[1],sur2i1[2],sur1i1[0],sur1i1[1],sur1i1[2])
    [dy1,dy2,dy3]=sub(sur2i2[0],sur2i2[1],sur2i2[2],sur1i2[0],sur1i2[1],sur1i2[2])
    [dz1,dz2,dz3]=sub(sur2i3[0],sur2i3[1],sur2i3[2],sur1i3[0],sur1i3[1],sur1i3[2])
    # CE
    [ex1,ex2,ex3]=sub(sur3i1[0],sur3i1[1],sur3i1[2],sur1i1[0],sur1i1[1],sur1i1[2])
    [ey1,ey2,ey3]=sub(sur3i2[0],sur3i2[1],sur3i2[2],sur1i2[0],sur1i2[1],sur1i2[2])
    [ez1,ez2,ez3]=sub(sur3i3[0],sur3i3[1],sur3i3[2],sur1i3[0],sur1i3[1],sur1i3[2])
    #
    #vecBA=np.array([[-bx1,-bx2, bx3],[-by1,-by2, by3],[-bz1,-bz2, bz3]]) # line segment BA = -AB
    vecBA=np.array([[ bx1, bx2, bx3],[ by1, by2, by3],[ bz1, bz2, bz3]]) # line segment BA
    vecCD=np.array([[ dx1, dx2, dx3],[ dy1, dy2, dy3],[ dz1, dz2, dz3]]) # edge segment of triangle CDE, CD
    vecCE=np.array([[ ex1, ex2, ex3],[ ey1, ey2, ey3],[ ez1, ez2, ez3]]) # edge segment of triangle CDE, CE
    #vecCA=np.array([[-cx1,-cx2, cx3],[-cy1,-cy2, cy3],[-cz1,-cz2, cz3]]) # CA = -AC
    vecCA=np.array([[ cx1, cx2, cx3],[ cy1, cy2, cy3],[ cz1, cz2, cz3]]) # CA
    #
    # below part consists of numerica calculations....
    #
    tmp1=np.array([0])
    f1,f2,f3=det_matrix(vecCD,vecCE,vecBA)
    val1=(f1+f2*TAU)/(f3)
    if f1==0 and f2==0:
        if verbose>1:
            print('          line segment and triangle are parallel')
        else:
            pass
        tmp1a=intersection_two_segment(segment_1,segment_2,surface_1,surface_2,verbose-1)
        if len(tmp1a)!=1:
            if len(tmp1)==1:
                tmp1=tmp1a
            else:
                tmp1=np.append(tmp1a,tmp1)
        else:
            pass
        tmp1a=intersection_two_segment(segment_1,segment_2,surface_1,surface_3,verbose-1)
        if len(tmp1a)!=1:
            if len(tmp1)==1:
                tmp1=tmp1a
            else:
                tmp1=np.append(tmp1a,tmp1)
        else:
            pass
        tmp1a=intersection_two_segment(segment_1,segment_2,surface_2,surface_3,verbose)
        if len(tmp1a)!=1:
            if len(tmp1)==1:
                tmp1=tmp1a
            else:
                tmp1=np.append(tmp1a,tmp1)
        else:
            pass
        if verbose>0:
            print('          Intersectiong point:',tmp1)
        else:
            pass
        return tmp1
    else:
        f4,f5,f6=det_matrix(vecCA,vecCE,vecBA)
        val2=(f4+f5*TAU)/f6
        #
        #   u = val2/val1:
        #  g4,g5,g6 = div(f4,f5,f6,f1,f2,f3)
        #
        #   v = val3/val1:
        #  g7,g8,g9 = div(f7,f8,f9,f1,f2,f3)
        #
        #   t = val4/val1:
        #  g1,g2,g3 = div(f10,f11,f12,f1,f2,f3)
        if val2/val1>=0.0 and val2/val1<=1.0:
            f7,f8,f9=det_matrix(vecCD,vecCA,vecBA)
            val3=(f7+f8*TAU)/f9
            if val3/val1>=0.0 and (val2+val3)/val1<=1.0:
                f10,f11,f12=det_matrix(vecCD,vecCE,vecCA)
                val4=(f10+f11*TAU)/f12
                if val4/val1>=0.0 and val4/val1<=1.0: # t = val4/val1
                    g1,g2,g3=div(f10,f11,f12,f1,f2,f3) # t in TAU-style
                    #
                    #interval=line1a+t*(line1b-line1a)
                    #
                    # line1b-line1a
                    [h1,h2,h3]=sub(segment_2[0][0],segment_2[0][1],segment_2[0][2],segment_1[0][0],segment_1[0][1],segment_1[0][2])
                    [h4,h5,h6]=sub(segment_2[1][0],segment_2[1][1],segment_2[1][2],segment_1[1][0],segment_1[1][1],segment_1[1][2])
                    [h7,h8,h9]=sub(segment_2[2][0],segment_2[2][1],segment_2[2][2],segment_1[2][0],segment_1[2][1],segment_1[2][2])
                    [h10,h11,h12]=sub(segment_2[3][0],segment_2[3][1],segment_2[3][2],segment_1[3][0],segment_1[3][1],segment_1[3][2])
                    [h13,h14,h15]=sub(segment_2[4][0],segment_2[4][1],segment_2[4][2],segment_1[4][0],segment_1[4][1],segment_1[4][2])
                    [h16,h17,h18]=sub(segment_2[5][0],segment_2[5][1],segment_2[5][2],segment_1[5][0],segment_1[5][1],segment_1[5][2])
                    #
                    # line1a
                    i1,i2,i3=segment_1[0][0],segment_1[0][1],segment_1[0][2]
                    i4,i5,i6=segment_1[1][0],segment_1[1][1],segment_1[1][2]
                    i7,i8,i9=segment_1[2][0],segment_1[2][1],segment_1[2][2]
                    i10,i11,i12=segment_1[3][0],segment_1[3][1],segment_1[3][2]
                    i13,i14,i15=segment_1[4][0],segment_1[4][1],segment_1[4][2]
                    i16,i17,i18=segment_1[5][0],segment_1[5][1],segment_1[5][2]
                    #
                    [j1,j2,j3]=mul(g1,g2,g3,h1,h2,h3)
                    [j1,j2,j3]=add(j1,j2,j3,i1,i2,i3)
                    #
                    [j4,j5,j6]=mul(g1,g2,g3,h4,h5,h6)
                    [j4,j5,j6]=add(j4,j5,j6,i4,i5,i6)
                    #
                    [j7,j8,j9]=mul(g1,g2,g3,h7,h8,h9)
                    [j7,j8,j9]=add(j7,j8,j9,i7,i8,i9)
                    #
                    [j10,j11,j12]=mul(g1,g2,g3,h10,h11,h12)
                    [j10,j11,j12]=add(j10,j11,j12,i10,i11,i12)
                    #
                    [j13,j14,j15]=mul(g1,g2,g3,h13,h14,h15)
                    [j13,j14,j15]=add(j13,j14,j15,i13,i14,i15)
                    #
                    [j16,j17,j18]=mul(g1,g2,g3,h16,h17,h18)
                    [j16,j17,j18]=add(j16,j17,j18,i16,i17,i18)

                    tmp1=np.array([j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18])
                else:
                    pass
            else:
                pass
        else:
            pass
        return tmp1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray intersection_two_segment(np.ndarray[DTYPE_int_t, ndim=2] segment_1_A,
                                        np.ndarray[DTYPE_int_t, ndim=2] segment_1_B,
                                        np.ndarray[DTYPE_int_t, ndim=2] segment_2_C,
                                        np.ndarray[DTYPE_int_t, ndim=2] segment_2_D,
                                        int verbose):
    cdef double s,t
    cdef long ax1,ax2,ax3,ay1,ay2,ay3,az1,az2,az3
    cdef long bx1,bx2,bx3,by1,by2,by3,bz1,bz2,bz3
    cdef long cx1,cx2,cx3,cy1,cy2,cy3,cz1,cz2,cz3
    cdef long a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3,e1,e2,e3,f1,f2,f3,g1,g2,g3
    cdef long m1,m2,m3,n1,n2,n3,o1,o2,o3,p1,p2,p3
    cdef long z1,z2,z3
    cdef long ddx1,ddx2,ddx3
    cdef long h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18
    cdef long i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18
    cdef long j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c
    cdef np.ndarray[DTYPE_int_t,ndim=1] seg1ai1,seg1ai2,seg1ai3,seg1bi1,seg1bi2,seg1bi3
    cdef np.ndarray[DTYPE_int_t,ndim=1] seg2ci1,seg2ci2,seg2ci3,seg2di1,seg2di2,seg2di3
    #cdef int verbose
    
    #verbose=0
    
    if verbose>0:
        print('           intersection_two_segment()')
    else:
        pass
        
    if verbose>1:
        print('           segment_1',segment_1_A)
        print('           segment_1',segment_1_B)
        print('           segment_2',segment_2_C)
        print('           segment_2',segment_2_D)
    else:
        pass
    #tmp1a,tmp1b,tmp1c,seg1ai1,seg1ai2,seg1ai3=projection(segment_1_A[0],segment_1_A[1],segment_1_A[2],segment_1_A[3],segment_1_A[4],segment_1_A[5])
    #tmp1a,tmp1b,tmp1c,seg1bi1,seg1bi2,seg1bi3=projection(segment_1_B[0],segment_1_B[1],segment_1_B[2],segment_1_B[3],segment_1_B[4],segment_1_B[5])
    #tmp1a,tmp1b,tmp1c,seg2ci1,seg2ci2,seg2ci3=projection(segment_2_C[0],segment_2_C[1],segment_2_C[2],segment_2_C[3],segment_2_C[4],segment_2_C[5])
    #tmp1a,tmp1b,tmp1c,seg2di1,seg2di2,seg2di3=projection(segment_2_D[0],segment_2_D[1],segment_2_D[2],segment_2_D[3],segment_2_D[4],segment_2_D[5])

    _,_,_,seg1ai1,seg1ai2,seg1ai3=projection(segment_1_A[0],segment_1_A[1],segment_1_A[2],segment_1_A[3],segment_1_A[4],segment_1_A[5])
    _,_,_,seg1bi1,seg1bi2,seg1bi3=projection(segment_1_B[0],segment_1_B[1],segment_1_B[2],segment_1_B[3],segment_1_B[4],segment_1_B[5])
    _,_,_,seg2ci1,seg2ci2,seg2ci3=projection(segment_2_C[0],segment_2_C[1],segment_2_C[2],segment_2_C[3],segment_2_C[4],segment_2_C[5])
    _,_,_,seg2di1,seg2di2,seg2di3=projection(segment_2_D[0],segment_2_D[1],segment_2_D[2],segment_2_D[3],segment_2_D[4],segment_2_D[5])
    
    # vec AB
    [ax1,ax2,ax3]=sub(seg1bi1[0],seg1bi1[1],seg1bi1[2],seg1ai1[0],seg1ai1[1],seg1ai1[2])
    [ay1,ay2,ay3]=sub(seg1bi2[0],seg1bi2[1],seg1bi2[2],seg1ai2[0],seg1ai2[1],seg1ai2[2])
    [az1,az2,az3]=sub(seg1bi3[0],seg1bi3[1],seg1bi3[2],seg1ai3[0],seg1ai3[1],seg1ai3[2])

    # vec AC
    [bx1,bx2,bx3]=sub(seg2ci1[0],seg2ci1[1],seg2ci1[2],seg1ai1[0],seg1ai1[1],seg1ai1[2])
    [by1,by2,by3]=sub(seg2ci2[0],seg2ci2[1],seg2ci2[2],seg1ai2[0],seg1ai2[1],seg1ai2[2])
    [bz1,bz2,bz3]=sub(seg2ci3[0],seg2ci3[1],seg2ci3[2],seg1ai3[0],seg1ai3[1],seg1ai3[2])
    
    # vec CD
    [cx1,cx2,cx3]=sub(seg2di1[0],seg2di1[1],seg2di1[2],seg2ci1[0],seg2ci1[1],seg2ci1[2])
    [cy1,cy2,cy3]=sub(seg2di2[0],seg2di2[1],seg2di2[2],seg2ci2[0],seg2ci2[1],seg2ci2[2])
    [cz1,cz2,cz3]=sub(seg2di3[0],seg2di3[1],seg2di3[2],seg2ci3[0],seg2ci3[1],seg2ci3[2])
    
    # dot_product(vecAC,vecCD)
    a1,a2,a3=dot_product(np.array([bx1,bx2,bx3]),np.array([by1,by2,by3]),np.array([bz1,bz2,bz3]),np.array([cx1,cx2,cx3]),np.array([cy1,cy2,cy3]),np.array([cz1,cz2,cz3]))
    # dot_product(vecCD,vecAB)
    b1,b2,b3=dot_product(np.array([cx1,cx2,cx3]),np.array([cy1,cy2,cy3]),np.array([cz1,cz2,cz3]),np.array([ax1,ax2,ax3]),np.array([ay1,ay2,ay3]),np.array([az1,az2,az3]))
    # dot_product(vecCD,vecCD)
    c1,c2,c3=dot_product(np.array([cx1,cx2,cx3]),np.array([cy1,cy2,cy3]),np.array([cz1,cz2,cz3]),np.array([cx1,cx2,cx3]),np.array([cy1,cy2,cy3]),np.array([cz1,cz2,cz3]))
    # dot_product(vecAC,vecAB)
    d1,d2,d3=dot_product(np.array([bx1,bx2,bx3]),np.array([by1,by2,by3]),np.array([bz1,bz2,bz3]),np.array([ax1,ax2,ax3]),np.array([ay1,ay2,ay3]),np.array([az1,az2,az3]))
    # dot_product(vecAB,vecCD)
    e1,e2,e3=dot_product(np.array([ax1,ax2,ax3]),np.array([ay1,ay2,ay3]),np.array([az1,az2,az3]),np.array([cx1,cx2,cx3]),np.array([cy1,cy2,cy3]),np.array([cz1,cz2,cz3]))
    # dot_product(vecAB,vecAB)
    f1,f2,f3=dot_product(np.array([ax1,ax2,ax3]),np.array([ay1,ay2,ay3]),np.array([az1,az2,az3]),np.array([ax1,ax2,ax3]),np.array([ay1,ay2,ay3]),np.array([az1,az2,az3]))

    #bunbo=dot_product(vecAB,vecCD)*dot_product(vecCD,vecAB)-dot_product(vecAB,vecAB)*dot_product(vecCD,vecCD)
    m1,m2,m3=mul(e1,e2,e3,b1,b2,b3)
    n1,n2,n3=mul(f1,f2,f3,c1,c2,c3)
    p1,p2,p3=sub(m1,m2,m3,n1,n2,n3)    
    tmp1a=np.array([0])
    if p1!=0 or p2!=0:
        # bunshi=dot_product(vecAC,vecCD)*dot_product(vecCD,vecAB)-dot_product(vecCD,vecCD)*dot_product(vecAC,vecAB)
        m1,m2,m3=mul(a1,a2,a3,b1,b2,b3)
        n1,n2,n3=mul(c1,c2,c3,d1,d2,d3)
        o1,o2,o3=sub(m1,m2,m3,n1,n2,n3)
        #
        # Numerical calc
        #
        # s=bunshi/bunbo
        s1,s2,s3=div(o1,o2,o3,p1,p2,p3)
        s=(s1+TAU*s2)/s3
        #t=(-dot_product(vecAC,vecCD)+s*dot_product(vecAB,vecCD))/dot_product(vecCD,vecCD)
        # dot_product(vecAB,vecCD)
        g1,g2,g3=dot_product(np.array([ax1,ax2,ax3]),np.array([ay1,ay2,ay3]),np.array([az1,az2,az3]),np.array([cx1,cx2,cx3]),np.array([cy1,cy2,cy3]),np.array([cz1,cz2,cz3]))
        h1,h2,h3=mul(s1,s2,s3,g1,g2,g3)
        h1,h2,h3=sub(h1,h2,h3,a1,a2,a3)
        if verbose>1:
            print('                           s = %8.6f'%(s))
        else:
            pass
        if c1!=0 or c2!=0:
            t1,t2,t3=div(h1,h2,h3,c1,c2,c3)
            t=(t1+TAU*t2)/t3
            if verbose>1:
                print('                           t = %8.6f'%(t))
            else:
                pass
            if s>=0.0 and s<=1.0 and t>=0.0 and t<=1.0:
                # ddx=(L2ax-L1ax)-s*(L1bx-L1ax)+t*(L2bx-L2ax)
                #bx1,bx2,bx3 # (L2ax-L1ax)
                #mul(s1,s2,s3,ax1,ax2,ax3) # s*(L1bx-L1ax)
                #mul(t1,t2,t3,cx1,cx2,cx3) # t*(L2bx-L2ax)
                ddx1,ddx2,ddx3=mul(s1,s2,s3,ax1,ax2,ax3)
                ddx1,ddx2,ddx3=sub(bx1,bx2,bx3,ddx1,ddx2,ddx3)
                z1,z2,z3=mul(t1,t2,t3,cx1,cx2,cx3)
                ddx1,ddx2,ddx3=add(ddx1,ddx2,ddx3,z1,z2,z3)
                # ddx**2
                ddx1,ddx2,ddx3=mul(ddx1,ddx2,ddx3,ddx1,ddx2,ddx3)
                if verbose>1:
                    print('           ddx1,ddx2 = %d %d'%(ddx1,ddx2))
                else:
                    pass
                # ddy=(L2ay-L1ay)-s*(L1by-L1ay)+t*(L2by-L2ay)
                ddy1,ddy2,ddy3=mul(s1,s2,s3,ay1,ay2,ay3)
                ddy1,ddy2,ddy3=sub(by1,by2,by3,ddy1,ddy2,ddy3)
                z1,z2,z3=mul(t1,t2,t3,cy1,cy2,cy3)
                ddy1,ddy2,ddy3=add(ddy1,ddy2,ddy3,z1,z2,z3)
                # ddy**2
                ddy1,ddy2,ddy3=mul(ddy1,ddy2,ddy3,ddy1,ddy2,ddy3)
                if verbose>1:
                    print('           ddy1,ddy2 = %d %d'%(ddy1,ddy2))
                else:
                    pass
                # ddz=(L2az-L1az)-s*(L1bz-L1az)+t*(L2bz-L2az)
                ddz1,ddz2,ddz3=mul(s1,s2,s3,az1,az2,az3)
                ddz1,ddz2,ddz3=sub(bz1,bz2,bz3,ddz1,ddz2,ddz3)
                z1,z2,z3=mul(t1,t2,t3,cz1,cz2,cz3)
                ddz1,ddz2,ddz3=add(ddz1,ddz2,ddz3,z1,z2,z3)
                # ddz**2
                ddz1,ddz2,ddz3=mul(ddz1,ddz2,ddz3,ddz1,ddz2,ddz3)
                if verbose>1:
                    print('           ddz1,ddz2 = %d %d'%(ddz1,ddz2))
                else:
                    pass
                    
                z1,z2,z3=add(ddx1,ddx2,ddx3,ddy1,ddy2,ddy3)
                z1,z2,z3=add(z1,z2,z3,ddz1,ddz2,ddz3)
                if verbose>1:
                    print('           z1,z2 = %d %d'%(z1,z2))
                else:
                    pass
                #if ddx**2+ddy**2+ddz**2<EPS:
                #if ddx1==0 and ddx2==0 and ddy1==0 and ddy2==0 and ddz1==0 and ddz2==0:
                if z1==0 and z2==0:
                    #
                    #interval=line1a+s*(line1b-line1a)
                    #
                    # line1b-line1a
                    [h1,h2,h3]=sub(segment_1_B[0][0],segment_1_B[0][1],segment_1_B[0][2],segment_1_A[0][0],segment_1_A[0][1],segment_1_A[0][2])
                    [h4,h5,h6]=sub(segment_1_B[1][0],segment_1_B[1][1],segment_1_B[1][2],segment_1_A[1][0],segment_1_A[1][1],segment_1_A[1][2])
                    [h7,h8,h9]=sub(segment_1_B[2][0],segment_1_B[2][1],segment_1_B[2][2],segment_1_A[2][0],segment_1_A[2][1],segment_1_A[2][2])
                    [h10,h11,h12]=sub(segment_1_B[3][0],segment_1_B[3][1],segment_1_B[3][2],segment_1_A[3][0],segment_1_A[3][1],segment_1_A[3][2])
                    [h13,h14,h15]=sub(segment_1_B[4][0],segment_1_B[4][1],segment_1_B[4][2],segment_1_A[4][0],segment_1_A[4][1],segment_1_A[4][2])
                    [h16,h17,h18]=sub(segment_1_B[5][0],segment_1_B[5][1],segment_1_B[5][2],segment_1_A[5][0],segment_1_A[5][1],segment_1_A[5][2])
                    #
                    # line1a
                    i1,i2,i3=segment_1_A[0][0],segment_1_A[0][1],segment_1_A[0][2]
                    i4,i5,i6=segment_1_A[1][0],segment_1_A[1][1],segment_1_A[1][2]
                    i7,i8,i9=segment_1_A[2][0],segment_1_A[2][1],segment_1_A[2][2]
                    i10,i11,i12=segment_1_A[3][0],segment_1_A[3][1],segment_1_A[3][2]
                    i13,i14,i15=segment_1_A[4][0],segment_1_A[4][1],segment_1_A[4][2]
                    i16,i17,i18=segment_1_A[5][0],segment_1_A[5][1],segment_1_A[5][2]
                    #
                    [j1,j2,j3]=mul(s1,s2,s3,h1,h2,h3)
                    [j1,j2,j3]=add(j1,j2,j3,i1,i2,i3)
                    #
                    [j4,j5,j6]=mul(s1,s2,s3,h4,h5,h6)
                    [j4,j5,j6]=add(j4,j5,j6,i4,i5,i6)
                    #
                    [j7,j8,j9]=mul(s1,s2,s3,h7,h8,h9)
                    [j7,j8,j9]=add(j7,j8,j9,i7,i8,i9)
                    #
                    [j10,j11,j12]=mul(s1,s2,s3,h10,h11,h12)
                    [j10,j11,j12]=add(j10,j11,j12,i10,i11,i12)
                    #
                    [j13,j14,j15]=mul(s1,s2,s3,h13,h14,h15)
                    [j13,j14,j15]=add(j13,j14,j15,i13,i14,i15)
                    #
                    [j16,j17,j18]=mul(s1,s2,s3,h16,h17,h18)
                    [j16,j17,j18]=add(j16,j17,j18,i16,i17,i18)
                    tmp1a=np.array([j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18])
                    #if verbose>2:
                    #    print(tmp1a)
                    #else:
                    #    pass
                else:
                    pass
            else:
                pass
        else:
            pass
    else:
        pass
    return tmp1a

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray tetrahedralization_points(np.ndarray[DTYPE_int_t, ndim=3] points,
                                            int verbose):
    cdef int i,num,counter
    cdef long t1,t2,t3,t4,v1,v2,v3,w1,w2,w3
    cdef double vx,vy,vz
    #cdef np.ndarray[DTYPE_int_t,ndim=1] xe,ye,ze
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,xi,yi,zi
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    cdef np.ndarray[DTYPE_double_t,ndim=1] tmp1v
    cdef np.ndarray[DTYPE_double_t,ndim=2] tmp2v
    cdef list ltmp
    
    if verbose>0:
        print('           tetrahedralization_points()')
        print('            N of points: %3d :'%(len(points)))
    else:
        pass
    
    tmp3a=points
    for i in range(len(tmp3a)):
        #xe,ye,ze,xi,yi,zi=projection(tmp3a[i][0],tmp3a[i][1],tmp3a[i][2],tmp3a[i][3],tmp3a[i][4],tmp3a[i][5])
        _,_,_,xi,yi,zi=projection(tmp3a[i][0],tmp3a[i][1],tmp3a[i][2],tmp3a[i][3],tmp3a[i][4],tmp3a[i][5])
        #vx=(xi[0]+xi[1]*TAU)/float(xi[2]) # numeric value of xi
        #vy=(yi[0]+yi[1]*TAU)/float(yi[2])
        #vz=(zi[0]+zi[1]*TAU)/float(zi[2])
        vx=(xi[0]+xi[1]*TAU)/xi[2] # numeric value of xi
        vy=(yi[0]+yi[1]*TAU)/yi[2]
        vz=(zi[0]+zi[1]*TAU)/zi[2]
        if i==0:
            tmp1v=np.array([vx,vy,vz])
            #print '     %8.6f %8.6f %8.6f'%(vx,vy,vz)
        else:
            tmp1v=np.append(tmp1v,[vx,vy,vz])
            #print '     %8.6f %8.6f %8.6f'%(vx,vy,vz)
    tmp2v=tmp1v.reshape(int(len(tmp1v)/3),3)
    ltmp=decomposition(tmp2v)
    
    if verbose>0:
        print('            -> N of tetrahedron: %3d'%(len(ltmp)))
    else:
        pass
    
    tmp4a=np.array([[[[0]]]])
    if ltmp!=[0]:
        #w1,w2,w3=0,0,1
        counter=0
        for i in range(len(ltmp)):
            tmp3b=np.array([tmp3a[ltmp[i][0]],tmp3a[ltmp[i][1]],tmp3a[ltmp[i][2]],tmp3a[ltmp[i][3]]]).reshape(4,6,3)
            #print 'tmp3b',tmp3b
            v1,v2,v3=tetrahedron_volume_6d(tmp3b)
            if v1==0 and v2==0:
                if verbose>0:
                    print('            %d-th tet, empty'%(i))
                else:
                    pass
                pass
            #elif (v1+v2*TAU)/float(v3)<0.0:
            #    print '     %d-th tet, volume : %d %d %d (%8.6f) ignored!'%(i,v1,v2,v3,(v1+v2*TAU)/float(v3))
            #    pass
            else:
                if counter==0:
                    tmp1a=tmp3b.reshape(72) # 4*6*3=72
                    if verbose>0:
                        print('            %d-th tet, volume : %d %d %d (%8.6f)'%(i,v1,v2,v3,(v1+v2*TAU)/v3))
                    else:
                        pass
                else:
                    tmp1a=np.append(tmp1a,tmp3b)
                    if verbose>0:
                        print('            %d-th tet, volume : %d %d %d (%8.6f)'%(i,v1,v2,v3,(v1+v2*TAU)/v3))
                    else:
                        pass
                #w1,w2,w3=add(v1,v2,v3,w1,w2,w3)
                counter+=1
        if counter!=0:
            tmp4a=tmp1a.reshape(int(len(tmp1a)/72),4,6,3) # 4*6*3=72    
            if verbose>0:
                w1,w2,w3=obj_volume_6d(tmp4a)
                print('            -> Total : %d %d %d (%8.6f)'%(w1,w2,w3,(w1+w2*TAU)/w3))
            else:
                pass
        else:
            pass
            #tmp4a=np.array([[[[0]]]])
        #return tmp4a
    else:
        pass
        #print('tmp2v',tmp2v)
        #return np.array([[[[0]]]])
        #return tmp4a
    return tmp4a

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray tetrahedralization_1(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                    np.ndarray[DTYPE_int_t, ndim=2] point,
                                    int verbose):
    cdef int i1,counter1
    cdef long v1,v2,v3
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    #cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4
    
    if verbose>0:
        print('    tetrahedralization_1()')
        v1,v2,v3=obj_volume_6d(obj)     # check volume of 'tetrahedron'
        print('      volume of initial tetrahedron: %d %d %d'%(v1,v2,v3))
    else:
        pass
    
    counter1=0
    for i1 in range(len(obj)):
        if inside_outside_tetrahedron(point,obj[i1][0],obj[i1][1],obj[i1][2],obj[i1][3])==0: # point is inside obj
            counter1+=1
            break
        else: # outside
            pass
    if counter1!=0:
        if verbose>0:
            print('      added point is inside obj')
        else:
            pass
        #tmp1a=obj.reshape(len(obj)*72)
        #tmp1b=point.reshape(18)
        #tmp1a=np.append(tmp1a,tmp1b)
        #tmp3a=tmp1a.reshape(int(len(tmp1a)/18),6,3)
        tmp1a=np.append(obj,point)
        tmp3a=tmp1a.reshape(int(len(tmp1a)/18),6,3)
        tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
        return tetrahedralization_points(tmp3a,verbose)
    else:
        if verbose>0:
            print('      added point is out of obj')
        else:
            pass
        return obj

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray tetrahedralization(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron,
                                    np.ndarray[DTYPE_int_t, ndim=3] intersecting_point,
                                    int verbose):
    cdef int i,num,counter
    cdef long t1,t2,t3,t4,v1,v2,v3,w1,w2,w3
    #cdef long w1,w2,w3
    cdef double vx,vy,vz
    #cdef np.ndarray[DTYPE_int_t,ndim=1] xe,ye,ze
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,xi,yi,zi
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    cdef np.ndarray[DTYPE_double_t,ndim=1] tmp1v
    cdef np.ndarray[DTYPE_double_t,ndim=2] tmp2v
    cdef list ltmp
    # 
    if verbose>0:
        print('           tetrahedralization()')
        v1,v2,v3=tetrahedron_volume_6d(tetrahedron)     # check volume of 'tetrahedron'
        print('            volume of initial tetrahedron: %d %d %d'%(v1,v2,v3))
    else:
        pass
    
    tmp1a=np.append(tetrahedron,intersecting_point)
    tmp3a=tmp1a.reshape(int(len(tmp1a)/18),6,3)
    
    if verbose>0:
        print('            numbre of points: %3d :'%len(intersecting_point))
    else:
        pass
    for i in range(len(tmp3a)):
        #xe,ye,ze,xi,yi,zi=projection(tmp3a[i][0],tmp3a[i][1],tmp3a[i][2],tmp3a[i][3],tmp3a[i][4],tmp3a[i][5])
        _,_,_,xi,yi,zi=projection(tmp3a[i][0],tmp3a[i][1],tmp3a[i][2],tmp3a[i][3],tmp3a[i][4],tmp3a[i][5])
        #vx=(xi[0]+xi[1]*TAU)/float(xi[2]) # numeric value of xi
        #vy=(yi[0]+yi[1]*TAU)/float(yi[2])
        #vz=(zi[0]+zi[1]*TAU)/float(zi[2])
        vx=(xi[0]+xi[1]*TAU)/xi[2] # numeric value of xi
        vy=(yi[0]+yi[1]*TAU)/yi[2]
        vz=(zi[0]+zi[1]*TAU)/zi[2]
        if i==0:
            tmp1v=np.array([vx,vy,vz])
            #print '     %8.6f %8.6f %8.6f'%(vx,vy,vz)
        else:
            tmp1v=np.append(tmp1v,[vx,vy,vz])
            #print '     %8.6f %8.6f %8.6f'%(vx,vy,vz)
    tmp2v=tmp1v.reshape(int(len(tmp1v)/3),3)
    ltmp=decomposition(tmp2v)
    tmp4a=np.array([[[[0]]]])
    if ltmp!=[0]:
        #print '   -> N of tetrahedron: %3d'%(len(ltmp))
        #w1,w2,w3=0,0,1
        counter=0
        for i in range(len(ltmp)):
            tmp3b=np.array([tmp3a[ltmp[i][0]],tmp3a[ltmp[i][1]],tmp3a[ltmp[i][2]],tmp3a[ltmp[i][3]]]).reshape(4,6,3)
            #v1,v2,v3=tetrahedron_volume_6d(tmp3b)
            if v1==0 and v2==0:
                pass
            else:
                if counter==0:
                    tmp1a=tmp3b.reshape(72) # 4*6*3=72
                    #print '     volume : (%d+%d*TAU)/%d'%(v1,v2,v3)
                else:
                    tmp1a=np.append(tmp1a,tmp3b)
                    #print '     volume : (%d+%d*TAU)/%d'%(v1,v2,v3)
                    #w1,w2,w3=add(v1,v2,v3,w1,w2,w3)
                counter+=1
        if counter!=0:
            tmp4a=tmp1a.reshape(int(len(tmp1a)/72),4,6,3) # 4*6*3=72    
            if verbose>0:
                v1,v2,v3=obj_volume_6d(tmp4a)
                print('              -> numbre of tetrahedra: %3d:'%(int(len(tmp4a))))
                print('                 Total volume : %d %d %d (%8.6f)'%(v1,v2,v3,(v1+v2*TAU)/v3))
                #print('            -> Total2 : %d %d %d (%8.6f)'%(w1,w2,w3,(w1+w2*TAU)/w3))
            else:
                pass
        else:
            pass
        return tmp4a
    else:
        #return np.array([[[[0]]]])
        #return tmp4a
        #pass
        return tmp4a

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list decomposition(np.ndarray[DTYPE_double_t, ndim=2] tmp2v):
    cdef int i
    cdef list tmp=[]
    try:
        tri=Delaunay(tmp2v)
    except:
        print('error in decomposition()')
        tmp=[0]
    else:
        for i in range(len(tri.simplices)):
            tet=tri.simplices[i]
            tmp.append([tet[0],tet[1],tet[2],tet[3]])
    return tmp

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray centroid_obj(np.ndarray[DTYPE_int_t, ndim=4] obj):
    #  geometric center, centroid of OBJ
    cdef int i1,i2,i3,num
    cpdef long v1,v2,v3
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    tmp1a=np.array([0])
    num=len(obj)
    for i3 in range(6):
        v1,v2,v3=0,0,1
        for i1 in range(num):
            for i2 in range(4):
                v1,v2,v3=add(v1,v2,v3,obj[i1][i2][i3][0],obj[i1][i2][i3][1],obj[i1][i2][i3][2])
        v1,v2,v3=mul(v1,v2,v3,1,0,num*4)
        if i3!=0:
            tmp1a=np.append(tmp1a,[v1,v2,v3])
        else:
            tmp1a=np.array([v1,v2,v3])
    return tmp1a.reshape(6,3)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double ball_radius(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron,
                        np.ndarray[DTYPE_int_t, ndim=2] centroid):
    #  this transforms a tetrahedron to a boll which covers the tetrahedron
    #  the centre of the boll is the centroid of the tetrahedron.
    cdef int i1,i2,counter
    cdef long w1,w2,w3
    cdef double dd,radius
    #cdef np.ndarray[DTYPE_int_t,ndim=1] v1,v2,v3
    cdef np.ndarray[DTYPE_int_t,ndim=1] v4,v5,v6,tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3b
    
    counter=0
    tmp1a=np.array([0])
    for i1 in range(4):
        for i2 in range(6):
            w1,w2,w3=sub(centroid[i2][0],centroid[i2][1],centroid[i2][2],tetrahedron[i1][i2][0],tetrahedron[i1][i2][1],tetrahedron[i1][i2][2])
            if counter!=0:
                tmp1a=np.append(tmp1a,[w1,w2,w3])
            else:
                tmp1a=np.array([w1,w2,w3])
                counter+=1
    tmp3b=tmp1a.reshape(4,6,3)
    radius=0.0
    for i1 in range(4):
        #v1,v2,v3,v4,v5,v6=projection(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
        _,_,_,v4,v5,v6=projection(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
        dd=np.sqrt(((v4[0]+v4[1]*TAU)/v4[2])**2+((v5[0]+v5[1]*TAU)/v5[2])**2+((v6[0]+v6[1]*TAU)/v6[2])**2)
        if dd>radius:
            radius=dd
        else:
            pass
    return radius

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double ball_radius_obj(np.ndarray[DTYPE_int_t, ndim=4] obj,
                        np.ndarray[DTYPE_int_t, ndim=2] centroid):
    #  this transforms an OBJ to a boll which covers the OBJ
    #  the centre of the boll is the centroid of the OBJ.
    cdef int i1,i2,i3,num1,counter
    cdef long w1,w2,w3
    cdef double dd,radius
    #cdef np.ndarray[DTYPE_int_t,ndim=1] v1,v2,v3
    cdef np.ndarray[DTYPE_int_t,ndim=1] v4,v5,v6,tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    
    counter=0
    tmp1a=np.array([0])
    num1=len(obj)
    for i1 in range(num1):
        for i2 in range(4):
            for i3 in range(6):
                w1,w2,w3=sub(centroid[i3][0],centroid[i3][1],centroid[i3][2],obj[i1][i2][i3][0],obj[i1][i2][i3][1],obj[i1][i2][i3][2])
                if counter!=0:
                    tmp1a=np.append(tmp1a,[w1,w2,w3])
                else:
                    tmp1a=np.array([w1,w2,w3])
                    counter+=1
    tmp4a=tmp1a.reshape(num1,4,6,3)
    radius=0.0
    for i1 in range(num1):
        for i2 in range(4):
            #v1,v2,v3,v4,v5,v6=projection(tmp4a[i1][i2][0],tmp4a[i1][i2][1],tmp4a[i1][i2][2],tmp4a[i1][i2][3],tmp4a[i1][i2][4],tmp4a[i1][i2][5])
            _,_,_,v4,v5,v6=projection(tmp4a[i1][i2][0],tmp4a[i1][i2][1],tmp4a[i1][i2][2],tmp4a[i1][i2][3],tmp4a[i1][i2][4],tmp4a[i1][i2][5])
            dd=np.sqrt(((v4[0]+v4[1]*TAU)/v4[2])**2+((v5[0]+v5[1]*TAU)/v5[2])**2+((v6[0]+v6[1]*TAU)/v6[2])**2)
            if dd>radius:
                radius=dd
            else:
                pass
    return radius

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double distance_in_perp_space(np.ndarray[DTYPE_int_t, ndim=2] pos1,
                             np.ndarray[DTYPE_int_t, ndim=2] pos2):
    cdef int i1
    cdef long w1,w2,w3
    cdef double dd
    #cdef np.ndarray[DTYPE_int_t,ndim=1] v1,v2,v3
    cdef np.ndarray[DTYPE_int_t,ndim=1] v4,v5,v6,tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a
    
    tmp1a=np.array([0])
    for i1 in range(6):
        w1,w2,w3=sub(pos1[i1][0],pos1[i1][1],pos1[i1][2],pos2[i1][0],pos2[i1][1],pos2[i1][2])
        if i1!=0:
            tmp1a=np.append(tmp1a,[w1,w2,w3])
        else:
            tmp1a=np.array([w1,w2,w3])
    tmp2a=tmp1a.reshape(6,3)
    #v1,v2,v3,v4,v5,v6=projection(tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5])
    _,_,_,v4,v5,v6=projection(tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5])
    dd=np.sqrt(((v4[0]+v4[1]*TAU)/v4[2])**2+((v5[0]+v5[1]*TAU)/v5[2])**2+((v6[0]+v6[1]*TAU)/v6[2])**2)
    return dd
