import numpy as np
import cython
from numpy.typing import NDArray
import time # in object_subtraction_dev1, tetrahedron_not_obj
import itertools

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import numeric as num
import qnndarray as qna
import prjop as prj

def ball_radius_obj(obj: qnv.Qnvec, centroid: qnv.Qnvec) -> qnn.Qnnum: #float:
    """estimate maximum distance between verices of given OBJ and its centroid.
    
    Parameters
    ----------
    obj: array (ndim=4)
        in TAU-style
    centroid: array, (ndim=2)
        a n-dimensional coordinates in TAU-style
    
    Returns
    -------
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

def rough_check_intersection_triangle_obj(triangle: qna.QnNdarray,\
         cententer: qnv.Qnvec, distance: qnn.Qnnum) -> bool:
    cen1=centroid(triangle)
    dd1=ball_radius(triangle,cen1)
    dd0=distance_in_perp_space(cen1,cententer)
    if dd0 <= dd1+distance: # two balls are intersecting.
        return True
    else:
        return False

def check_intersection_two_triangles(triangle_1: qna.QnNdarray, triangle_2: qna.QnNdarray) -> int:
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
            if check_intersection_segment_surface_numerical_nd_tau(segment,surface): # intersecting
                counter1+=1
                break
            else:
                pass
            # case 2: intersection between
            # 3 edges of triangle_2
            # 1 surfaces of triangle_1
            segment=np.stack([triangle_2[c[0]],triangle_2[c[1]]])
            surface=triangle_1
            if check_intersection_segment_surface_numerical_nd_tau(segment,surface): # intersecting
                counter1+=1
                break
            else:
                pass
        if counter1>0:
            return 3 # intersecting
        else:
            return 0 # no intersection

def intersection_two_segment(segment_1: qna.QnNdarray, segment_2: qna.QnNdarray) -> qna.QnNdarray:
    """check intersection between two line segments.
    
    Parameters
    ----------
    line_segment_1 line_segment_2: array
        n-dimensional coordinates of line segment,(xyzuvw1, xyzuvw2) and (xyzuvw3, xyzuvw4), in TAU-style
   
    Returns
    -------
    
    """
    # check whether two line segments are intersecting or not by numerical calc.
    if num.check_intersection_two_segment_numerical_nd_tau(segment_1,segment_2): # intersecting
        # calc in TAU-style
        vecAB_nd=segment_1[1]-segment_1[0]   # line segment
        qnv.printqnv("vecAB_nd",vecAB_nd)    # for test
        vecAB=prj.projection3(vecAB_nd)            # AB
        #
        tmp=segment_2[1]-segment_2[0]
        vecCD=prj.projection3(tmp)                 # CD
        #
        #tmp=sub_vectors(segment_1[0],segment_2[0])
        #vecCA=projection3(tmp)                    # CA
        #
        tmp=segment_2[0]-segment_1[0]
        vecAC=prj.projection3(tmp)                 # AC
        
        bunbo=qnv.dot(vecAB,vecCD)*qnv.dot(vecCD,vecAB)-qnv.dot(vecAB,vecAB)*qnv.dot(vecCD,vecCD)
        #tmp1=qnv.qnv.dot(vecAB,vecCD)
        #tmp2=qnv.qnv.dot(vecCD,vecAB)
        #tmp3=tmp1*tmp2  #mul(tmp1,tmp2)
        #
        #tmp1=qnv.qnv.dot(vecAB,vecAB)
        #tmp2=qnv.qnv.dot(vecCD,vecCD)
        #tmp4=tmp1*tmp2 #mul(tmp1,tmp2)
        #bunbo=tmp3-tmp4  #sub(tmp3,tmp4)
        
        ## ????
        #if bunbo[0]==0 and bunbo[1]==bunbo:
        #    return 
        #else:
        bunshi=qnv.dot(vecAC,vecCD)*qnv.dot(vecCD,vecAB)-qnv.dot(vecCD,vecCD)*qnv.dot(vecAC,vecAB)
        #tmp1=qnv.qnv.dot(vecAC,vecCD)
        #tmp2=qnv.qnv.dot(vecCD,vecAB)
        #tmp3=tmp1*tmp2  #mul(tmp1,tmp2)
        ##
        #tmp1=qnv.qnv.dot(vecCD,vecCD)
        #tmp2=qnv.qnv.dot(vecAC,vecAB)
        #tmp4=tmp1*tmp2  #mul(tmp1,tmp2)
        #bunshi=tmp3-tmp4
        if bunbo==qnn.zero():
            return
        s=bunshi/bunbo
        qnn.printqnn("s",s)  # for test
        #s=div(bunshi,bunbo)
        #
        # OP = OA + s*AB
        tmp=qnm.mul_scl(vecAB_nd,s) # vec tunes scale
        qnv.printqnv("tmp",tmp)
        return segment_1[0]+tmp
            
    else: # no intersection
        return 

def intersection_segment_surface(segment: qna.QnNdarray, surface: qna.QnNdarray) -> qna.QnNdarray:
    """check intersection between a line segment and a triangle.
    
    Möller–Trumbore intersection algorithm
    https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    
    Parameters
    ----------
    line_segment: array
        n-dimensional coordinates of line segment,xyzuvw1, xyzuvw2, in TAU-style
    triangle: array
        containing n-dimensional coordinates of tree vertecies of a triangle, xyzuvw1, xyzuvw2, xyzuvw3, xyzuvw4 in TAU-style
    
    Returns
    -------
    
    """
    # check whether the line segment and the surface are intersecting or not by numerical calc.
    if check_intersection_segment_surface_numerical_nd_tau(segment,surface): # intersecting
        
        """
        # calc in TAU-style
        vec6AB=sub_vectors(segment[1],segment[0])
        vecAB=projection3(vec6AB)              # AB # R
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
        n=crs.n
        return add_vectors(segment[0],tmp).reshape(1,n,3)
        """
        #  edge: 0-1,0-2,1-2
        comb=[[0,1],[0,2],[1,2]]
        counter=0
        n=crs.n
        for j in comb:
            segment1=np.vstack([surface[j[0]],surface[j[1]]])
            tmp1=intersection_two_segment(segment,segment1.reshape(2,n,3))
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
    
def intersection_two_triangles(triangle_1: qna.QnNdarray, triangle_2: qna.QnNdarray) -> qna.QnNdarray:
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
        segment=np.stack([triangle_1[c[0]],triangle_1[c[1]]]) # ???
        surface=np.stack([triangle_2[c[2]],triangle_2[c[3]],triangle_2[c[4]]])
        vtx=intersection_segment_surface(segment,surface)
        if np.all(vtx==None):
            pass
        else:
            if counter==0 :
                tmp=vtx # intersection points
            else:
                tmp=np.vstack([tmp,vtx]) # intersecting points
            counter+=1
        # case 2: intersection between (edge of triangle_2) and (surface of triangle_1)
        segment=np.stack([triangle_2[c[0]],triangle_2[c[1]]])
        surface=np.stack([triangle_1[c[2]],triangle_1[c[3]],triangle_1[c[4]]])
        vtx=intersection_segment_surface(segment,surface)
        if np.all(vtx==None):
            pass
        else:
            if counter==0:
                tmp=vtx # intersection points
            else:
                tmp=np.vstack([tmp,vtx]) # intersecting points
            counter+=1
    n=crs.n
    tmp=tmp.reshape(int(len(tmp)/n),n,3)
    
    # get vertces of triangle_1 that are inside triangle_2
    n=crs.n
    for vtx in triangle_1:
        if inside_outside_triangle_tau(vtx,triangle_2): # inside
            if counter==0:
                tmp=vtx.reshape(1,n,3)
            else:
                tmp=np.vstack([tmp,[vtx]])
            counter+=1
    # get vertces of triangle_2 that are inside triangle_1
    n=crs.n
    for vtx in triangle_2:
        if inside_outside_triangle_tau(vtx,triangle_1): # inside
            if counter==0:
                tmp=vtx.reshape(1,n,3)
            else:
                tmp=np.vstack([tmp,[vtx]])
            counter+=1
        else:
            pass
    
    if counter>=3:
        n=crs.n
        tmp=remove_doubling_in_perp_space(tmp)
        if len(tmp)>3:
            tmp4=triangulation_points(tmp)
            if np.all(tmp4==None):
                return 
            else:
                return tmp4
        elif len(tmp)==3:
            return tmp.reshape(1,3,n,3)
        else:
            return 
    else:
        return 

def intersection_two_obj_1(obj1: qnv.Qnvec,obj2: qnv.Qnvec,select=None,verbose: int=0) -> qnv.Qnvec:
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
    n=crs.n
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
                # tetrahedron_1 is fully inside triangle2
                if flag==1:
                    if counter0==0:
                        common4=triangle1.reshape(1,3,n,3)
                        counter0+=1
                    else:
                        common4=np.vstack([common4,[triangle1]])
                    break
                #
                # tetrahedron_2 is fully inside triangle1
                elif flag==2:
                    if counter1==0:
                        tmp_common4=triangle2.reshape(1,3,n,3)
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
                        #v=obj_volume_nd(tmp4)
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
                #vol2=obj_area_nd(tmp_common4)
                #print('vol2',vol2,numeric_value(vol2))
                if select=='simple':
                    vol1=triangle_area_nd(triangle1)
                    vol2=obj_area_nd(tmp_common4)
                    if np.all(vol1==vol2):
                        if counter0==0:
                            common4=triangle1.reshape(1,3,n,3)
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

def intersection_two_obj_convex(obj1: qnv.Qnvec, obj2: qnv.Qnvec, verbose: int=0) -> qnv.Qnvec:
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
    n=crs.n
    for vrtx in vertices1:
        counter2a=0
        for triangle2 in obj2:
            if inside_outside_ttriangle_tau(vrtx,triangle2):
                counter2a+=1
                break
            else:
                pass
        if counter2a>0:
            if counter1a==0:
                point_a1=vrtx.reshape(1,n,3)
            else:
                point_a1=np.vstack([point_a1,[vrtx]])
            counter1a+=1
        #else:
        #    if counter2==0:
        #        point_b2=vrtx.reshape(1,n,3)
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
    n=crs.n
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
                point_b1=vrtx.reshape(1,n,3)
            else:
                point_b1=np.vstack([point_b1,[vrtx]])
            counter1b+=1
        #else:
        #    if counter2==0:
        #        point_a2=vrtx.reshape(1,n,3)
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
                if check_intersection_segment_surface_numerical_nd_tau(ed2,tr1): # intersection
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
                if check_intersection_segment_surface_numerical_nd_tau(ed1,tr2): # intersection
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
            n=crs.n
            point1=remove_doubling_in_perp_space(p.reshape(int(len(p)/n),n,3))
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

#########
###
###   WIP
###
#########
def subtraction_two_obj(obj1: qnv.Qnvec, obj2: qnv.Qnvec, verbose: int=0) -> qnv.Qnvec:
    """Operate A not B (= A NOT (A AND B)).

    Parameters
    ----------
    n=crs.n
    obj1: array,(number of triangles, 3, n, 3)
        Object A to be subtracted.
    obj2: array, (number of triangles, 3, n, 3)
        Object B that subtracts the triangle.
    verbose: int
    
    Returns
    -------
    obj: array, (number of triangles, 3, n, 3)
    
    """
    
    if verbose>0:
        print('      generating surface_obj2')
        start=time.time()
    #
    #surface_obj2=generator_surface_1(obj2,verbose-1)
    #
    if verbose>0:
        end=time.time()
        time_diff=end-start
        print('         ends in %4.3f sec'%time_diff)
    
    if verbose>0:
        print('      tetrahedron_not_obj_1 starts...')
        start=time.time()
    #
    
    flag=0
    out=None
    counter1=0
    n=crs.n
    for triangle in obj1:
        if verbose>0:
            print('       %d-th triangle in obj1'%(counter1))
        a=triangle_not_obj_1(triangle.reshape(1,3,n,3),obj2,verbose)
        if np.all(a==None):
            out=None
            flag=1
            break
        else:
            if counter1==0:
                out=a
            else:
                out=np.vstack([out,a])
        counter1+=1
    
    if flag==0:
        if verbose>0:
            end=time.time()
            time_diff=end-start
            print('         ends in %4.3f sec'%time_diff)
    return out

def triangle_not_obj_1(triangle: qnv.Qnvec, obj: qnv.Qnvec, verbose: int=0) -> qnv.Qnvec:
    """Operate tetrahedron not object = tetrahedron not (triangle and object).
    
    Parameters
    ----------
    n=crs.n
    triangle: array, (1, 4, n, 3)
        Triangle to be subtracted.
    obj: array, (number of triangles, 3, n, 3)
        Object that subtracts the triangle.
    surface_obj: array, (number of triangles, 3, n, 3)
        Surface trianges of the object.
    verbose: int
    
    Returns
    -------
    obj: array, (number of triangles, 3, n, 3)
    
    Note
    ----
    Current implementation may return wrong object when the intersecting between the objects is not simple.
    
    """
    
    #print('        triangle_not_obj_1()')
        
    # surface triangles and vertices of common
    #print('         intersection_two_obj_1()')
    #start=time.time()
    common=intersection_two_obj_1(triangle,obj)
    #end=time.time()
    #time_diff=end-start
    #print('          ends in %4.3f sec'%time_diff)
    #surface_common=generator_surface_1(common,verbose-1)=
    surface_common=common
    #vertx_common=remove_doubling_in_perp_space(surface_common)
    
    vol0=obj_volume_nd(triangle)
    vol1=obj_volume_nd(common)
    vol2=sub(vol0,vol1)
    if verbose>0:
        print('        triangle volume:',vol0,numeric_value(vol0))
        print('        common volume:',vol1,numeric_value(vol1))
        print('        triangle NOT obj:',vol2,numeric_value(vol2))
    
    out=None
    
    # get surface triangles of common part which are on the surface of obj
    ################################################
    # 問題点
    # ここではtriangle NOT objは以下の2点からなると想定している。
    # (1) objに含まれないtriangle頂点と、
    # (2) triangle AND objの三角形のうち、objの表面にある三角形
    # しかし、tetrahedron AND objがobj自身である場合など、tetrahedronとobjが
    # ほとんど重なっている場合、必ずしも上記(2)が求めたい頂点のみを含むとは限らず、
    # 余計なもまで作ってしまう。
    ################################################
    counter2=0
    n=crs.n
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
                tmp=triangle2.reshape(1,3,n,3)
            else:
                tmp=np.vstack([tmp,[triangle2]])
            counter2+=1
        else:
            pass
    triangle_common=tmp
    #print('triangle_common.shape',triangle_common.shape)
    
    # get vertices of tetrahedron which are NOT inside obj
    counter2=0
    n=crs.n
    tetrahedron=tetrahedron.reshape(4,n,3)
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
                tmp=vrtx1.reshape(1,n,3)
            else:
                tmp=np.vstack([tmp1a,[vrtx1]])
            counter2+=1
        else:
            pass
    vrtx1_out=tmp
    #print('vrtx1_out.shape',vrtx1_out.shape)
    
    
    
    # 四面体の4つの頂点のうちobjの外側にある頂点vrtx1_outの個数Nは、1,2,3,4。
    # 以下、それぞれの場合について考える。
    ############################################################ 
    ### N=1の場合は、triangle_commonにある各三角形と頂点を結んだものが求めたいものになる。
    ############################################################
    if counter2==1:
        if verbose>0:
            print('         case 1')
        counter3=0
        
        
        #---------------------------------------
        # triangle_commonにある各三角形の重心を求め、
        # vrtx1_outからの距離が近い順にソートする。
        #---------------------------------------
        num=len(triangle_common)
        #print('len(triangle_common)',num)
        dd=np.zeros(num,dtype=np.float_)
        n=crs.n
        i1=0
        for triangle in triangle_common:
            vt=centroid(triangle)
            #print('vt.shape',vt.shape)
            #print('vt',vt)
            vn=get_internal_component_numerical(vt)
            #print('vrtx1_out.shape',vrtx1_out.shape)
            #print('vrtx1_out',vrtx1_out)
            vn_out=get_internal_component_numerical(vrtx1_out.reshape(n,3))
            vn=vn-vn_out
            #print('vn',vn)
            dd[i1]=np.sqrt(vn[0]**2+vn[1]**2+vn[2]**2)
            i1+=1
        #print('dd',dd)
        indx_dd=np.argsort(dd)
        #print('indx_dd',indx_dd)
        tmp=np.zeros((num,3,n,3),dtype=np.int64)
        for i1 in range(len(indx_dd)):
            tmp[i1]=triangle_common[indx_dd[i1]]
        triangle_common=tmp
        
        #print('triangle_common.shape',triangle_common.shape)
        for triangle in triangle_common:
            #print('triangle.shape',triangle.shape)
            tet=np.vstack([vrtx1_out,triangle]).reshape(1,4,n,3)
            #print('tet.shape',tet.shape)
            if counter3==0:
                tmp=tet
            else:
                tmp=np.vstack([tmp,tet])
            counter3+=1
        #print('tmp.shape',tmp.shape)
        vol=obj_volume_nd(tmp)
        if verbose>1:
            print('    obtained volume:',vol,numeric_value(vol))
        if np.all(vol==vol2):
            out=tmp
        else:
            ################################################
            # 上記の問題点があるため、不要なものを除く必要がある。
            # 以下では、tmpにある四面体の集合について、可能な組み合わせ
            # のうち正しい体積になるものを見つける。
            # 体積が正しいからといって、求めたいものではないは可能性ある。
            #
            # triangle_commonの中のtrianglenをvrtx1_outと近い順に
            # ソートておいた方が早く目的のtetrahedronが見つかるはず。
            # 上でソートておいた。
            ################################################
            #print('len(tmp)',len(tmp))
            vol=np.array([0,0,1])
            lst=list(range(0,len(tmp)))
            for num in range(1,len(tmp)-1):
                flag=0
                for comb in list(itertools.combinations(lst,num)):
                    #print(comb)
                    for i1 in range(num):
                        v=tetrahedron_volume_nd(tmp[comb[i1]])
                        #print('    ',v)
                        vol=add(vol,v)
                    if np.all(vol==vol2):
                        flag=1
                        break
                    else:
                        pass
                if flag==1:
                    for i1 in range(num):
                        if i1==0:
                            tmp1=tmp[comb[i1]].reshape(1,4,n,3)
                        else:
                            tmp1=np.vstack([tmp1,[tmp[comb[i1]]]])
                    break
                else:
                    pass
            if flag==1:
                out=tmp1
                vol=obj_volume_nd(out)
                if verbose>1:
                    print('    obtained volume:',vol,numeric_value(vol))
                if verbose>0:
                    print('      succeeded.')
            else:
                if verbose>0:
                    print('      unsucceeded.')
                out=None
            ################################################
            #                                              #
            #                                              #
            #                                              #
            ################################################
        return out
    ############################################################
    #### N=2の場合。triangle_commonに含まれる三角形の個数で場合分けする。
    ############################################################
    elif counter2==2:
        if len(triangle_common)==1:
            if verbose>0:
                print('         case 2-1')
            out=tetrahedralization_points(np.vstack([vrtx1_out,triangle_common]))
        elif len(triangle_common)==2:
            if verbose>0:
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
            vola=obj_volume_nd(tet1)
            combination=[\
            [1,0],\
            [0,1]]
            for c1 in combination: 
                tet2=np.vstack([vrtx1_out[c1[0]],tr1])
                tet3=np.vstack([vrtx1_out[c1[1]],tr2])
                tet_tot=np.vstack([tet1,tet2,tet3])
                vol_tot=obj_volume_nd(tet_tot)
                if np.all(vol_tot==vol2):
                    out=tet_tot
                    break
        elif len(triangle_common)==3:
           if verbose>0:
                print('         case 2-3')
        elif len(triangle_common)==4:
            if verbose>0:
                print('         case 2-4')
        else:
            if verbose>0:
                print('         case 2-X')
        return out
    ############################################################
    #### N=3の場合。triangle_commonに含まれる三角形の個数で場合分けする。
    ############################################################
    elif counter2==3:
        if len(triangle_common)==1:
            if verbose>0:
                print('         case 3-1')
            out=tetrahedralization_points(np.vstack([vrtx1_out,triangle_common]))
        elif len(triangle_common)==2:
            if verbose>0:
                print('         case 3-2')
        elif len(triangle_common)==3:
            if verbose>0:
                print('         case 3-3')
        elif len(triangle_common)==4:
            if verbose>0:
                print('         case 3-4')
        else:
            if verbose>0:
                print('         case 3-X')
        return out
    ############################################################
    #### N=4の場合。triangle_commonに含まれる三角形の個数で場合分けする。
    ############################################################
    else:
        if len(triangle_common)==3:
            if verbose>0:
                print('         case 4-3')
        elif len(triangle_common)==4:
            if verbose>0:
                print('         case 4-4')
        else:
            if verbose>0:
                print('         case 4-X')
        return out

def tetrahedron_not_obj_2(tetrahedron: qnv.Qnvec, obj: qnv.Qnvec) -> qnv.Qnvec:
    """Operate tetrahedron not object = tetrahedron not (tetrahedron and object).
    
    tetrahedronからobjを引いた物体の表面にある三角形を求める（本当は四面体を求めたいが難しい）
    アルゴリズム
    1. tetrahedronとobjの共通部分Aを求める。
    2. A表面の三角形T1を求める
    3. objの表面の三角形T2を求める。
    4. T1のうち、T2上にあるものT1'を得る。T1'は求めたい差分tetrahedron NOT objの表面の一部になる。
    5. T1のうち、T2の外側にあるものとT1'を合わせる。
    
    Parameters
    ----------
    n=crs.n
    tetrahedron: array, (1, 4, n, 3)
        Tetrahedron to be subtracted.
    obj: array, (number of tetrahedra, 4, n, 3)
        Object that subtracts the tetrahedron.
    
    Returns
    -------
    obj: array, (number of tetrahedra, 4, n, 3)
    
    Note
    ----
    Under development.
    
    """
    
    vol0=obj_volume_nd(tetrahedron)
    print('tetrahedron volume:',vol0,numeric_value(vol0))
    
    # 1. tetrahedronとobjの共通部分Aを求める。
    # intersection between tetrahedron and obj
    common=intersection_two_obj_1(tetrahedron,obj,kind='standard')
    vol1=obj_volume_nd(common)
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
