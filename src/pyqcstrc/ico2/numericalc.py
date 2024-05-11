#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import numpy as np
from numpy.typing import NDArray
import random

TAU=(1+np.sqrt(5))/2.0
EPS=1e-6

def coplanar_check_numeric_tau(pts: NDArray[np.int64], num_iteration: int=5) -> bool:
    """check the points (pts) are in coplanar or not
    
    Parameters
    ----------
    pns: array
        6d coordinates of the points, xyz, in TAU-style
    num_iteration: int
        number of iterations.
    
    Returns
    -------
    bool
    
    """
    p=get_internal_component_sets_numerical(pts)
    return coplanar_check_numeric(p,num_iteration)

def coplanar_check_numeric(pns: NDArray[np.float_],num_iteration: int=5) -> bool:
    """check the points (pns) are in coplanar or not
    メモ：xyz1とxyz2の選び方次第で、outer_product(v1,v2)が小さくなりcoplanarと間違って判定する場合がある。
    これを避けるために適切なxyz1とxyz2の選び方が必要。以下では、ランダムにxyz1とxyz2の選ぶ。
    
    Parameters
    ----------
    pns: array
        coordinate of the points in Eperp, xyz.
    num_iteration: int
        number of iterations.
    
    Returns
    -------
    bool
    """
    num=len(pns)
    if num>3:
        flag=0
        lst0=[i for i in range(num)]
        for _ in range(num_iteration):
            lst3=random.sample(lst0, 3)
            #
            xyz1=pns[lst3[1]]-pns[lst3[0]]
            xyz2=pns[lst3[2]]-pns[lst3[0]]
            vec=np.cross(xyz1,xyz2)
            flg=0
            if np.all(abs(vec)<EPS):
                pass
            else:
                flag=1
                break
        if flag==1:
            counter=0
            lst=list(filter(lambda x: x not in lst3, lst0))
            for i in lst:
            #for i in list(filter(lambda x: x not in lst3, lst0)):
                xyzi=pns[i]-pns[lst3[0]]
                if abs(np.dot(vec,xyzi))<1e-10:
                    pass
                else:
                    counter=1
                    break
            if counter==0:
                return True
            else:
                return False
        else:
            'error in coplanar_check_numeric. increase num_iteration.'
            return 
    else:
        return True

def point_on_segment(point: NDArray[np.int64], line_segment: NDArray[np.int64]) -> bool:
    """judge whether a point is on a line segment, A-B, or not.
    
    Parameters
    ----------
    point: array
        coordinate of the point, xyz
    line_segment: array
        two coordinates of line segment, xyz0, xyz1
    
    Returns
    -------
    int
    """
    
    point=numerical_vector(point)
    ln=numerical_vectors(line_segment)
    
    xyx0=projection3_numerical(point)
    tmp=projection3_sets_numerical(ln)
    xyx1=tmp[0]
    xyx2=tmp[1]
    
    vecPA=xyx0-xyx1 # np.array([x0-x1,y0-y1,z0-z1])
    vecBA=xyx2-xyx1 # np.array([x2-x1,y2-y1,z2-z1])
    lPA=np.linalg.norm(vecPA)
    lBA=np.linalg.norm(vecBA)
    if lBA>0.0 and abs(np.dot(vecPA,vecBA)-lPA*lBA)<EPS:
        s=lPA/lBA
        if s>=0.0 and s<=1.0:
            return True
        elif s>1.0:
            return False #       A==B P
        else:
            return False #    P A==B
    else:
        return False

def on_out_surface(point: NDArray[np.int64], triangle: NDArray[np.int64]) -> bool:
    """
    check whether the point is inside the triangle.
    
    Parameters
    ----------
    point: array
        6d coordinates of the point in TAU-style.
    triangle: array
        6d coordinates of three vertices of triangle in TAU-style
    
    Returns
    -------
    float
    """
    
    def func(p_xyz,tr_xyz,indx):
        out=np.zeros((3,3),dtype=np.float64)
        for i in range(3):
            if i==indx:
                out[i]=p_xyz
            else:
                out[i]=tr_xyz[i]
        return out
        
    p=get_internal_component_numerical(point)
    triangle0=get_internal_component_sets_numerical(triangle)
    
    area0=triangle_area_numerical(triangle0)
    
    triangle1=func(p,triangle0,0)
    triangle2=func(p,triangle0,1)
    triangle3=func(p,triangle0,2)
    
    area1=triangle_area_numerical(triangle1)+triangle_area_numerical(triangle2)+triangle_area_numerical(triangle3)
    
    if abs(area0-area1)< EPS:
        return True
    else:
        return False

def numeric_value(t: NDArray[np.int64]) -> float:
    """Numeric value of a TAU-style value, a.

    Parameters
    ----------
    t: array
        value in TAU-style
    
    Returns
    -------
    float
    """
    return (t[0]+t[1]*TAU)/t[2]

def numerical_vector(vt: NDArray[np.int64]) -> NDArray[np.int64]:
    """Numeric value of a TAU-style vector, v.

    Parameters
    ----------
    vt: array
        vector in TAU-style
    
    Returns
    -------
    array
    """
    n=len(vt)
    w=np.zeros(n,dtype=np.float64)
    for i in range(n):
        w[i]=numeric_value(vt[i])
    return w

def numerical_vectors(vts: NDArray[np.int64]) -> NDArray[np.int64]:
    """Numeric value of a TAU-style vector, v.

    Parameters
    ----------
    vts: array
        vector in TAU-style
    
    Returns
    -------
    array
    """
    n1,n2,_=vts.shape
    w=np.zeros((n1,n2),dtype=np.float64)
    for i1 in range(n1):
        w[i1]=numerical_vector(vts[i1])
    return w

def length_numerical(vt: NDArray[np.int64]) -> float:
    """numerical value of norm of vector, v, in Tau-style
    
    Parameters
    ----------
    vt: array
        vector in TAU-style
    
    Returns
    -------
    
    """
    vn=numerical_vector(vt)
    return np.linalg.norm(vn)
    #return np.sqrt(np.sum(np.abs(v**2)))



def check_intersection_segment_surface_numerical_6d_tau(line_segment: NDArray[np.int64], triangle: NDArray[np.int64]) -> bool:
    """check intersection between a line segment and a triangle.
    
    Parameters
    ----------
    line_segment: array
        6-dimensional coordinates of line segment,xyzuvw1, xyzuvw2, in TAU-style
    triangle: array
        containing 6-dimensional coordinates of tree vertecies of a triangle (a) in TAU-style
    
    Returns
    -------
    
    """
    ln=get_internal_component_sets_numerical(line_segment)
    tr=get_internal_component_sets_numerical(triangle)
    return check_intersection_segment_surface_numerical(ln,tr)
    
def check_intersection_segment_surface_numerical(ln: NDArray[np.float_], tr: NDArray[np.float_]) -> bool:
    """check intersection between a line segment and a triangle.
    
    Möller–Trumbore intersection algorithm
    https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    
    Parameters
    ----------
    line_segment: array
        two 3-dimensional coordinates of line segment, xyz1, xyz2.
    triangle: array
        containing 3-dimensional coordinates of tree vertecies of a triangle (a), xyz1, xyz2, xyz3.
    
    Returns
    -------
    
    """
    vecAB=ln[1]-ln[0] # AB # R
    vecCD=tr[1]-tr[0] # CD # E1
    vecCE=tr[2]-tr[0] # CE # E2
    vecCA=ln[0]-tr[0] # CA # T
    
    vecP=np.cross(vecAB,vecCE) # P
    vecQ=np.cross(vecCA,vecCD) # Q
    
    bunbo=np.dot(vecP,vecCD)
    #print('bunbo=',bunbo)
    if abs(bunbo)<EPS: # the line_segment is parrallel to the triangle.
        return False
    else:
        u=np.dot(vecP,vecCA)/bunbo
        #print('u=',u)
        if u>=0.0-EPS and u<=1.0+EPS:
            v=np.dot(vecQ,vecAB)/bunbo
            #print('v=',v)
            if v>=0.0-EPS and u+v<=1.0+EPS:
                t=np.dot(vecQ,vecCE)/bunbo
                #print('t=',t)
                if t>=0.0-EPS and t<=1.0+EPS:
                    #print('  intersect\n')
                    return True # intersect
                else:
                    #print('\n')
                    return False
            else:
                #print('\n')
                return False
        else:
            #print('\n')
            return False

def check_intersection_two_segment_numerical_6d_tau(segment_1: NDArray[np.int64], segment_2: NDArray[np.int64]) -> bool:
    """check intersection between two line segments
    
    Parameters
    ----------
    line_segment_1,line_segment_2 : array
        two 6-dimensional coordinates of line segment,xyzuvw1, xyzuvw2, in TAU-style
    triangle: array
        containing 3-dimensional coordinates of tree vertecies of a triangle (a), xyz1, xyz2, xyz3.
    
    Returns
    -------
    
    """
    # TAU-style to Float
    #segment_1=numerical_vectors(segment_1)
    #segment_2=numerical_vectors(segment_2)
    
    ln1=get_internal_component_sets_numerical(segment_1)
    ln2=get_internal_component_sets_numerical(segment_2)
    return check_intersection_two_segment_numerical(ln1,ln2)

def check_intersection_two_segment_numerical(ln1: NDArray[np.float_], ln2: NDArray[np.float_]) -> bool:
    """check intersection between two line segments.
    
    Parameters
    ----------
    line_segment_1,line_segment_2: array
        two 3-dimensional coordinates of line segment, xyz1, xyz2.
        line_segment_1: A--B
        line_segment_2: C--D
    
    Returns
        int, out = 0 (Intersection was found when a view allong to Z-axis)
                   1 (Intersection was found when a view allong to X-axis)
                   2 (Intersection was found when a view allong to Y-axis)
                   3 (No intersection was found)
    -------
    
    """
    vecAB=ln1[1]-ln1[0] # AB
    vecCD=ln2[2]-ln2[0] # CD
    vecCA=ln1[0]-ln2[1] # CA
    
    # check whether two line-segments are intersecting or not.
    comb=[\
    [0,1,2],\
    [1,2,0],\
    [2,0,1]]
    out=0
    for c in comb:
        bunbo=vecAB[c[0]]*vecCD[c[1]]-vecCD[c[0]]*vecAB[c[1]]
        if abs(bunbo)<EPS:
            t=(vecAB[c[0]]*vecCA[c[1]]-vecCA[c[0]]*vecAB[c[1]])/bunbo
            if t>=0.0 and t<=1.0:
                s=(vecCD[c[0]]*vecCA[c[1]]-vecCA[c[0]]*vecCD[c[1]])/bunbo
                if s>=0.0 and s<=1.0:
                    if abs(-s*vecAB[c[2]]+t*vecCD[c[2]]-vecCA[c[2]])<=EPS:
                        break
                    else:
                        out+=1
                else:
                    out+=1
            else:
                out+=1
        else:
            out+=1
    return out
    
def triangle_area(a: NDArray[np.int64]) -> float:
    """Numerial calcuration of area of given triangle, a.
    The coordinates of the tree vertecies of the triangle are given in TAU-style.
    
    Parameters
    ----------
    a: array containing 3-dimensional coordinates of tree vertecies of a triangle (a) in TAU-style
    
    Returns
    -------
    area of given triangle: float
    """
    
    x1=numeric_value(a[1][0])-numeric_value(a[0][0])
    y1=numeric_value(a[1][1])-numeric_value(a[0][1])
    z1=numeric_value(a[1][2])-numeric_value(a[0][2])
    
    x2=numeric_value(a[2][0])-numeric_value(a[0][0])
    y2=numeric_value(a[2][1])-numeric_value(a[0][1])
    z2=numeric_value(a[2][2])-numeric_value(a[0][2])
    
    v1=np.array([x1,y1,z1])
    v2=np.array([x2,y2,z2])
    
    v3=np.cross(v2,v1) # cross product
    return np.sqrt(np.sum(np.abs(v3**2)))/2.0

def triangle_area_numerical(a: NDArray[np.float_]) -> float:
    """Numerial calcuration of area of given triangle, a.
    The coordinates of the tree vertecies of the triangle are given.
    
    Parameters
    ----------
    a: array containing 3-dimensional coordinates of tree vertecies of a triangle (a)
    
    Returns
    -------
    area of given triangle: float
    """
    
    x1=a[1][0]-a[0][0]
    y1=a[1][1]-a[0][1]
    z1=a[1][2]-a[0][2]
    
    x2=a[2][0]-a[0][0]
    y2=a[2][1]-a[0][1]
    z2=a[2][2]-a[0][2]
    
    v1=np.array([x1,y1,z1])
    v2=np.array([x2,y2,z2])
    
    v3=np.cross(v2,v1) # cross product
    return np.sqrt(np.sum(np.abs(v3**2)))/2.0

def inside_outside_obj_tau(point: NDArray[np.int64], obj: NDArray[np.int64]) -> bool:
    
    # TAU-style to Float
    point=numerical_vector(point)
    obj=numerical_vectors(obj)
    # 
    point=get_internal_component_numerical(ln)
    obj=get_internal_component_sets_numerical(obj)
    return inside_outside_obj(point,obj)
    
def inside_outside_obj(point: NDArray[np.float_], obj: NDArray[np.float_]) -> bool:
    """this function judges whether the point is inside an object (set of tetrahedra) or not
        
    Parameters
    ----------
    point: array
        coordinate of the point,xyz
    obj: array
        vertex coordinates of tetrahedra, (xyz1, xyz2, xyz3, xyz4), (), (), ...
    """
    flg=0
    for tetrahedron in obj:
        if inside_outside_tetrahedron(point,tetrahedron):
            flg+=1
            break
    if flg==0:
        return True # inside
    else:
        return False # outside

def inside_outside_tetrahedron_tau(point: NDArray[np.int64], tetrahedron: NDArray[np.int64]) -> bool:
    """this function judges whether the point is inside a tetrahedron or not
        
    Parameters
    ----------
    point: array
        6d coordinate of the point in TAU-style.
    tetrahedron: array
        6d vertex coordinates of tetrahedron in TAU-style.
    """
    #point=numerical_vector(point)
    #tetrahedron=numerical_vectors(tetrahedron)
    # 
    point=get_internal_component_numerical(point)
    tetrahedron=get_internal_component_sets_numerical(tetrahedron)
    return inside_outside_tetrahedron(point,tetrahedron)

def inside_outside_tetrahedron(point: NDArray[np.float_], tetrahedron: NDArray[np.float_]) -> bool:
    """this function judges whether the point is inside a tetrahedron or not
        
    Parameters
    ----------
    point: array
        coordinate of the point,xyz
    tetrahedron: array
        vertex coordinates of tetrahedron, (xyz1, xyz2, xyz3, xyz4)
    """
    vol0=tetrahedron_volume_numerical(tetrahedron)
    
    def small_tetrahedron(indx,p,tetrahedron0):
        tet=np.zeros((4,3),dtype=np.float64)
        for i in range(4):
            if i==indx:
                tet[i]=p
            else:
                tet[i]=tetrahedron0[i]
        return tet
    
    tet1=small_tetrahedron(0,point,tetrahedron)
    vol1=tetrahedron_volume_numerical(tet1)
    #
    tet2=small_tetrahedron(1,point,tetrahedron)
    vol2=tetrahedron_volume_numerical(tet2)
    #
    tet3=small_tetrahedron(2,point,tetrahedron)
    vol3=tetrahedron_volume_numerical(tet3)
    #
    tet4=small_tetrahedron(3,point,tetrahedron)
    vol4=tetrahedron_volume_numerical(tet4)
    
    if abs(vol0-vol1-vol2-vol3-vol4)<EPS*vol0:
        return True # inside
    else:
        return False # outside






def obj_volume_6d_numerical(obj: NDArray[np.int64]) -> float:
    """This function returns volume of an object (set of tetrahedra).
        
    Parameters
    ----------
    object: array
        6-dimensional vertex coordinates of tetrahedra.
    """
    vol=0
    for tetrahedron in obj:
        vol+=tetrahedron_volume_6d_numerical(tetrahedron)
    return vol

def tetrahedron_volume_6d_numerical(tetrahedron: NDArray[np.int64]) -> float:
    """This function returns volume of a tetrahedron
        
    Parameters
    ----------
    tetrahedron: array
        6-dimensional vertex coordinates of the tetrahedron, xyzuvw0,xyzuvw1,xyzuvw2,xyzuvw3
    """
    a=get_internal_component_sets_numerical(tetrahedron)
    return tetrahedron_volume_numerical(a)

def obj_volume_numerical(obj: NDArray[np.float_]) -> float:
    """This function returns volume of an object (set of tetrahedra).
        
    Parameters
    ----------
    object: array
        3-dimensional vertex coordinates of tetrahedra.
    """
    vol=0
    for tetrahedron in obj:
        vol+=tetrahedron_volume_numerical(tetrahedron)
    return vol

def tetrahedron_volume_numerical(tetrahedron: NDArray[np.float_]) -> float:
    """This function returns volume of a tetrahedron
        
    Parameters
    ----------
    tetrahedron: array
        vertex coordinates of the tetrahedron, xyz0,xyz1,xyz2,xyz3
    """
    xyz=np.zeros((3,3),dtype=np.float64)
    for i in range(3):
        xyz[i]=tetrahedron[i+1]-tetrahedron[0]
    detm = np.linalg.det(xyz)
    return abs(detm)/6.0










def get_internal_component_numerical(vt: NDArray[np.int64]) -> NDArray[np.float_]:
    """
    Parameters
    ----------
    vn: array
        6-dimensional vector, xyzuvw.
    """
    vn=numerical_vector(vt)
    return projection3_numerical(vn)

def get_internal_component_sets_numerical(vts: NDArray[np.int64]) -> NDArray[np.float_]:
    """parallel and perpendicular components of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    vns=numerical_vectors(vts)
    return projection3_sets_numerical(vns)

def projection_numerical(vn: NDArray[np.float_]) -> NDArray[np.float_]:
    """parallel and perpendicular components of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vn: array
        6-dimensional vector, xyzuvw.
    """
    v1 =  (vn[0]-vn[4]) + TAU*(vn[1]+vn[2]) # x in Epar
    v2 =  (vn[3]+vn[5]) + TAU*(vn[0]+vn[4]) # y in Epar
    v3 =  (vn[1]-vn[2]) - TAU*(vn[3]-vn[5]) # z in Epar
    v4 = -(vn[1]+vn[2]) + TAU*(vn[0]-vn[4]) # x in Eperp
    v5 = -(vn[0]+vn[4]) + TAU*(vn[3]+vn[5]) # y in Eperp
    v6 =  (vn[3]-vn[5]) + TAU*(vn[1]-vn[2]) # z in Eperp
    return np.array([v1,v2,v3,v4,v5,v6],dtype=np.float64)

def projection_sets_numerical(vns: NDArray[np.float_]) -> NDArray[np.float_]:
    """parallel and perpendicular components of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    num=len(vns)
    m=np.zeros((num,6),dtype=np.float64)
    for i in range(num):
        m[i]=projection_numerical(vns[i])
    return m
    
def projection3_numerical(vn: NDArray[np.float_]) -> NDArray[np.float_]:
    """perpendicular component of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vn: array
        6-dimensional vector, xyzuvw.
    """
    v4 = -(vn[1]+vn[2]) + TAU*(vn[0]-vn[4]) # x in Eperp
    v5 = -(vn[0]+vn[4]) + TAU*(vn[3]+vn[5]) # y in Eperp
    v6 =  (vn[3]-vn[5]) + TAU*(vn[1]-vn[2]) # z in Eperp
    return np.array([v4,v5,v6],dtype=np.float64)

def projection3_sets_numerical(vns: NDArray[np.float_]) -> NDArray[np.float_]:
    """perpendicular component of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    num=len(vns)
    m=np.zeros((num,3),dtype=np.float64)
    for i in range(num):
        m[i]=projection3_numerical(vns[i])
    return m



################
# Unnecessary functions？？？
################

def matrix_dot(m1,m2):
    return np.dot(m1,m2)

def inner_product_numerical(v1, v2):
    return np.dot(v1,v2)

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
    
    """
    print('check projection')
    
    nset=2
    vst=generate_random_vectors(nset)
    vsn=numerical_vectors(vst)
    #
    # TAU-style
    vset=get_internal_component_sets_numerical(vst)
    for v in vset:
        print(v)
    # float
    viset=projection3_sets_numerical(vsn)
    for v in viset:
        print(v)
    vset=projection_sets_numerical(vsn)
    for v in vset:
        print(v)
    """
    
    """
    print('check tetrahedron')
    tetrahedron=generate_random_tetrahedron() # in TAU-style
    tetrahedron_num=numerical_vectors(tetrahedron) # in float
    #print(tetrahedron_num)
    vol=tetrahedron_volume_6d_numerical(tetrahedron_num)
    print(vol)
    """
    
    ln=np.array([\
    [1.61803399, -1.,          0. ],\
    [3.73607, -0.19098,1.30902 ]])
    
    tr=np.array([\
    [3.61803399, -1.,          0.38197],\
    [2.61803399, -0.38196601,  0.],\
    [2.61803399, 0,1. ]])
    
    a=check_intersection_segment_surface_numerical(ln,tr)
    print(a)
    