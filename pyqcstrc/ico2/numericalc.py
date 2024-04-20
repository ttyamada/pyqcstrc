#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import numpy as np

TAU=(1+np.sqrt(5))/2.0
EPS=1e-6

def numeric_value(t):
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

def numerical_vector(vt):
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

def numerical_vectors(vts):
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

def length_numerical(vt):
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






def point_on_segment(point, line_segment):
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
    xyzP=projection3_numerical(point)
    xyzL=projection3_sets_numerical(line_segment)
    
    vecPA=xyzP-xyzL[0]
    vecBA=xyzL[1]-xyzL[0]
    
    lPA=np.linalg.norm(vecPA)
    lBA=np.linalg.norm(vecBA)
    if lBA>0.0 and abs(np.dot(vecPA,vecBA)-lPA*lBA)<EPS:
        s=lPA/lBA
        if s>=0.0 and s<=1.0:
            return 0
        elif s>1.0:
            return 1 #       A==B P
        else:
            return -1 #    P A==B
    else:
        return 2

def check_intersection_segment_surface_numerical_6d_tau(line_segment,triangle):
    """check intersection between a line segment and a triangle.
    
    Parameters
    ----------
    line_segment: array
        two 6-dimensional coordinates of line segment,xyzuvw1, xyzuvw2, in TAU-style
    triangle: array
        containing 6-dimensional coordinates of tree vertecies of a triangle (a) in TAU-style
    
    Returns
    -------
    
    """
    ln=get_internal_component_sets_numerical(line_segment)
    tr=get_internal_component_sets_numerical(triangle)
    return check_intersection_segment_surface_numerical_6d_xyz(ln,tr)
    
def check_intersection_segment_surface_numerical(line_segment,triangle):
    """check intersection between a line segment and a triangle.
    
    Parameters
    ----------
    line_segment: array
        two 3-dimensional coordinates of line segment, xyz1, xyz2.
    triangle: array
        containing 3-dimensional coordinates of tree vertecies of a triangle (a), xyz1, xyz2, xyz3.
    
    Returns
    -------
    
    """
    
    def fuc(a,b,c):
        m=np.zeros((3,3),dtype=np.float64)
        m[0]=a
        m[1]=b
        m[2]=c
        return m
    
    vecBA=ln[0]-ln[1] # line segment BA
    vecCD=tr[1]-tr[0] # edge segment of triangle CDE, CD
    vecCE=tr[2]-tr[0] # edge segment of triangle CDE, CE
    vecCA=ln[0]-ln[0] # CA
    
    tmp=fuc(vecCD,vecCE,vecBA)
    bunbo=np.linalg.det(tmp)
    if abs(bunbo)<EPS:
        return False
    else:
        tmp=fuc(vecCA,vecCE,vecBA)
        u=np.linalg.det(tmp)
        if u>=0.0 and u<=bunbo:
            tmp=fuc(vecCD,vecCA,vecBA)
            v=np.linalg.det(tmp)
            if v>=0.0 and u+v<=bunbo:
                tmp=fuc(vecCD,vecCE,vecCA)
                t=np.linalg.det(tmp)
                if t>=0.0 and t<=bunbo:
                    return True # intersect
                else:
                    return False
            else:
                return False
        else:
            return False

def triangle_area(a):
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

def inside_outside_obj(point,obj):
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
        
def inside_outside_tetrahedron(point,tetrahedron):
    """this function judges whether the point is inside a tetrahedron or not
        
    Parameters
    ----------
    point: array
        coordinate of the point,xyz
    tetrahedron: array
        vertex coordinates of tetrahedron, (xyz1, xyz2, xyz3, xyz4)
    """
    vol0=tetrahedron_volume_6d_numerical(tetrahedron)
    #
    tet1=tetrahedron
    tet1[0]=point
    vol1=tetrahedron_volume_6d_numerical(tet1)
    #
    tet2=tetrahedron
    tet2[1]=point
    vol2=tetrahedron_volume_6d_numerical(tet2)
    #
    tet3=tetrahedron
    tet3[2]=point
    vol3=tetrahedron_volume_6d_numerical(tet3)
    #
    tet4=tetrahedron
    tet4[3]=point
    vol4=tetrahedron_volume_6d_numerical(tet4)
    
    if abs(vol0-vol1-vol2-vol3-vol4)<EPS*vol0:
        return True # inside
    else:
        return False # outside







def obj_volume_6d_numerical(obj):
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

def tetrahedron_volume_6d_numerical(tetrahedron):
    """This function returns volume of a tetrahedron
        
    Parameters
    ----------
    tetrahedron: array
        6-dimensional vertex coordinates of the tetrahedron, xyzuvw0,xyzuvw1,xyzuvw2,xyzuvw3
    """
    a=get_internal_component_sets_numerical(tetrahedron)
    return tetrahedron_volume_numerical(a)

def obj_volume_numerical(obj):
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

def tetrahedron_volume_numerical(tetrahedron):
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










def get_internal_component_numerical(vn):
    """
    Parameters
    ----------
    vn: array
        6-dimensional vector, xyzuvw.
    """
    #vn=numerical_vector(vn)
    return projection3_numerical(vn)

def get_internal_component_sets_numerical(vsn):
    """parallel and perpendicular components of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    num=len(vsn)
    m=np.zeros((num,3),dtype=np.float64)
    for i in range(num):
        m[i]=get_internal_component_numerical(vsn[i])
    return m

def projection_numerical(vn):
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

def projection_sets_numerical(vsn):
    """parallel and perpendicular components of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    num=len(vsn)
    m=np.zeros((num,6),dtype=np.float64)
    for i in range(num):
        m[i]=projection_numerical(vsn[i])
    return m
    
def projection3_numerical(vn):
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

def projection3_sets_numerical(vsn):
    """perpendicular component of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    num=len(vsn)
    m=np.zeros((num,3),dtype=np.float64)
    for i in range(num):
        m[i]=projection3_numerical(vsn[i])
    return m



################
# 不要な関数？？？
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
        
    print('check tetrahedron')
    tetrahedron=generate_random_tetrahedron() # in TAU-style
    tetrahedron_num=numerical_vectors(tetrahedron) # in float
    #print(tetrahedron_num)
    vol=tetrahedron_volume_6d_numerical(tetrahedron_num)
    print(vol)
    