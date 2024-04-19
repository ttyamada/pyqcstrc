#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import numpy as np

TAU=(1+np.sqrt(5))/2.0
EPS=1e-6

def numeric_value(a):
    """Numeric value of a TAU-style value, a.

    Parameters
    ----------
    a: array
        value in TAU-style
    
    Returns
    -------
    float
    """
    return (a[0]+a[1]*TAU)/a[2]

def numerical_vector(v):
    """Numeric value of a TAU-style vector, v.

    Parameters
    ----------
    v: array
        vector in TAU-style
    
    Returns
    -------
    array
    """
    n=len(v)
    w=np.zeros(n,dtype=np.float64)
    for i in range(n):
        w[i]=numeric_value(v[i])
    return w













# Numerical version
cdef double inner_product_numerical(np.ndarray[DTYPE_double_t, ndim=1] vector_1, np.ndarray[DTYPE_double_t, ndim=1] vector_2):
    return vector_1[0]*vector_2[0]+vector_1[1]*vector_2[1]+vector_1[2]*vector_2[2]

def length_numerical(v):
    return np.sqrt(np.sum(np.abs(v**2)))
    
# Numerical version
cpdef int point_on_segment(np.ndarray[DTYPE_int_t, ndim=2] point, np.ndarray[DTYPE_int_t, ndim=2] lineA, np.ndarray[DTYPE_int_t, ndim=2] lineB):
    # judge whether a point is on a line segment, A-B, or not.
    cdef list tmp
    cdef double lPA,lBA,s
    cdef double x0,y0,z0,x1,y1,z1,x2,y2,z2
    cdef np.ndarray[DTYPE_double_t, ndim=1] vecPA,vecBA
    #
    #tmp=projection_numerical((point[0][0]+TAU*point[0][1])/(point[0][2]),\
    #                        (point[1][0]+TAU*point[1][1])/(point[1][2]),\
    #                        (point[2][0]+TAU*point[2][1])/(point[2][2]),\
    #                        (point[3][0]+TAU*point[3][1])/(point[3][2]),\
    #                        (point[4][0]+TAU*point[4][1])/(point[4][2]),\
    #                        (point[5][0]+TAU*point[5][1])/(point[5][2]))
    #x0,y0,z0=tmp[3],tmp[4],tmp[5]
    x0,y0,z0=projection3_numerical((point[0][0]+TAU*point[0][1])/(point[0][2]),\
                                (point[1][0]+TAU*point[1][1])/(point[1][2]),\
                                (point[2][0]+TAU*point[2][1])/(point[2][2]),\
                                (point[3][0]+TAU*point[3][1])/(point[3][2]),\
                                (point[4][0]+TAU*point[4][1])/(point[4][2]),\
                                (point[5][0]+TAU*point[5][1])/(point[5][2]))
    #tmp=projection_numerical((lineA[0][0]+TAU*lineA[0][1])/(lineA[0][2]),\
    #                        (lineA[1][0]+TAU*lineA[1][1])/(lineA[1][2]),\
    #                        (lineA[2][0]+TAU*lineA[2][1])/(lineA[2][2]),\
    #                        (lineA[3][0]+TAU*lineA[3][1])/(lineA[3][2]),\
    #                        (lineA[4][0]+TAU*lineA[4][1])/(lineA[4][2]),\
    #                        (lineA[5][0]+TAU*lineA[5][1])/(lineA[5][2]))
    #x1,y1,z1=tmp[3],tmp[4],tmp[5]
    x1,y1,z1=projection3_numerical((lineA[0][0]+TAU*lineA[0][1])/(lineA[0][2]),\
                                (lineA[1][0]+TAU*lineA[1][1])/(lineA[1][2]),\
                                (lineA[2][0]+TAU*lineA[2][1])/(lineA[2][2]),\
                                (lineA[3][0]+TAU*lineA[3][1])/(lineA[3][2]),\
                                (lineA[4][0]+TAU*lineA[4][1])/(lineA[4][2]),\
                                (lineA[5][0]+TAU*lineA[5][1])/(lineA[5][2]))
    #tmp=projection_numerical((lineB[0][0]+TAU*lineB[0][1])/(lineB[0][2]),\
    #                        (lineB[1][0]+TAU*lineB[1][1])/(lineB[1][2]),\
    #                        (lineB[2][0]+TAU*lineB[2][1])/(lineB[2][2]),\
    #                        (lineB[3][0]+TAU*lineB[3][1])/(lineB[3][2]),\
    #                        (lineB[4][0]+TAU*lineB[4][1])/(lineB[4][2]),\
    #                        (lineB[5][0]+TAU*lineB[5][1])/(lineB[5][2]))
    #x2,y2,z2=tmp[3],tmp[4],tmp[5]
    x2,y2,z2=projection3_numerical((lineB[0][0]+TAU*lineB[0][1])/(lineB[0][2]),\
                                (lineB[1][0]+TAU*lineB[1][1])/(lineB[1][2]),\
                                (lineB[2][0]+TAU*lineB[2][1])/(lineB[2][2]),\
                                (lineB[3][0]+TAU*lineB[3][1])/(lineB[3][2]),\
                                (lineB[4][0]+TAU*lineB[4][1])/(lineB[4][2]),\
                                (lineB[5][0]+TAU*lineB[5][1])/(lineB[5][2]))
    vecPA=np.array([x0-x1,y0-y1,z0-z1])
    vecBA=np.array([x2-x1,y2-y1,z2-z1])
    lPA=length_numerical(vecPA)
    lBA=length_numerical(vecBA)
    if lBA>0.0 and abs(inner_product_numerical(vecPA,vecBA)-lPA*lBA)<EPS:
        s=lPA/lBA
        if s>=0.0 and s<=1.0:
            return 0
        elif s>1.0:
            return 1 #       A==B P
        else:
            return -1 #    P A==B
    else:
        return 2

cdef double triangle_area(np.ndarray[DTYPE_int_t, ndim=2] v1,\
                            np.ndarray[DTYPE_int_t, ndim=2] v2,\
                            np.ndarray[DTYPE_int_t, ndim=2] v3):
    cdef double vx0,vx1,vx2,vy0,vy1,vy2,vz0,vz1,vz2,area
    cdef np.ndarray[DTYPE_double_t, ndim=1] vec1,vec2,vec3
    cdef np.ndarray[DTYPE_int_t, ndim=1] x0,x1,x2,y0,y1,y2,z0,z1,z2
    x0=v1[0]
    y0=v1[1]
    z0=v1[2]
    #
    x1=v2[0]
    y1=v2[1]
    z1=v2[2]
    #
    x2=v3[0]
    y2=v3[1]
    z2=v3[2]
    #
    vx0=(x0[0]+x0[1]*TAU)/(x0[2])
    vx1=(x1[0]+x1[1]*TAU)/(x1[2])
    vx2=(x2[0]+x2[1]*TAU)/(x2[2])
    vy0=(y0[0]+y0[1]*TAU)/(y0[2])
    vy1=(y1[0]+y1[1]*TAU)/(y1[2])
    vy2=(y2[0]+y2[1]*TAU)/(y2[2])
    vz0=(z0[0]+z0[1]*TAU)/(z0[2])
    vz1=(z1[0]+z1[1]*TAU)/(z1[2])
    vz2=(z2[0]+z2[1]*TAU)/(z2[2])
    vec1=np.array([vx1-vx0,vy1-vy0,vz1-vz0])
    vec2=np.array([vx2-vx0,vy2-vy0,vz2-vz0])
    
    vec3=np.cross(vec2,vec1) # cross product
    area=np.sqrt(vec3[0]**2+vec3[1]**2+vec3[2]**2)/2.0
    area=abs(area)
    return area

cdef double tetrahedron_volume_numerical(double x0, double y0, double z0,\
                                        double x1, double y1, double z1,\
                                        double x2, double y2, double z2,\
                                        double x3, double y3, double z3):
    # This function returns volume of a tetrahedron
    # parameters: vertex coordinates of the tetrahedron, (x0,y0,z0),(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)
    cdef double detm,vol
    cdef np.ndarray[DTYPE_double_t, ndim=2] m
    m = np.array([[x1-x0,y1-y0,z1-z0],\
                  [x2-x0,y2-y0,z2-z0],\
                  [x3-x0,y3-y0,z3-z0]])
    detm = np.linalg.det(m)
    vol = abs(detm)/6.0
    return vol

cpdef int inside_outside_obj(list point,\
                            np.ndarray[DTYPE_int_t, ndim=4] obj):
    cdef int i1,flag
    cdef double x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3
    cdef np.ndarray[DTYPE_int_t,ndim=1] x1e,y1e,z1e,x1i,y1i,z1i,x2i,y2i,z2i,x3i,y3i,z3i,x4i,y4i,z4i

    flag=0
    for i1 in range(len(obj)):
        x0,y0,z0=get_internal_component_numerical(obj[i1][0])
        x1,y1,z1=get_internal_component_numerical(obj[i1][1])
        x2,y2,z2=get_internal_component_numerical(obj[i1][2])
        x3,y3,z3=get_internal_component_numerical(obj[i1][3])
        if inside_outside_tetrahedron_1(point,[x0,y0,z0],[x1,y1,z1],[x2,y2,z2],[x3,y3,z3])==0:
            flag+=1
            break
        else:
            pass
    if flag>0:
        return 0 # inside
    else:
        return 1 # outside

cdef int inside_outside_tetrahedron_1(list p, list v1, list v2, list v3, list v4):
    cdef double vol0,vol1,vol2,vol3,vol4
    vol0=tetrahedron_volume_numerical(v1[0],v1[1],v1[2],v2[0],v2[1],v2[2],v3[0],v3[1],v3[2],v4[0],v4[1],v4[2])
    vol1=tetrahedron_volume_numerical( p[0], p[1], p[2],v2[0],v2[1],v2[2],v3[0],v3[1],v3[2],v4[0],v4[1],v4[2])
    vol2=tetrahedron_volume_numerical(v1[0],v1[1],v1[2], p[0], p[1], p[2],v3[0],v3[1],v3[2],v4[0],v4[1],v4[2])
    vol3=tetrahedron_volume_numerical(v1[0],v1[1],v1[2],v2[0],v2[1],v2[2], p[0], p[1], p[2],v4[0],v4[1],v4[2])
    vol4=tetrahedron_volume_numerical(v1[0],v1[1],v1[2],v2[0],v2[1],v2[2],v3[0],v3[1],v3[2], p[0], p[1], p[2])
    if abs(vol0-vol1-vol2-vol3-vol4)<EPS*vol0:
        return 0 # inside
    else:
        return 1 # outside

# numerical version
cpdef int inside_outside_tetrahedron(np.ndarray[DTYPE_int_t, ndim=2] point,\
                                    np.ndarray[DTYPE_int_t, ndim=2] tetrahedron_v1,\
                                    np.ndarray[DTYPE_int_t, ndim=2] tetrahedron_v2,\
                                    np.ndarray[DTYPE_int_t, ndim=2] tetrahedron_v3,\
                                    np.ndarray[DTYPE_int_t, ndim=2] tetrahedron_v4):
    # this function judges whether the point is inside a traiangle or not
    # input:
    # (1) vertex coordinates of the triangle,xyz0, xyz1, xyz2, xyz3
    # (2) coordinate of the point,xyz4
    cdef np.ndarray[DTYPE_int_t, ndim=1] m1,m2,m3,m4,m5,m6
    cdef double volume0,volume1,volume2,volume3,volume4,volume_sum
    cdef double x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4
    #
    tetrahedron0=np.append(tetrahedron_v1,tetrahedron_v2)
    tetrahedron0=np.append(tetrahedron0,tetrahedron_v3)
    tetrahedron0=np.append(tetrahedron0,tetrahedron_v4)
    tetrahedron0=tetrahedron0.reshape(4,6,3)
    volume0=tetrahedron_volume_6d_numerical(tetrahedron0)
    #
    tetrahedron1=np.append(point,tetrahedron_v2)
    tetrahedron1=np.append(tetrahedron1,tetrahedron_v3)
    tetrahedron1=np.append(tetrahedron1,tetrahedron_v4)
    tetrahedron1=tetrahedron1.reshape(4,6,3)
    volume1=tetrahedron_volume_6d_numerical(tetrahedron1)
    #
    tetrahedron2=np.append(tetrahedron_v1,point)
    tetrahedron2=np.append(tetrahedron2,tetrahedron_v3)
    tetrahedron2=np.append(tetrahedron2,tetrahedron_v4)
    tetrahedron2=tetrahedron2.reshape(4,6,3)
    volume2=tetrahedron_volume_6d_numerical(tetrahedron2)
    #
    tetrahedron3=np.append(tetrahedron_v1,tetrahedron_v2)
    tetrahedron3=np.append(tetrahedron3,point)
    tetrahedron3=np.append(tetrahedron3,tetrahedron_v4)
    tetrahedron3=tetrahedron3.reshape(4,6,3)
    volume3=tetrahedron_volume_6d_numerical(tetrahedron3)
    #
    tetrahedron4=np.append(tetrahedron_v1,tetrahedron_v2)
    tetrahedron4=np.append(tetrahedron4,tetrahedron_v3)
    tetrahedron4=np.append(tetrahedron4,point)
    tetrahedron4=tetrahedron4.reshape(4,6,3)
    volume4=tetrahedron_volume_6d_numerical(tetrahedron4)
    #
    if abs(volume0-volume1-volume2-volume3-volume4)<EPS*volume0:
        return 0 # inside
    else:
        return 1 # outside

def matrix_dot(array_1, array_2):

    mx1 = array_1.shape[0]
    my1 = array_1.shape[1]
    mx2 = array_2.shape[0]
    my2 = array_2.shape[1]
    array_3 = np.zeros((mx1,my2), dtype=np.int64)
    for x in range(array_1.shape[0]):
        for y in range(array_2.shape[1]):
            for z in range(array_1.shape[1]):
                array_3[x][y] += array_1[x][z] * array_2[z][y]
    return array_3

cdef np.ndarray matrix_pow(np.ndarray array_1, int n):
    cdef Py_ssize_t mx
    cdef Py_ssize_t my
    cdef Py_ssize_t x,y,z
    cdef int i
    mx = array_1.shape[0]
    my = array_1.shape[1]
    array_2 = np.identity(mx, dtype=int)
    if mx == my:
        if n == 0:
            return np.identity(mx, dtype=int)
        elif n<0:
            return np.zeros((mx,mx), dtype=int)
        else:
            for i in range(n):
                tmp = np.zeros((6, 6), dtype=int)
                for x in range(array_2.shape[0]):
                    for y in range(array_1.shape[1]):
                        for z in range(array_2.shape[1]):
                            tmp[x][y] += array_2[x][z] * array_1[z][y]
                array_2 = tmp
            return array_2
    else:
        print('ERROR: matrix has not regular shape')
        return 

###########################
#     Numerical Calc      #
###########################

cpdef double obj_volume_6d_numerical(np.ndarray[DTYPE_int_t, ndim=4] obj):
    cdef int i
    cdef double volume,vol
    volume=0.0
    for i in range(len(obj)):
        vol=tetrahedron_volume_6d_numerical(obj[i])
        volume+=vol    
    return volume

def tetrahedron_volume_6d_numerical(tetrahedron):
    x0,y0,z0=get_internal_component_numerical(tetrahedron[0])
    x1,y1,z1=get_internal_component_numerical(tetrahedron[1])
    x2,y2,z2=get_internal_component_numerical(tetrahedron[2])
    x3,y3,z3=get_internal_component_numerical(tetrahedron[3])
    return tetrahedron_volume_numerical(x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3)

def get_internal_component_numerical(v):
    n1=(vec6d[0][0]+TAU*vec6d[0][1])/(vec6d[0][2])
    n2=(vec6d[1][0]+TAU*vec6d[1][1])/(vec6d[1][2])
    n3=(vec6d[2][0]+TAU*vec6d[2][1])/(vec6d[2][2])
    n4=(vec6d[3][0]+TAU*vec6d[3][1])/(vec6d[3][2])
    n5=(vec6d[4][0]+TAU*vec6d[4][1])/(vec6d[4][2])
    n6=(vec6d[5][0]+TAU*vec6d[5][1])/(vec6d[5][2])
    #v1,v2,v3,v4,v5,v6=projection_numerical(n1,n2,n3,n4,n5,n6)
    v4,v5,v6=projection3_numerical(n1,n2,n3,n4,n5,n6)
    return [v4,v5,v6]

cpdef list projection_numerical(double n1, double n2, double n3, double n4, double n5, double n6):
    #    parallel and perpendicular components of a 6D lattice vector in direct space.
    cdef double const,v1,v2,v3,v4,v5,v6
    cdef np.ndarray[DTYPE_double_t, ndim=2] n,m1,v
    
    #n = np.array([[n1],[n2],[n3],[n4],[n5],[n6]])
    #const = lattice_a
    #const = 1.0/np.sqrt(2.0+TAU)
    #const = 1.0
    #m1 = const*np.array([[ 1.0,  TAU,  TAU,  0.0, -1.0,  0.0],\
    #                    [ TAU,  0.0,  0.0,  1.0,  TAU,  1.0],\
    #                    [ 0.0,  1.0, -1.0, -TAU,  0.0,  TAU],\
    #                    [ TAU, -1.0, -1.0,  0.0, -TAU,  0.0],\
    #                    [-1.0,  0.0,  0.0,  TAU, -1.0,  TAU],\
    #                    [ 0.0,  TAU, -TAU,  1.0,  0.0, -1.0]])
    #v = matrix_dot_cy(m1,n)
    #v1 = v[0][0] # x in Epar
    #v2 = v[1][0] # y in Epar
    #v3 = v[2][0] # z in Epar
    #v4 = v[3][0] # x in Eperp
    #v5 = v[4][0] # y in Eperp
    #v6 = v[5][0] # z in Eperp
    v1 =  (n1-n5) + TAU*(n2+n3) # x in Epar
    v2 =  (n4+n6) + TAU*(n1+n5) # y in Epar
    v3 =  (n2-n3) - TAU*(n4-n6) # z in Epar
    v4 = -(n2+n3) + TAU*(n1-n5) # x in Eperp
    v5 = -(n1+n5) + TAU*(n4+n6) # y in Eperp
    v6 =  (n4-n6) + TAU*(n2-n3) # z in Eperp
    return [v1,v2,v3,v4,v5,v6]

cpdef list projection3_numerical(double n1, double n2, double n3, double n4, double n5, double n6):
    #    perpendicular components of a 6D lattice vector in direct space.
    #cdef double const,v4,v5,v6
    #cdef np.ndarray[DTYPE_double_t, ndim=2] n,m1,v
    
    #n = np.array([[n1],[n2],[n3],[n4],[n5],[n6]])
    #const = lattice_a
    #const = 1.0/np.sqrt(2.0+TAU)
    #const = 1.0
    #m1 = const*np.array([[ 1.0,  TAU,  TAU,  0.0, -1.0,  0.0],\
    #                    [ TAU,  0.0,  0.0,  1.0,  TAU,  1.0],\
    #                    [ 0.0,  1.0, -1.0, -TAU,  0.0,  TAU],\
    #                    [ TAU, -1.0, -1.0,  0.0, -TAU,  0.0],\
    #                    [-1.0,  0.0,  0.0,  TAU, -1.0,  TAU],\
    #                    [ 0.0,  TAU, -TAU,  1.0,  0.0, -1.0]])
    #v = matrix_dot_cy(m1,n)
    v4 = -(n2+n3) + TAU*(n1-n5) # x in Eperp
    v5 = -(n1+n5) + TAU*(n4+n6) # y in Eperp
    v6 =  (n4-n6) + TAU*(n2-n3) # z in Eperp
    #v4 = v[3][0] # x in Eperp
    #v5 = v[4][0] # y in Eperp
    #v6 = v[5][0] # z in Eperp
    return [v4,v5,v6]

cdef np.ndarray matrix_dot_cy(np.ndarray array_1, np.ndarray array_2):
#def matrix_dot_cy(np.ndarray array_1, np.ndarray array_2):
    cdef Py_ssize_t mx1,my1,mx2,my2
    cdef Py_ssize_t x,y,z
    
    mx1 = array_1.shape[0]
    my1 = array_1.shape[1]
    mx2 = array_2.shape[0]
    my2 = array_2.shape[1]
        
    cdef np.ndarray[DTYPE_double_t, ndim=2] array_3
    
    array_3 = np.zeros((mx1,my2), dtype=np.float64)
    for x in range(array_1.shape[0]):
        for y in range(array_2.shape[1]):
            for z in range(array_1.shape[1]):
                array_3[x][y] += array_1[x][z] * array_2[z][y]
    return array_3

cdef double det_matrix(np.ndarray[DTYPE_double_t,ndim=1] a, np.ndarray[DTYPE_double_t,ndim=1] b, np.ndarray[DTYPE_double_t,ndim=1] c):
    #cdef double a1,a2,a3,b1,b2,b3,c1,c2,c3
    return a[0]*b[1]*c[2]+a[2]*b[0]*c[1]+a[1]*b[2]*c[0]-a[2]*b[1]*c[0]-a[1]*b[0]*c[2]-a[0]*b[2]*c[1]

cdef double get_numerical(np.ndarray[DTYPE_int_t, ndim=1] a):
    return (a[0]+a[1]*TAU)/a[2]
    
cdef list get_vec_numerical(np.ndarray[DTYPE_int_t, ndim=2] b):
    cdef int i
    a=[]
    for i in range(6):
        a.append(get_numerical(b[i]))
    return a

cpdef int check_intersection_segment_surface_numerical(np.ndarray[DTYPE_int_t, ndim=2] segment_1,
                                                np.ndarray[DTYPE_int_t, ndim=2] segment_2,
                                                np.ndarray[DTYPE_int_t, ndim=2] surface_1,
                                                np.ndarray[DTYPE_int_t, ndim=2] surface_2,
                                                np.ndarray[DTYPE_int_t, ndim=2] surface_3):
    # check intersection between a line segment (line1a,line1b) and a plane

    cdef double s,t,u,a2x,a2y,a2z,b1x,b1y,b1z,b2x,b2y,b2z,b3x,b3y,b3z,bunbo
    cdef np.ndarray[DTYPE_double_t,ndim=1] line1a,line1b,interval
    cdef np.ndarray[DTYPE_double_t,ndim=1] vec1,vecBA,vecCD,vecCE,vecCA

    cdef double ea0,ea1,ea2,ea3,ea4,ea5
    cdef double eb0,eb1,eb2,eb3,eb4,eb5
    cdef double sa0,sa1,sa2,sa3,sa4,sa5
    cdef double sb0,sb1,sb2,sb3,sb4,sb5
    cdef double sc0,sc1,sc2,sc3,sc4,sc5
    
    ea0,ea1,ea2,ea3,ea4,ea5=get_vec_numerical(segment_1)
    eb0,eb1,eb2,eb3,eb4,eb5=get_vec_numerical(segment_2)
    sa0,sa1,sa2,sa3,sa4,sa5=get_vec_numerical(surface_1)
    sb0,sb1,sb2,sb3,sb4,sb5=get_vec_numerical(surface_2)
    sc0,sc1,sc2,sc3,sc4,sc5=get_vec_numerical(surface_3)
    
    line1a=np.array([ea0,ea1,ea2,ea3,ea4,ea5])
    line1b=np.array([eb0,eb1,eb2,eb3,eb4,eb5])
    aa1=projection3_numerical(ea0,ea1,ea2,ea3,ea4,ea5)
    aa2=projection3_numerical(eb0,eb1,eb2,eb3,eb4,eb5)
    bb1=projection3_numerical(sa0,sa1,sa2,sa3,sa4,sa5)
    bb2=projection3_numerical(sb0,sb1,sb2,sb3,sb4,sb5)
    bb3=projection3_numerical(sc0,sc1,sc2,sc3,sc4,sc5)
    # line segment
    a2x,a2y,a2z=aa2[0]-aa1[0],aa2[1]-aa1[1],aa2[2]-aa1[2] # AB
    # plane CDE
    b1x,b1y,b1z=bb1[0]-aa1[0],bb1[1]-aa1[1],bb1[2]-aa1[2] # AC
    b2x,b2y,b2z=bb2[0]-bb1[0],bb2[1]-bb1[1],bb2[2]-bb1[2] # CD
    b3x,b3y,b3z=bb3[0]-bb1[0],bb3[1]-bb1[1],bb3[2]-bb1[2] # CE
    
    vecBA=np.array([-a2x,-a2y,-a2z]) # line segment BA
    vecCD=np.array([ b2x, b2y, b2z]) # edge segment of triangle CDE, CD
    vecCE=np.array([ b3x, b3y, b3z]) # edge segment of triangle CDE, CE
    vecCA=np.array([-b1x,-b1y,-b1z]) # CA

    bunbo=det_matrix(vecCD,vecCE,vecBA)
    if abs(bunbo)<EPS:
        return 1
    else:
        u=det_matrix(vecCA,vecCE,vecBA)/bunbo
        if u>=0.0 and u<=1.0:
            v=det_matrix(vecCD,vecCA,vecBA)/bunbo
            if v>=0.0 and u+v<=1.0:
                t=det_matrix(vecCD,vecCE,vecCA)/bunbo
                if t>=0.0 and t<=1.0:
                    #interval=line1a+t*(line1b-line1a)
                    #interval_1=projection3_numerical(interval[0],interval[1],interval[2],interval[3],interval[4],interval[5])
                    #return interval[0],interval[1],interval[2],interval[3],interval[4],interval[5],interval_1[0],interval_1[1],interval_1[2],u,v,t,bunbo
                    return 0 # intersect
                else:
                    return 1
            else:
                return 1
        else:
            return 1


if __name__ == '__main__':
    
    a=np.array([1,1,1])
    a=numeric_value(a)
    print(a)