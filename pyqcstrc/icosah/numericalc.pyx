#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
cimport numpy as np
cimport cython

DTYPE_double = np.float64
DTYPE_int = int

cdef np.float64_t TAU=(1+np.sqrt(5))/2.0
cdef np.float64_t EPS=1e-6 # tolerance

# Numerical version
cdef double inner_product_numerical(np.ndarray[DTYPE_double_t, ndim=1] vector_1, np.ndarray[DTYPE_double_t, ndim=1] vector_2):
    return vector_1[0]*vector_2[0]+vector_1[1]*vector_2[1]+vector_1[2]*vector_2[2]

cdef double length_numerical(np.ndarray[DTYPE_double_t, ndim=1] vector):
    return np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)

# Numerical version
cpdef int point_on_segment(np.ndarray[DTYPE_int_t, ndim=2] point, np.ndarray[DTYPE_int_t, ndim=2] lineA, np.ndarray[DTYPE_int_t, ndim=2] lineB):
    # judge whether a point is on a line segment, A-B, or not.
    # http://marupeke296.com/COL_2D_No2_PointToLine.html
    cdef list tmp
    cdef double lPA,lBA,s
    cdef double x0,y0,z0,x1,y1,z1,x2,y2,z2
    cdef np.ndarray[DTYPE_double_t, ndim=1] vecPA,vecBA
    #
    tmp=projection_numerical((point[0][0]+TAU*point[0][1])/float(point[0][2]),\
                            (point[1][0]+TAU*point[1][1])/float(point[1][2]),\
                            (point[2][0]+TAU*point[2][1])/float(point[2][2]),\
                            (point[3][0]+TAU*point[3][1])/float(point[3][2]),\
                            (point[4][0]+TAU*point[4][1])/float(point[4][2]),\
                            (point[5][0]+TAU*point[5][1])/float(point[5][2]))
    x0,y0,z0=tmp[3],tmp[4],tmp[5]
    tmp=projection_numerical((lineA[0][0]+TAU*lineA[0][1])/float(lineA[0][2]),\
                            (lineA[1][0]+TAU*lineA[1][1])/float(lineA[1][2]),\
                            (lineA[2][0]+TAU*lineA[2][1])/float(lineA[2][2]),\
                            (lineA[3][0]+TAU*lineA[3][1])/float(lineA[3][2]),\
                            (lineA[4][0]+TAU*lineA[4][1])/float(lineA[4][2]),\
                            (lineA[5][0]+TAU*lineA[5][1])/float(lineA[5][2]))
    x1,y1,z1=tmp[3],tmp[4],tmp[5]
    tmp=projection_numerical((lineB[0][0]+TAU*lineB[0][1])/float(lineB[0][2]),\
                            (lineB[1][0]+TAU*lineB[1][1])/float(lineB[1][2]),\
                            (lineB[2][0]+TAU*lineB[2][1])/float(lineB[2][2]),\
                            (lineB[3][0]+TAU*lineB[3][1])/float(lineB[3][2]),\
                            (lineB[4][0]+TAU*lineB[4][1])/float(lineB[4][2]),\
                            (lineB[5][0]+TAU*lineB[5][1])/float(lineB[5][2]))
    x2,y2,z2=tmp[3],tmp[4],tmp[5]
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
    vx0=(x0[0]+x0[1]*TAU)/float(x0[2])
    vx1=(x1[0]+x1[1]*TAU)/float(x1[2])
    vx2=(x2[0]+x2[1]*TAU)/float(x2[2])
    vy0=(y0[0]+y0[1]*TAU)/float(y0[2])
    vy1=(y1[0]+y1[1]*TAU)/float(y1[2])
    vy2=(y2[0]+y2[1]*TAU)/float(y2[2])
    vz0=(z0[0]+z0[1]*TAU)/float(z0[2])
    vz1=(z1[0]+z1[1]*TAU)/float(z1[2])
    vz2=(z2[0]+z2[1]*TAU)/float(z2[2])
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

cdef np.ndarray matrix_dot(np.ndarray array_1, np.ndarray array_2):
    cdef Py_ssize_t mx1,my1,mx2,my2
    cdef Py_ssize_t x,y,z    
    mx1 = array_1.shape[0]
    my1 = array_1.shape[1]
    mx2 = array_2.shape[0]
    my2 = array_2.shape[1]
    array_3 = np.zeros((mx1,my2), dtype=int)
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

cpdef double tetrahedron_volume_6d_numerical(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron):
    cdef double x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3
    cdef np.ndarray[DTYPE_int_t,ndim=1] x1e,y1e,z1e,x1i,y1i,z1i,x2i,y2i,z2i,x3i,y3i,z3i,x4i,y4i,z4i
    x0,y0,z0=get_internal_component_numerical(tetrahedron[0])
    x1,y1,z1=get_internal_component_numerical(tetrahedron[1])
    x2,y2,z2=get_internal_component_numerical(tetrahedron[2])
    x3,y3,z3=get_internal_component_numerical(tetrahedron[3])
    return tetrahedron_volume_numerical(x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3)

cdef list get_internal_component_numerical(np.ndarray[DTYPE_int_t, ndim=2] vec6d):
    cdef double n1,n2,n3,n4,n5,n6
    cdef double v1,v2,v3,v4,v5,v6
    #cdef np.ndarray[DTYPE_double_t, ndim=1] a1,a2,a3,a4,a5,a6
    #print 'a=',vec6d[0][2]
    #print 'b=',float(vec6d[0][2])
    n1=(vec6d[0][0]+TAU*vec6d[0][1])/float(vec6d[0][2])
    #print 'n1=',n1
    n2=(vec6d[1][0]+TAU*vec6d[1][1])/float(vec6d[1][2])
    n3=(vec6d[2][0]+TAU*vec6d[2][1])/float(vec6d[2][2])
    n4=(vec6d[3][0]+TAU*vec6d[3][1])/float(vec6d[3][2])
    n5=(vec6d[4][0]+TAU*vec6d[4][1])/float(vec6d[4][2])
    n6=(vec6d[5][0]+TAU*vec6d[5][1])/float(vec6d[5][2])
    v1,v2,v3,v4,v5,v6=projection_numerical(n1,n2,n3,n4,n5,n6)
    return [v4,v5,v6]

cpdef list projection_numerical(double n1, double n2, double n3, double n4, double n5, double n6):
    #    parallel and perpendicular components of a 6D lattice vector in direct space.
    cdef double const,v1,v2,v3,v4,v5,v6
    cdef np.ndarray[DTYPE_double_t, ndim=2] n,m1,v
    
    n = np.array([[n1],[n2],[n3],[n4],[n5],[n6]])
    #const = lattice_a
    #const = 1.0/np.sqrt(2.0+TAU)
    const = 1.0
    m1 = const*np.array([[ 1.0,  TAU,  TAU,  0.0, -1.0,  0.0],\
                        [ TAU,  0.0,  0.0,  1.0,  TAU,  1.0],\
                        [ 0.0,  1.0, -1.0, -TAU,  0.0,  TAU],\
                        [ TAU, -1.0, -1.0,  0.0, -TAU,  0.0],\
                        [-1.0,  0.0,  0.0,  TAU, -1.0,  TAU],\
                        [ 0.0,  TAU, -TAU,  1.0,  0.0, -1.0]])
    v = matrix_dot_cy(m1,n)
    v1 = v[0][0] # x in Epar
    v2 = v[1][0] # y in Epar
    v3 = v[2][0] # z in Epar
    v4 = v[3][0] # x in Eperp
    v5 = v[4][0] # y in Eperp
    v6 = v[5][0] # z in Eperp
    return [v1,v2,v3,v4,v5,v6]

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

cdef det_matrix_cy(np.ndarray a, np.ndarray b, np.ndarray c):
    cdef double a1,a2,a3,b1,b2,b3,c1,c2,c3
    return a[0]*b[1]*c[2]+a[2]*b[0]*c[1]+a[1]*b[2]*c[0]-a[2]*b[1]*c[0]-a[1]*b[0]*c[2]-a[0]*b[2]*c[1]
