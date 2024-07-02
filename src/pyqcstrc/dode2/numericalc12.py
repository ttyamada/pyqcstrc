#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np

DTYPE_double = np.float64
DTYPE_int = np.int64

SIN=np.sqrt(3)/2.0
EPS=1e-6 # tolerance

def numerical_value(vt):
    """"
    numerical value of a value in SIN-style
    """
    return vt[0]+SIN*vt[1])(vt[2]
    
def numerical_value_vector(vt):
    """"
    numerical value of a 6D vector in SIN-style
    """
    a=np.zseros((6),dtype=DTYPE_double)
    for i in range(6):
        a[i]=numerical_value(vt[i])
    return a
    
def inner_product_numerical(v1, v2):
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

def length_numerical(v):
    return np.sqrt(v[0]**2+v[1]**2+v[2]**2)

def point_on_segment(point,lineA, lineB):
    # judge whether a point is on a line segment, A-B, or not.
    # http://marupeke296.com/COL_2D_No2_PointToLine.html
    p=numerical_value_vector(point)
    xyz0=projection3_numerical(p)
    
    a=numerical_value_vector(lineA)
    xyz1=projection3_numerical(a)
    
    b=numerical_value_vector(lineB)
    xyz2=projection3_numerical(b)
    
    vecPA=np.array([xyz0[0]-xyz1[0],xyz0[1]-xyz1[1],xyz0[2]-xyz1[2]])
    vecBA=np.array([xyz2[0]-xyz1[0],xyz2[1]-xyz1[1],xyz2[2]-xyz[2]])
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

def get_internal_component_numerical(vt):
    a=numerical_value_vector(vt)
    return projection3_numerical(a)
    
def get_internal_component_phason_numerical(vec6d,pmatrx):
    n1=(vec6d[0][0]+SIN*vec6d[0][1])/vec6d[0][2]
    n2=(vec6d[1][0]+SIN*vec6d[1][1])/vec6d[1][2]
    n3=(vec6d[2][0]+SIN*vec6d[2][1])/vec6d[2][2]
    n4=(vec6d[3][0]+SIN*vec6d[3][1])/vec6d[3][2]
    n5=(vec6d[4][0]+SIN*vec6d[4][1])/vec6d[4][2]
    n6=(vec6d[5][0]+SIN*vec6d[5][1])/vec6d[5][2]
    a11=pmatrx[0][0]
    a12=pmatrx[0][1]
    a21=pmatrx[1][0]
    a22=pmatrx[1][1]
    return projection3_phason_numerical(n1,n2,n3,n4,n5,n6,a11,a12,a21,a22)
    
cpdef list projection_numerical(np.ndarray[DTYPE_int_t, ndim=2] vec6d):
    cdef double n1,n2,n3,n4,n5,n6
    #cdef double v1,v2,v3,v4,v5,v6
    cdef double a11,a12,a13,a21,a22,a23,a31,a32,a33
    n1=(vec6d[0][0]+SIN*vec6d[0][1])/vec6d[0][2]
    n2=(vec6d[1][0]+SIN*vec6d[1][1])/vec6d[1][2]
    n3=(vec6d[2][0]+SIN*vec6d[2][1])/vec6d[2][2]
    n4=(vec6d[3][0]+SIN*vec6d[3][1])/vec6d[3][2]
    n5=(vec6d[4][0]+SIN*vec6d[4][1])/vec6d[4][2]
    n6=(vec6d[5][0]+SIN*vec6d[5][1])/vec6d[5][2]
    #v1,v2,v3,v4,v5,v6=projection_1_numerical(n1,n2,n3,n4,n5,n6)
    #return [v1,v2,v3,v4,v5,v6]
    return projection_1_numerical(n1,n2,n3,n4,n5,n6)
    
cpdef list projection_1_numerical(double h1,double h2,double h3,double h4,double h5,double h6):
    # projection of a 6d vector onto Eperp, using "SIN-style"
    #
    # NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    # alpha = 2*a/np.sqrt(6)
    # see Yamamoto ActaCrystal (1997)
    cdef double v1e,v2e,v3e
    cdef double v1i,v2i,v3i
    
    v1e= SIN*h1+h2-0.5*h4
    v2e=-0.5*h1+h3+SIN*h4
    v3e=h5 # dummy
    v1i=-SIN*h1+h2-0.5*h4
    v2i=-0.5*h1+h3-SIN*h4
    v3i=h6 # dummy
    
    return [v1e,v2e,v3e,v1i,v2i,v3i]

cpdef list projection_2_numerical(double h1,double h2,double h3,double h4,double h5,double h6,
                                double u11,double u12,
                                double u21,double u22):
    # projection of a 6d vector onto Eperp, using "SIN-style"
    #
    # NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    # alpha = 2*a/np.sqrt(6)
    # see Yamamoto ActaCrystal (1997)
    cdef double v1e,v2e,v3e
    cdef double v1i,v2i,v3i
    v1e=SIN*h1+h2-0.5*h4
    v2e=-0.5*h1+h3+SIN*h4
    v3e=h5 # dummy
    v1i=(-SIN+u11*SIN-0.5*u21)*h1 + (1+u11)*h2 + u21*h3 + (-0.5-0.5*u11+SIN*u21)*h4
    v2i=(-0.5+SIN*u12-0.5*u22)*h1 + u12*h2 + (1+u22)*h3 + (-SIN-0.5*u12+SIN*u22)*h4
    v3i=h6 # dummy
    return [v1e,v2e,v3e,v1i,v2i,v3i]

"""
cpdef list projection_numerical_phason(np.ndarray[DTYPE_int_t, ndim=2] vec6d,
                                    np.ndarray[DTYPE_int_t, ndim=3] pmatrx):
    #
    # phason matrix being defined using "SIN-style"
    #
    cdef double n1,n2,n3,n4,n5,n6
    cdef double a11,a12,a21,a22
    n1=(vec6d[0][0]+SIN*vec6d[0][1])/vec6d[0][2]
    n2=(vec6d[1][0]+SIN*vec6d[1][1])/vec6d[1][2]
    n3=(vec6d[2][0]+SIN*vec6d[2][1])/vec6d[2][2]
    n4=(vec6d[3][0]+SIN*vec6d[3][1])/vec6d[3][2]
    n5=(vec6d[4][0]+SIN*vec6d[4][1])/vec6d[4][2]
    n6=(vec6d[5][0]+SIN*vec6d[5][1])/vec6d[5][2]
    a11=(pmatrx[0][0][0]+pmatrx[0][0][1]*SIN)/pmatrx[0][0][2]
    a12=(pmatrx[0][1][0]+pmatrx[0][1][1]*SIN)/pmatrx[0][1][2]
    a21=(pmatrx[1][0][0]+pmatrx[1][0][1]*SIN)/pmatrx[1][0][2]
    a22=(pmatrx[1][1][0]+pmatrx[1][1][1]*SIN)/pmatrx[1][1][2]
    return projection_2_numerical(n1,n2,n3,n4,n5,n6,a11,a12,a21,a22)
"""

cpdef list projection_numerical_phason_1(np.ndarray[DTYPE_int_t, ndim=2] vec6d,
                                    np.ndarray[double, ndim=2] pmatrx):
    #
    cdef double n1,n2,n3,n4,n5,n6
    cdef double a11,a12,a21,a22
    n1=(vec6d[0][0]+SIN*vec6d[0][1])/vec6d[0][2]
    n2=(vec6d[1][0]+SIN*vec6d[1][1])/vec6d[1][2]
    n3=(vec6d[2][0]+SIN*vec6d[2][1])/vec6d[2][2]
    n4=(vec6d[3][0]+SIN*vec6d[3][1])/vec6d[3][2]
    n5=(vec6d[4][0]+SIN*vec6d[4][1])/vec6d[4][2]
    n6=(vec6d[5][0]+SIN*vec6d[5][1])/vec6d[5][2]
    a11=pmatrx[0][0]
    a12=pmatrx[0][1]
    a21=pmatrx[1][0]
    a22=pmatrx[1][1]
    return projection_2_numerical(n1,n2,n3,n4,n5,n6,a11,a12,a21,a22)

cdef list projection3_numerical(double h1,
                                double h2,
                                double h3,
                                double h4,
                                double h5,
                                double h6):
    # projection of a 6d vector onto Eperp, using "SIN-style"
    #
    # NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    # alpha = 2*a/np.sqrt(6)
    # see Yamamoto ActaCrystal (1997)
    
    cdef double v1i,v2i,v3i
    v1i=-SIN*h1+h2-0.5*h4
    v2i=-0.5*h1+h3-SIN*h4
    v3i=h6 # dummy
    return [v1i,v2i,v3i]

cdef list projection3_phason_numerical(double h1,double h2,double h3,double h4,double h5,double h6,
                                double u11,double u12,
                                double u21,double u22):
    # projection of a 6d vector onto Eperp, using "SIN-style"
    #
    # NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    # alpha = 2*a/np.sqrt(6)
    # see Yamamoto ActaCrystal (1997)
    
    cdef double v1i,v2i,v3i
    v1i=(-SIN+u11*SIN-0.5*u21)*h1 + (1+u11)*h2 + u21*h3 + (-0.5-0.5*u11+SIN*u21)*h4
    v2i=(-0.5+SIN*u12-0.5*u22)*h1 + u12*h2 + (1+u22)*h3 + (-SIN-0.5*u12+SIN*u22)*h4
    v3i=h6 # dummy
    return [v1i,v2i,v3i]

cpdef int inout_occupation_domain_numerical(np.ndarray[DTYPE_int_t, ndim=4] obj, 
                                    np.ndarray[DTYPE_double_t, ndim=1] point,
                                    int verbose):
    cdef int i1,i2,counter,num
    cdef list lst,vec
    cdef np.ndarray[DTYPE_double_t, ndim=3] triangles
    
    counter=0
    num=len(obj)
    lst=[]
    for i1 in range(num):
        for i2 in range(3):
            vec=get_internal_component_numerical(obj[i1][i2])
            lst.append(vec)
    triangles=np.array(lst).reshape(num,3,3)
    
    for i1 in range(num):
        if inside_outside_triangle_numerical(point,triangles[i1],verbose-1)==0: # inside
            counter+=1
            break
        else:
            pass
    if counter>0:
        return 0
    else:
        return 1

cpdef int inout_occupation_domain_phason_numerical(np.ndarray[DTYPE_int_t, ndim=4] obj, 
                                    np.ndarray[DTYPE_double_t, ndim=1] point,
                                    np.ndarray[double, ndim=2] pmatrx,
                                    int verbose):
    cdef int i1,i2,counter,num
    cdef list lst,vec
    cdef np.ndarray[DTYPE_double_t, ndim=3] triangles
    
    counter=0
    num=len(obj)
    lst=[]
    for i1 in range(num):
        for i2 in range(3):
            vec=get_internal_component_phason_numerical(obj[i1][i2],pmatrx)
            lst.append(vec)
    triangles=np.array(lst).reshape(num,3,3)
    
    for i1 in range(num):
        if inside_outside_triangle_numerical(point,triangles[i1],verbose-1)==0: # inside
            counter+=1
            break
        else:
            pass
    if counter>0:
        return 0
    else:
        return 1

# numerical version
cdef int inside_outside_triangle_numerical(np.ndarray[DTYPE_double_t, ndim=1] point,\
                                            np.ndarray[DTYPE_double_t, ndim=2] triangle,\
                                            int verbose):
    cdef np.ndarray[DTYPE_double_t, ndim=1] tmp1
    cdef np.ndarray[DTYPE_double_t, ndim=2] tmp2
    cdef double area0,area1,area2,area3
    cdef list lst
    
    if verbose>0:
        print('            inside_outside_triangle_numerical()')
    else:
        pass
    #
    
    tmp1=np.append(triangle[0],triangle[1])
    tmp1=np.append(tmp1,triangle[2])
    tmp2=tmp1.reshape(3,3)
    area0=triangle_area(tmp2)
    #
    tmp1=np.append(point,triangle[1])
    tmp1=np.append(tmp1,triangle[2])
    tmp2=tmp1.reshape(3,3)
    area1=triangle_area(tmp2)
    #
    tmp1=np.append(point,triangle[0])
    tmp1=np.append(tmp1,triangle[2])
    tmp2=tmp1.reshape(3,3)
    area2=triangle_area(tmp2)
    #
    tmp1=np.append(point,triangle[0])
    tmp1=np.append(tmp1,triangle[1])
    tmp2=tmp1.reshape(3,3)
    area3=triangle_area(tmp2)
    #
    if verbose>0:
        print('             total0: %8.6f'%(area0))
        print('             total1: %8.6f'%(area1+area2+area3))
    else:
        pass
    #if abs(area0-area1-area2-area3)<area0*EPS:
    if abs(area0-area1-area2-area3)<EPS:
        return 0 # inside
    else:
        return 1 # outside

cpdef int inside_outside_triangle(np.ndarray[DTYPE_int_t, ndim=2] point,\
                                np.ndarray[DTYPE_int_t, ndim=3] triangle,\
                                int verbose):
    cdef np.ndarray[DTYPE_int_t, ndim=1] tmp1
    cdef np.ndarray[DTYPE_int_t, ndim=3] tmp3
    cdef double area0,area1,area2,area3
    
    if verbose>0:
        print('            inside_outside_triangle()')
    else:
        pass
    #
    tmp1=np.append(triangle[0],triangle[1])
    tmp1=np.append(tmp1,triangle[2])
    tmp3=tmp1.reshape(3,6,3)
    area0=triangle_area_6d_numerical(tmp3)
    #
    tmp1=np.append(point,triangle[1])
    tmp1=np.append(tmp1,triangle[2])
    tmp3=tmp1.reshape(3,6,3)
    area1=triangle_area_6d_numerical(tmp3)
    #
    tmp1=np.append(point,triangle[0])
    tmp1=np.append(tmp1,triangle[2])
    tmp3=tmp1.reshape(3,6,3)
    area2=triangle_area_6d_numerical(tmp3)
    #
    tmp1=np.append(point,triangle[0])
    tmp1=np.append(tmp1,triangle[1])
    tmp3=tmp1.reshape(3,6,3)
    area3=triangle_area_6d_numerical(tmp3)
    #
    if verbose>0:
        print('             total0: %8.6f'%(area0))
        print('             total1: %8.6f'%(area1+area2+area3))
    else:
        pass
    if abs(area0-area1-area2-area3)<EPS:
    #if abs(area0-area1-area2-area3)<EPS*area0:
        return 0 # inside
    else:
        return 1 # outside

cdef double obj_area_6d_numerical(np.ndarray[DTYPE_int_t, ndim=4] obj):
    cdef int i
    cdef double area,tmp
    area=0.0
    for i in range(len(obj)):
        area+=triangle_area_6d_numerical(obj[i])
    return area

cdef double triangle_area_6d_numerical(np.ndarray[DTYPE_int_t, ndim=3] triangle):
    cdef double x0,y0,z0,x1,y1,z1,x2,y2,z2
    cdef np.ndarray[DTYPE_double_t,ndim=2] tmp2
    x0,y0,z0=get_internal_component_numerical(triangle[0])
    x1,y1,z1=get_internal_component_numerical(triangle[1])
    x2,y2,z2=get_internal_component_numerical(triangle[2])
    tmp2=np.array([[x0,y0,z0],[x1,y1,z1],[x2,y2,z2]])
    return triangle_area(tmp2)

cdef double triangle_area(np.ndarray[DTYPE_double_t, ndim=2] triangle):
    cdef double area
    cdef np.ndarray[DTYPE_double_t, ndim=1] vec1,vec2,vec3

    vec1=np.array([triangle[1][0]-triangle[0][0],triangle[1][1]-triangle[0][1],triangle[1][2]-triangle[0][2]])
    vec2=np.array([triangle[2][0]-triangle[0][0],triangle[2][1]-triangle[0][1],triangle[2][2]-triangle[0][2]])
    
    vec3=np.cross(vec2,vec1) # cross product
    #print(vec3)
    area=np.sqrt(vec3[0]**2+vec3[1]**2+vec3[2]**2)/2.0
    return area

cdef double det_matrix(np.ndarray[DTYPE_double_t,ndim=1] a, np.ndarray[DTYPE_double_t,ndim=1] b, np.ndarray[DTYPE_double_t,ndim=1] c):
    return a[0]*b[1]*c[2]+a[2]*b[0]*c[1]+a[1]*b[2]*c[0]-a[2]*b[1]*c[0]-a[1]*b[0]*c[2]-a[0]*b[2]*c[1]

cdef double dot_matrix(np.ndarray[DTYPE_double_t,ndim=1] a, np.ndarray[DTYPE_double_t,ndim=1] b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef double dot_matrix(double a1, double a2, double a3, double b1, double b2, double b3):
#    return a1*b1+a2*b2+a3*b3

cdef double get_numerical(np.ndarray[DTYPE_int_t, ndim=1] a):
    return (a[0]+a[1]*SIN)/a[2]

cdef list get_vec_numerical(np.ndarray[DTYPE_int_t, ndim=2] b):
    cdef int i
    a=[]
    for i in range(6):
        a.append(get_numerical(b[i]))
    return a

cpdef int check_intersection_two_triangles(np.ndarray[DTYPE_int_t, ndim=3] triangle_1,
                                                np.ndarray[DTYPE_int_t, ndim=3] triangle_2,
                                                int verbose):
    cdef int j,counter1,counter2,counter3
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1,tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3
    cdef list comb
    
    if verbose>0:
        print('       check_intersection_two_triangles()')
    else:
        pass
    
    if rough_check_intersection_two_triangles(triangle_1,triangle_2,verbose-1)==0:
        
        #  edge: 0-1,0-2,1-2
        comb=[[0,1],[0,2],[1,2]]
        counter1=0
        # get intersecting points
        for j in range(3):
            tmp3=np.vstack([triangle_1[comb[j][0]],triangle_1[comb[j][1]]]).reshape(2,6,3)
            
            if check_intersection_line_segment_triangle(tmp3,triangle_2,verbose-1)==0: # intersecting
                counter1+=1
                break
            else:
                pass
    
        counter2=0
        # get vertces of triangle_1 which are inside triangle_2
        for i1 in range(3):
            if inside_outside_triangle(triangle_1[i1],triangle_2,verbose-1)==0:
                counter2+=1
                #break
            else:
                pass
    
        counter3=0
        # get vertces of triangle_2 which are inside triangle_1
        for i1 in range(3):
            if inside_outside_triangle(triangle_2[i1],triangle_1,verbose-1)==0:
                counter3+=1
                #break
            else:
                pass

        if counter2==3 :
            if verbose>1:
                print('        triangle_1 is inside triangle_2')
            else:
                pass
            return 2 # triangle_1 is inside triangle_2
        elif counter3==3:
            if verbose>1:
                print('        triangle_2 is inside triangle_1')
            else:
                pass
            return 3 # triangle_2 is inside triangle_1
        else:
            if counter1>0:
                if verbose>1:
                    print('        intersection')
                else:
                    pass
                return 0 # intersecting
            else:
                if verbose>1:
                    print('        no intersection')
                else:
                    pass
                return 1 # no intersecting
    else:
        pass
    return 1 # no intersecting

cpdef int check_intersection_line_segment_triangle(np.ndarray[DTYPE_int_t, ndim=3] line_segment,
                                                    np.ndarray[DTYPE_int_t, ndim=3] triangle,
                                                    int verbose):
    cdef int j,counter
    cdef list comb
    
    if verbose>0:
        print('        check_intersection_line_segment_triangle()')
    else:
        pass
    #  edge: 0-1,0-2,1-2
    comb=[[0,1],[0,2],[1,2]]
    counter=0
    for j in range(3):
        if check_intersection_two_segment_numerical(line_segment,triangle[comb[j]],verbose-1)==0:
            counter+=1
            break
        else:
            pass
    if counter>0:
        if verbose>1:
            print('         intersection')
        else:
            pass
        return 0
    else:
        if verbose>1:
            print('         no intersection')
        else:
            pass
        return 1

cpdef int check_intersection_two_segment_numerical(np.ndarray[DTYPE_int_t, ndim=3] segment_1,
                                                np.ndarray[DTYPE_int_t, ndim=3] segment_2,
                                                int verbose):
    cdef double s,t,
    cdef double ddx,ddy,ddz
    cdef double t1,t2,t3,t4,t5,t6
    cdef double L1ax,L1ay,L1az
    cdef double L1bx,L1by,L1bz
    cdef double L2ax,L2ay,L2az
    cdef double L2bx,L2by,L2bz
    cdef np.ndarray[DTYPE_double_t,ndim=1] vec1,vecBA,vecCD,vecCE,vecCA
    cdef np.ndarray[DTYPE_double_t,ndim=1] line1a,line1b,line2a,line2b
    #cdef np.ndarray[DTYPE_double_t,ndim=1] intvl
    
    if verbose>0:
        print('         check_intersection_two_segment_numerical()')
    else:
        pass
    
    # line1-A
    t1,t2,t3,t4,t5,t6=get_vec_numerical(segment_1[0])
    L1ax,L1ay,L1az=projection3_numerical(t1,t2,t3,t4,t5,t6)
    line1a=np.array([t1,t2,t3,t4,t5,t6])
    
    # line1-B
    t1,t2,t3,t4,t5,t6=get_vec_numerical(segment_1[1])
    L1bx,L1by,L1bz=projection3_numerical(t1,t2,t3,t4,t5,t6)
    line1b=np.array([t1,t2,t3,t4,t5,t6])
    
    # line2-A
    t1,t2,t3,t4,t5,t6=get_vec_numerical(segment_2[0])
    L2ax,L2ay,L2az=projection3_numerical(t1,t2,t3,t4,t5,t6)
    line2a=np.array([t1,t2,t3,t4,t5,t6])
    
    # line2-B
    t1,t2,t3,t4,t5,t6=get_vec_numerical(segment_2[1])
    L2bx,L2by,L2bz=projection3_numerical(t1,t2,t3,t4,t5,t6)
    line2b=np.array([t1,t2,t3,t4,t5,t6])
    
    vecAB=np.array([L1bx-L1ax,L1by-L1ay,L1bz-L1az])
    vecAC=np.array([L2ax-L1ax,L2ay-L1ay,L2az-L1az])
    vecCD=np.array([L2bx-L2ax,L2by-L2ay,L2bz-L2az])
    
    # bunshi
    t1=dot_matrix(vecAC,vecCD)*dot_matrix(vecCD,vecAB)-dot_matrix(vecCD,vecCD)*dot_matrix(vecAC,vecAB)
    # bunbo
    t2=dot_matrix(vecAB,vecCD)*dot_matrix(vecCD,vecAB)-dot_matrix(vecAB,vecAB)*dot_matrix(vecCD,vecCD)
    
    if verbose>1:
        print('          bunshi = %8.6f'%(t1))
        print('          bunbo = %8.6f'%(t2))
    else:
        pass
        
    if abs(t2)<EPS:
        if verbose>1:
            print('          no intersection')
        else:
            pass
        return 1
    else:
        s=t1/t2
        t=(-dot_matrix(vecAC,vecCD)+s*dot_matrix(vecAB,vecCD))/dot_matrix(vecCD,vecCD)
        if verbose>1:
            print('          s = %8.6f'%(s))
            print('          t = %8.6f'%(t))
        else:
            pass
        if s>=0.0 and s<=1.0 and t>=0.0 and t<=1.0:
            ddx=(L2ax-L1ax)-s*(L1bx-L1ax)+t*(L2bx-L2ax)
            ddy=(L2ay-L1ay)-s*(L1by-L1ay)+t*(L2by-L2ay)
            ddz=(L2az-L1az)-s*(L1bz-L1az)+t*(L2bz-L2az)
            if verbose>1:
                print('          ddx = %8.6f'%(ddx))
                print('          ddy = %8.6f'%(ddy))
                print('          ddz = %8.6f'%(ddz))
                
            if ddx**2+ddy**2+ddz**2<EPS:
                if verbose>1:
                    print('          intersection')
                else:
                    pass
                #intvl=line1a+s*(line1b-line1a)
                #t1,t2,t3=projection3_numerical(intvl[0],intvl[1],intvl[2],intvl[3],intvl[4],intvl[5])
                #return intvl[0],intvl[1],intvl[2],intvl[3],intvl[4],intvl[5],t1,t2,t3
                return 0
            else:
                if verbose>1:
                    print('          no intersection')
                else:
                    pass
                return 1
        else:
            if verbose>1:
                print('          no intersection')
            else:
                pass
            return 1

cdef int rough_check_intersection_two_triangles(np.ndarray[DTYPE_int_t, ndim=3] triangle_1,
                                                np.ndarray[DTYPE_int_t, ndim=3] triangle_2,
                                                int verbose):
    cdef double r1,r2,dd
    cdef np.ndarray[DTYPE_double_t,ndim=1] cen1,cen2

    if verbose>0:
        print('        rough_check_intersection_two_triangles()')
    else:
        pass
    cen1=centroid_obj(triangle_1)
    r1=circle_radius(triangle_1,cen1)
    
    cen2=centroid_obj(triangle_2)
    r2=circle_radius(triangle_2,cen2)
    
    dd=distance(cen1,cen2)
    
    if dd<=r1+r2:
        if verbose>1:
            print('         intersecting')
        else:
            pass
        return 0
    else:
        if verbose>1:
            print('         no intersecting')
        else:
            pass
        return 1

cdef np.ndarray centroid_obj(np.ndarray[DTYPE_int_t, ndim=3] triangle):
    #  geometric center, centroid of triangle
    cdef int i1,i2
    cdef double t1,t2,t3,t4,t5,t6
    cdef double val
    cdef np.ndarray[DTYPE_double_t,ndim=2] tmp2a
    cdef list lsta
    
    lsta=[]
    for i1 in range(3):
        t1,t2,t3,t4,t5,t6=get_vec_numerical(triangle[i1])
        t1,t2,t3=projection3_numerical(t1,t2,t3,t4,t5,t6)
        lsta.append([t1,t2,t3])
    tmp2a=np.array(lsta)
    
    lsta=[]
    for i1 in range(3):
        val=0
        for i2 in range(3):
            val+=tmp2a[i1][i2]
        lsta.append(val)
    
    return np.array(lsta)

cdef double circle_radius(np.ndarray[DTYPE_int_t, ndim=3] triangle,
                        np.ndarray[DTYPE_double_t, ndim=1] centroid):
    #  this transforms a triangle to a circle which covers the triangle
    #  the centre of the circle is the centroid of the triangle.
    cdef int i1,counter
    cdef double dd,radius
    cdef np.ndarray[DTYPE_double_t,ndim=2] tmp2a
    cdef list lsta
    
    lsta=[]
    for i1 in range(3):
        t1,t2,t3,t4,t5,t6=get_vec_numerical(triangle[i1])
        t1,t2,t3=projection3_numerical(t1,t2,t3,t4,t5,t6)
        lsta.append([t1,t2,t3])
    tmp2a=np.array(lsta)
    
    counter=0
    tmp1a=np.array([0])
    radius=0.0
    for i1 in range(3):
        dd=distance(tmp2a[i1],centroid)
        if dd>radius:
            radius=dd
        else:
            pass
    return radius

cdef double distance(np.ndarray[DTYPE_double_t, ndim=1] p1,
                                np.ndarray[DTYPE_double_t, ndim=1] p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)
