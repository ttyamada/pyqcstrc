#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
from scipy.spatial import Delaunay
cimport numpy as np
cimport cython

from pyqcstrc.dode.math12 cimport projection3, dot_product, add, sub, mul, div
from pyqcstrc.dode.utils12 cimport triangle_area_6d, obj_area_6d, remove_doubling_dim4_in_perp_space, remove_doubling_dim3_in_perp_space
from pyqcstrc.dode.numericalc12 cimport inside_outside_triangle, check_intersection_two_triangles, check_intersection_line_segment_triangle, check_intersection_two_segment_numerical

DTYPE_double = np.float64
DTYPE_int = int

cdef np.float64_t SIN=np.sqrt(3)/2.0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray intersection_two_obj(np.ndarray[DTYPE_int_t, ndim=4] obj1,
                                        np.ndarray[DTYPE_int_t, ndim=4] obj2,
                                        int verbose):
                                        
    cdef int i1,counter
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,tmp4_common
    
    if verbose>0:
        print(' intersection_two_obj()')
    else:
        pass
    
    counter=0
    tmp4_common=np.array([[[[0]]]])
    for i1 in range(len(obj1)):
        tmp4a=intersection_triangle_obj(obj1[i1],obj2,verbose-1)
        if tmp4a.tolist()!=[[[[0]]]]:
            if counter==0:
                 tmp1a=tmp4a.reshape(len(tmp4a)*54)
            else:
                tmp1a=np.append(tmp1a,tmp4a)
            counter+=1
        else:
            pass
    if counter>0:
        tmp4_common=tmp1a.reshape(int(len(tmp1a)/54),3,6,3)
    else:
        pass
    return tmp4_common

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray intersection_triangle_obj(np.ndarray[DTYPE_int_t, ndim=3] triangle,
                                            np.ndarray[DTYPE_int_t, ndim=4] obj,
                                            int verbose):
    cdef int i1,i2,i3,i4,counter1,counter2,counter3,counter4,counter5,q1,q2,q3,flag,num,flag1,flag2
    cdef long v1a,v1b,v1c,v2a,v2b,v2c,v3a,v3b,v3c
    cdef double vol1,vol2,vol3,vol4,vol5
    cdef double dd1
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1b,tmp1c
    cdef np.ndarray[DTYPE_int_t,ndim=2] cent1
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4_common,tmp4a,tmp4b
    
    if verbose>0:
        print('   intersection_triangle_obj()')
    else:
        pass

    tmp1c=np.array([0])
    tmp4_common=np.array([[[[0]]]])
    
    # area check
    v1a,v1b,v1c=triangle_area_6d(triangle)
    vol1=(v1a+v1b*SIN)/v1c
    if verbose>0:
        print('    triangle, %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol1))
    else:
        pass
    
    counter1=0
    for i2 in range(len(obj)):
        if verbose>1:
            v2a,v2b,v2c=triangle_area_6d(obj[i2])
            vol2=(v2a+v2b*SIN)/v2c
            print('    %2d-th triangle in obj2, %d %d %d (%10.8f)'%(i2,v2a,v2b,v2c,vol2))
        else:
            pass
        
        tmp4_common=intersection_two_triangles(triangle,obj[i2],verbose-1)
        
        if tmp4_common.tolist()!=[[[[0]]]]: # if common part is not empty
            if verbose>1:
                v2a,v2b,v2c=obj_area_6d(tmp4_common)
                vol2=(v2a+v2b*SIN)/v2c
                print('     common obj, %d %d %d (%10.8f)'%(v2a,v2b,v2c,vol2))
            else:
                pass
            if counter1==0:
                tmp1c=tmp4_common.reshape(54*len(tmp4_common)) # 3*6*3
            else:
                tmp1c=np.append(tmp1c,tmp4_common)
            counter1+=1
        else:
            pass
    
    if counter1!=0:
        tmp4a=tmp1c.reshape(int(len(tmp1c)/54),3,6,3)
        # area check
        v3a,v3b,v3c=obj_area_6d(tmp4a)
        vol3=(v3a+v3b*SIN)/v3c
        if v3a==v1a and v3b==v1b and v3c==v1c: # vol1==vol3 ＃ この場合、tetrahedron全体が、objに含まれている
            if verbose>0:
                print('     common part (all_1), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol3))
            else:
                pass
            tmp4_common=triangle.reshape(1,3,6,3)
        else:
            if vol3<vol1:
                if verbose>0:
                    print('     common part (partial_1), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol3))
                else:
                    pass
                tmp4_common=tmp4a
            else:
                if verbose>0:
                    print('     common part, %d %d %d (%10.8f) out of expected!'%(v3a,v3b,v3c,vol3))
                else:
                    pass
                tmp4_common=np.array([[[[0]]]])
        if tmp4_common.tolist()!=[[[[0]]]]:
            if verbose>0:
                v1a,v1b,v1c=obj_area_6d(tmp4_common)
                vol2=(v1a+v1b*SIN)/v1c
                print('     common part, area = %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol2))
            else:
                pass
            return tmp4_common
        else:
            if verbose>0:
                print('     common part, empty')
            else:
                pass
            return np.array([[[[0]]]])
        
    else: # if common part (obj1_reduced and obj2) is NOT empty
        if verbose>0:
            print('     common part, empty')
        else:
            pass
        return np.array([[[[0]]]])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray intersection_two_triangles(np.ndarray[DTYPE_int_t, ndim=3] triangle_1,
                                            np.ndarray[DTYPE_int_t, ndim=3] triangle_2,
                                            int verbose):
    cdef int flg
    cdef int i1,i2,j,counter1
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1,tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    cdef list comb
    
    if verbose>0:
        print('       intersection_two_triangles()')
    else:
        pass
    
    tmp4a=np.array([[[[0]]]])
    
    # check two triangles are intersecting or not by numerical calc, (faster)
    flg=check_intersection_two_triangles(triangle_1,triangle_2,verbose-1)
    
    if flg==1: # no intersecting
        return tmp4a
    elif flg==2: # triangle_1 is inside triangle_2
        return triangle_1.reshape(1,3,6,3)
    elif flg==3: # triangle_2 is inside triangle_1
        return triangle_2.reshape(1,3,6,3)
    else: # flg==0: intersecting
        #
        #  edge: 0-1,0-2,1-2
        comb=[[0,1],[0,2],[1,2]]
        counter1=0
        # get intersecting points
        for j in range(len(comb)):
            tmp3=np.vstack([triangle_1[comb[j][0]],triangle_1[comb[j][1]]]).reshape(2,6,3)
            tmp1=intersection_line_segment_triangle(tmp3,triangle_2,verbose-1)
            if tmp1.tolist()!=[0]:
                if counter1==0:
                    tmp1a=tmp1
                else:
                    tmp1a=np.append(tmp1a,tmp1)
                counter1+=1
            else:
                pass
        # get vertces of triangle_1 which are inside triangle_2
        for i1 in range(3):
            if inside_outside_triangle(triangle_1[i1],triangle_2,verbose-1)==0:
                if counter1==0:
                    tmp1a=triangle_1[i1].reshape(18)
                else:
                    tmp1a=np.append(tmp1a,triangle_1[i1])
                counter1+=1
            else:
                pass
        # get vertces of triangle_2 which are inside triangle_1
        for i1 in range(3):
            if inside_outside_triangle(triangle_2[i1],triangle_1,verbose-1)==0:
                if counter1==0:
                    tmp1a=triangle_2[i1].reshape(18)
                else:
                    tmp1a=np.append(tmp1a,triangle_2[i1])
                counter1+=1
            else:
                pass
    
        #tmp4a=np.array([[[[0]]]])
        if counter1>0:
            if verbose>1:
                print('        number of points included in common part: %d'%(int(len(tmp1a)/18)))
            else:
                pass
            # remove doubling
            tmp3a=tmp1a.reshape(int(len(tmp1a)/18),6,3)
            tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
            if len(tmp3a)>=3:
                if verbose>1:
                    print('        number of points for triangulation, %d'%(len(tmp3a)))
                else:
                    pass
                # Triangulation
                if len(tmp3a)==3:
                    tmp4a=tmp3a.reshape(1,3,6,3)
                    if verbose>1:
                        a1,b1,c1=triangle_area_6d(tmp4a[0])
                        print('         area, %d %d %d (%8.6f)'%(a1,b1,c1,(a1+b1*SIN)/c1))
                    else:
                        pass
                else:
                    tmp4a=triangulation_points(tmp3a,verbose-1)
                if verbose>1:
                    print('         -> number of triangles,  %d'%(len(tmp4a)))
                    for i2 in range(len(tmp4a)):
                        a1,b1,c1=triangle_area_6d(tmp4a[i2])
                        print('         %d-th triangle area, %d %d %d (%8.6f)'%(i2,a1,b1,c1,(a1+b1*SIN)/c1))
                else:
                    pass
                return tmp4a
            else:
                return tmp4a
        else:
            return tmp4a

"""
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray intersection_two_triangles(np.ndarray[DTYPE_int_t, ndim=3] triangle_1,
                                            np.ndarray[DTYPE_int_t, ndim=3] triangle_2,
                                            int verbose):
    cdef int j,counter1
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1,tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3
    cdef list comb
    
    if verbose>0:
        print('       intersection_two_triangles()')
    else:
        pass
    
    #  edge: 0-1,0-2,1-2
    comb=[[0,1],[0,2],[1,2]]
    counter1=0
    # get intersecting points
    for j in range(3):
        tmp3=np.vstack([triangle_1[comb[j][0]],triangle_1[comb[j][1]]]).reshape(2,6,3)
        tmp1=intersection_line_segment_triangle(tmp3,triangle_2,verbose-1)
        if tmp1.tolist()!=[0]:
            if counter1==0:
                tmp1a=tmp1
            else:
                tmp1a=np.append(tmp1a,tmp1)
            counter1+=1
        else:
            pass
    # get vertces of triangle_1 which are inside triangle_2
    for i1 in range(3):
        if inside_outside_triangle(triangle_1[i1],triangle_2,verbose-1)==0:
            if counter1==0:
                tmp1a=triangle_1[i1].reshape(18)
            else:
                tmp1a=np.append(tmp1a,triangle_1[i1])
            counter1+=1
        else:
            pass
    # get vertces of triangle_2 which are inside triangle_1
    for i1 in range(3):
        if inside_outside_triangle(triangle_2[i1],triangle_1,verbose-1)==0:
            if counter1==0:
                tmp1a=triangle_2[i1].reshape(18)
            else:
                tmp1a=np.append(tmp1a,triangle_2[i1])
            counter1+=1
        else:
            pass
    
    tmp4a=np.array([[[[0]]]])
    if counter1>=3:
        if verbose>1:
            print('        number of points included in common part: %d'%(int(len(tmp1a)/18)))
            #print('        counter1: %d'%(counter1))
        else:
            pass
        # remove doubling
        tmp3a=tmp1a.reshape(int(len(tmp1a)/18),6,3)
        tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
        if len(tmp3a)>=3:
            if verbose>1:
                print('        number of points for triangulation, %d'%(len(tmp3a)))
            else:
                pass
            # Triangulation
            if len(tmp3a)==3:
                tmp4a=tmp3a.reshape(1,3,6,3)
                if verbose>1:
                    a1,b1,c1=triangle_area_6d(tmp4a[0])
                    vol1=(a1+b1*SIN)/c1
                    print('         area, %d %d %d (%8.6f)'%(a1,b1,c1,vol1))
                else:
                    pass
            else:
                tmp4a=triangulation_points(tmp3a,verbose-1)
            if verbose>1:
                print('         -> number of triangles,  %d'%(len(tmp4a)))
            else:
                pass
            return tmp4a
        else:
            return tmp4a
    else:
        return tmp4a
"""

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray intersection_line_segment_triangle(np.ndarray[DTYPE_int_t, ndim=3] line_segment,
                                                    np.ndarray[DTYPE_int_t, ndim=3] triangle,
                                                    int verbose):
    cdef int j,counter
    cdef np.ndarray[DTYPE_int_t,ndim=1] p,tmp1
    cdef list comb
    
    if verbose>0:
        print('        intersection_line_segment_triangle()')
    else:
        pass
    #  edge: 0-1,0-2,1-2
    comb=[[0,1],[0,2],[1,2]]
    counter=0
    for j in range(3):
        #
        #if check_intersection_two_segment_numerical(line_segment,triangle[comb[j][0]],triangle[comb[j][1]],verbose-1)==0:
        #    tmp1=intersection_two_segment(line_segment[0],line_segment[1],triangle[comb[j][0]],triangle[comb[j][1]],verbose-1)
        #
        #
        tmp1=intersection_two_segment(line_segment[0],line_segment[1],triangle[comb[j][0]],triangle[comb[j][1]],verbose-1)
        if tmp1.tolist()!=[0]:
            if counter==0:
                p=tmp1
            else:
                p=np.append(p,tmp1)
            counter+=1
        else:
            pass
    if counter>0:
        if verbose>0:
            print('         number of intersecting points: %d'%(int(len(p)/18)))
        else:
            pass
        return p
    else:
        if verbose>0:
            print('         no intersecting point found')
        else:
            pass
        return np.array([0])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray intersection_two_segment(np.ndarray[DTYPE_int_t, ndim=2] segment_1_A,
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
    
    if verbose>0:
        print('         intersection_two_segment()')
    else:
        pass
        
    if verbose>1:
        print('          segment_1',segment_1_A[0],segment_1_A[1],segment_1_A[2],segment_1_A[3],segment_1_A[4],segment_1_A[5])
        print('          segment_1',segment_1_B[0],segment_1_B[1],segment_1_B[2],segment_1_B[3],segment_1_B[4],segment_1_B[5])
        print('          segment_2',segment_2_C[0],segment_2_C[1],segment_2_C[2],segment_2_C[3],segment_2_C[4],segment_2_C[5])
        print('          segment_2',segment_2_D[0],segment_2_D[1],segment_2_D[2],segment_2_D[3],segment_2_D[4],segment_2_D[5])
    else:
        pass

    seg1ai1,seg1ai2,seg1ai3=projection3(segment_1_A[0],segment_1_A[1],segment_1_A[2],segment_1_A[3],segment_1_A[4],segment_1_A[5])
    seg1bi1,seg1bi2,seg1bi3=projection3(segment_1_B[0],segment_1_B[1],segment_1_B[2],segment_1_B[3],segment_1_B[4],segment_1_B[5])
    seg2ci1,seg2ci2,seg2ci3=projection3(segment_2_C[0],segment_2_C[1],segment_2_C[2],segment_2_C[3],segment_2_C[4],segment_2_C[5])
    seg2di1,seg2di2,seg2di3=projection3(segment_2_D[0],segment_2_D[1],segment_2_D[2],segment_2_D[3],segment_2_D[4],segment_2_D[5])
    
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
        s=(s1+SIN*s2)/s3
        #t=(-dot_product(vecAC,vecCD)+s*dot_product(vecAB,vecCD))/dot_product(vecCD,vecCD)
        # dot_product(vecAB,vecCD)
        g1,g2,g3=dot_product(np.array([ax1,ax2,ax3]),np.array([ay1,ay2,ay3]),np.array([az1,az2,az3]),np.array([cx1,cx2,cx3]),np.array([cy1,cy2,cy3]),np.array([cz1,cz2,cz3]))
        h1,h2,h3=mul(s1,s2,s3,g1,g2,g3)
        h1,h2,h3=sub(h1,h2,h3,a1,a2,a3)
        if verbose>1:
            print('            s = %8.6f'%(s))
        else:
            pass
        if c1!=0 or c2!=0:
            t1,t2,t3=div(h1,h2,h3,c1,c2,c3)
            t=(t1+SIN*t2)/t3
            if verbose>1:
                print('            t = %8.6f'%(t))
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
                    print('            ddx1,ddx2 = %d %d'%(ddx1,ddx2))
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
                    print('            ddy1,ddy2 = %d %d'%(ddy1,ddy2))
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
                    print('            ddz1,ddz2 = %d %d'%(ddz1,ddz2))
                else:
                    pass
                    
                z1,z2,z3=add(ddx1,ddx2,ddx3,ddy1,ddy2,ddy3)
                z1,z2,z3=add(z1,z2,z3,ddz1,ddz2,ddz3)
                if verbose>1:
                    print('            z1,z2 = %d %d'%(z1,z2))
                else:
                    pass
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
                    if verbose>1:
                        print('            intersecting point:')
                        print('            tmp1a = [[%d,%d,%d],[%d,%d,%d],[%d,%d,%d],[%d,%d,%d],[%d,%d,%d],[%d,%d,%d]]'%(j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18))
                    else:
                        pass
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
cdef np.ndarray centroid_obj(np.ndarray[DTYPE_int_t, ndim=4] obj):
    #  geometric center, centroid of OBJ
    cdef int i1,i2,i3,num
    cdef long v1,v2,v3
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    tmp1a=np.array([0])
    num=len(obj)
    for i3 in range(6):
        v1,v2,v3=0,0,1
        for i1 in range(num):
            for i2 in range(3):
                v1,v2,v3=add(v1,v2,v3,obj[i1][i2][i3][0],obj[i1][i2][i3][1],obj[i1][i2][i3][2])
        v1,v2,v3=mul(v1,v2,v3,1,0,num*3)
        if i3!=0:
            tmp1a=np.append(tmp1a,[v1,v2,v3])
        else:
            tmp1a=np.array([v1,v2,v3])
    return tmp1a.reshape(6,3)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double ball_radius(np.ndarray[DTYPE_int_t, ndim=3] triangle,
                        np.ndarray[DTYPE_int_t, ndim=2] centroid):
    #  this transforms a triangle to a circle which covers the triangle
    #  the centre of the circle is the centroid of the triangle.
    cdef int i1,i2,counter
    cdef long w1,w2,w3
    cdef double dd,radius
    cdef np.ndarray[DTYPE_int_t,ndim=1] v4,v5,v6,tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3b
    
    counter=0
    tmp1a=np.array([0])
    for i1 in range(3):
        for i2 in range(6):
            w1,w2,w3=sub(centroid[i2][0],centroid[i2][1],centroid[i2][2],triangle[i1][i2][0],triangle[i1][i2][1],triangle[i1][i2][2])
            if counter!=0:
                tmp1a=np.append(tmp1a,[w1,w2,w3])
            else:
                tmp1a=np.array([w1,w2,w3])
                counter+=1
    tmp3b=tmp1a.reshape(4,6,3)
    radius=0.0
    for i1 in range(3):
        v4,v5,v6=projection3(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
        dd=np.sqrt(((v4[0]+v4[1]*SIN)/v4[2])**2+((v5[0]+v5[1]*SIN)/v5[2])**2+((v6[0]+v6[1]*SIN)/v6[2])**2)
        if dd>radius:
            radius=dd
        else:
            pass
    return radius

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double ball_radius_obj(np.ndarray[DTYPE_int_t, ndim=4] obj,
                        np.ndarray[DTYPE_int_t, ndim=2] centroid):
    #  this transforms an OBJ to a circle which covers the OBJ
    #  the centre of the circle is the centroid of the OBJ.
    cdef int i1,i2,i3,num1,counter
    cdef long w1,w2,w3
    cdef double dd,radius
    cdef np.ndarray[DTYPE_int_t,ndim=1] v4,v5,v6,tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    
    counter=0
    tmp1a=np.array([0])
    num1=len(obj)
    for i1 in range(num1):
        for i2 in range(3):
            for i3 in range(6):
                w1,w2,w3=sub(centroid[i3][0],centroid[i3][1],centroid[i3][2],obj[i1][i2][i3][0],obj[i1][i2][i3][1],obj[i1][i2][i3][2])
                if counter!=0:
                    tmp1a=np.append(tmp1a,[w1,w2,w3])
                else:
                    tmp1a=np.array([w1,w2,w3])
                    counter+=1
    tmp4a=tmp1a.reshape(num1,3,6,3)
    radius=0.0
    for i1 in range(num1):
        for i2 in range(3):
            v4,v5,v6=projection3(tmp4a[i1][i2][0],tmp4a[i1][i2][1],tmp4a[i1][i2][2],tmp4a[i1][i2][3],tmp4a[i1][i2][4],tmp4a[i1][i2][5])
            dd=np.sqrt(((v4[0]+v4[1]*SIN)/v4[2])**2+((v5[0]+v5[1]*SIN)/v5[2])**2+((v6[0]+v6[1]*SIN)/v6[2])**2)
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
    v4,v5,v6=projection3(tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5])
    dd=np.sqrt(((v4[0]+v4[1]*SIN)/v4[2])**2+((v5[0]+v5[1]*SIN)/v5[2])**2+((v6[0]+v6[1]*SIN)/v6[2])**2)
    return dd

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray triangulation_points(np.ndarray[DTYPE_int_t, ndim=3] points,
                                        int verbose):
    cdef int i,num,counter
    cdef long t1,t2,t3,t4,v1,v2,v3,w1,w2,w3
    cdef double vx,vy,vz
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,xi,yi,zi
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    cdef np.ndarray[DTYPE_double_t,ndim=1] tmp1v
    cdef np.ndarray[DTYPE_double_t,ndim=2] tmp2v
    cdef list ltmp
    
    if verbose>0:
        print('           triangulation_points()')
        print('            number of points: %3d :'%(len(points)))
    else:
        pass
    
    tmp3a=points
    for i in range(len(tmp3a)):
        xi,yi,zi=projection3(tmp3a[i][0],tmp3a[i][1],tmp3a[i][2],tmp3a[i][3],tmp3a[i][4],tmp3a[i][5])
        vx=(xi[0]+xi[1]*SIN)/xi[2] # numeric value of xi
        vy=(yi[0]+yi[1]*SIN)/yi[2] # numeric value of yi
        #vz=(zi[0]+zi[1]*SIN)/zi[2] # numeric value of zi == 0
        if i==0:
            tmp1v=np.array([vx,vy])
        else:
            tmp1v=np.append(tmp1v,[vx,vy])
    tmp2v=tmp1v.reshape(int(len(tmp1v)/2),2)
    ltmp=decomposition(tmp2v)
    
    if verbose>0:
        print('            -> number of triangle: %3d'%(len(ltmp)))
    else:
        pass
    
    tmp4a=np.array([[[[0]]]])
    if ltmp!=[0]:
        counter=0
        for i in range(len(ltmp)):
            tmp3b=np.array([tmp3a[ltmp[i][0]],tmp3a[ltmp[i][1]],tmp3a[ltmp[i][2]]]).reshape(3,6,3)
            v1,v2,v3=triangle_area_6d(tmp3b)
            if v1==0 and v2==0:
                if verbose>0:
                    print('            %d-th tet, empty'%(i))
                    #print(projection3(tmp3b[0][0],tmp3b[0][1],tmp3b[0][2],tmp3b[0][3],tmp3b[0][4],tmp3b[0][5]))
                    #print(projection3(tmp3b[1][0],tmp3b[1][1],tmp3b[1][2],tmp3b[1][3],tmp3b[1][4],tmp3b[1][5]))
                    #print(projection3(tmp3b[2][0],tmp3b[2][1],tmp3b[2][2],tmp3b[2][3],tmp3b[2][4],tmp3b[2][5]))
                    #print(                        tmp3b[0],tmp3b[1],tmp3b[2],tmp3b[3],tmp3b[4],tmp3b[5])
                else:
                    pass
                pass
            else:
                if counter==0:
                    tmp1a=tmp3b.reshape(54) # 3*6*3=54
                    if verbose>0:
                        print('            %d-th tet, area : %d %d %d (%8.6f)'%(i,v1,v2,v3,(v1+v2*SIN)/v3))
                    else:
                        pass
                else:
                    tmp1a=np.append(tmp1a,tmp3b)
                    if verbose>0:
                        print('            %d-th tet, area : %d %d %d (%8.6f)'%(i,v1,v2,v3,(v1+v2*SIN)/v3))
                    else:
                        pass
                counter+=1
        if counter!=0:
            tmp4a=tmp1a.reshape(int(len(tmp1a)/54),3,6,3) # 3*6*3=54
            if verbose>0:
                w1,w2,w3=obj_area_6d(tmp4a)
                print('            -> Total : %d %d %d (%8.6f)'%(w1,w2,w3,(w1+w2*SIN)/w3))
            else:
                pass
        else:
            pass
    else:
        pass
    return tmp4a

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray triangulation(np.ndarray[DTYPE_int_t, ndim=3] triangle,
                                np.ndarray[DTYPE_int_t, ndim=3] intersecting_point,
                                int verbose):
    cdef int i,num,counter
    cdef long t1,t2,t3,t4,v1,v2,v3,w1,w2,w3
    cdef double vx,vy,vz
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,xi,yi,zi
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    cdef np.ndarray[DTYPE_double_t,ndim=1] tmp1v
    cdef np.ndarray[DTYPE_double_t,ndim=2] tmp2v
    cdef list ltmp
    # 
    if verbose>0:
        print('           triangulation()')
        v1,v2,v3=triangle_area_6d(triangle)     # check area of 'triangle'
        print('            area of initial triangle: %d %d %d (%8.6f)'%(v1,v2,v3,(v1+v2*SIN)/v3))
        print('            numbre of points: %3d :'%len(intersecting_point))
    else:
        pass
    
    tmp1a=np.append(triangle,intersecting_point)
    tmp3a=tmp1a.reshape(int(len(tmp1a)/18),6,3)
    
    for i in range(len(tmp3a)):
        xi,yi,zi=projection3(tmp3a[i][0],tmp3a[i][1],tmp3a[i][2],tmp3a[i][3],tmp3a[i][4],tmp3a[i][5])
        vx=(xi[0]+xi[1]*SIN)/xi[2] # numeric value of xi
        vy=(yi[0]+yi[1]*SIN)/yi[2] # numeric value of yi
        #vz=(zi[0]+zi[1]*SIN)/zi[2] # numeric value of zi == 0
        if i==0:
            #tmp1v=np.array([vx,vy,vz])
            tmp1v=np.array([vx,vy])
        else:
            #tmp1v=np.append(tmp1v,[vx,vy,vz])
            tmp1v=np.append(tmp1v,[vx,vy])
    #tmp2v=tmp1v.reshape(int(len(tmp1v)/3),3)
    tmp2v=tmp1v.reshape(int(len(tmp1v)/2),2)
    ltmp=decomposition(tmp2v)
    tmp4a=np.array([[[[0]]]])
    if ltmp!=[0]:
        counter=0
        for i in range(len(ltmp)):
            tmp3b=np.array([tmp3a[ltmp[i][0]],tmp3a[ltmp[i][1]],tmp3a[ltmp[i][2]]]).reshape(3,6,3)
            v1,v2,v3=triangle_area_6d(tmp3b)
            if v1==0 and v2==0:
                pass
            else:
                if counter==0:
                    tmp1a=tmp3b.reshape(72) # 4*6*3=72
                else:
                    tmp1a=np.append(tmp1a,tmp3b)
                counter+=1
        if counter!=0:
            tmp4a=tmp1a.reshape(int(len(tmp1a)/54),3,6,3)
            if verbose>0:
                v1,v2,v3=obj_area_6d(tmp4a)
                print('              -> numbre of triangle: %3d:'%(int(len(tmp4a))))
                print('                 Total area : %d %d %d (%8.6f)'%(v1,v2,v3,(v1+v2*SIN)/v3))
            else:
                pass
        else:
            pass
        return tmp4a
    else:
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
            tmp.append([tet[0],tet[1],tet[2]])
    return tmp
