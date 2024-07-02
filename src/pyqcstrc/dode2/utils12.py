#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np

from pyqcstrc.dode.math12 cimport dot_product, outer_product, projection, projection3, add, sub, mul, div
from pyqcstrc.dode.numericalc12 cimport point_on_segment, inout_occupation_domain_numerical

DTYPE_double = np.float64
DTYPE_int = np.int64

cdef np.float64_t SIN=np.sqrt(3)/2.0
cdef np.float64_t TOL=1e-6 # tolerance

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int check_two_vertices(np.ndarray[DTYPE_int_t, ndim=2] vertex1,
                            np.ndarray[DTYPE_int_t, ndim=2] vertex2,
                            int verbose):
    #"""
    cdef np.ndarray[DTYPE_int_t,ndim=1] a1,a2,a3,b1,b2,b3
    a1,a2,a3=projection3(vertex1[0],vertex1[1],vertex1[2],vertex1[3],vertex1[4],vertex1[5])
    b1,b2,b3=projection3(vertex2[0],vertex2[1],vertex2[2],vertex2[3],vertex2[4],vertex2[5])
    
    if verbose>0:
        print('          check_two_vertices()')
    else:
        pass

    if (np.all(a1==b1) and np.all(a2==b2) and np.all(a3==b3)):
        if verbose>1:
            print('           equivalent')
        else:
            pass
        return 1
    else:
        if verbose>1:
            print('           inequivalent')
        else:
            pass
        return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray two_segment_into_one(np.ndarray[DTYPE_int_t, ndim=3] line_segment_1,
                                    np.ndarray[DTYPE_int_t, ndim=3] line_segment_2,
                                    int verbose):
    cdef int i,counter
    cdef list comb
    cdef np.ndarray[DTYPE_int_t, ndim=1] tmp1
    cdef np.ndarray[DTYPE_int_t, ndim=1] edge1,removed_vtrx
    cdef np.ndarray[DTYPE_int_t, ndim=2] edge1a,edge1b,edge2a,edge2b
    
    if verbose>0:
        print('            two_segment_into_one()')
    else:
        pass

    comb=[[0,1,0,1],[0,1,1,0],[1,0,0,1]]
    #comb=[[0,1,0,1],[0,1,1,0],[1,0,0,1],[1,0,1,0]]
    counter=0

    for i1 in range(len(comb)):
        edge1a=line_segment_1[comb[i1][0]]
        edge1b=line_segment_1[comb[i1][1]]
        edge2a=line_segment_2[comb[i1][2]]
        edge2b=line_segment_2[comb[i1][3]]
        if check_two_vertices(edge1a,edge2a,verbose-1)==1: # equivalent
            if point_on_segment(edge1a,edge1b,edge2b)==0:
                tmp1=np.append(edge1b,edge2b)
                tmp1=np.append(tmp1,edge1a)
                counter+=1
                break
            else:
                pass
        else:
            pass
    if counter!=0:
        return tmp1.reshape(3,6,3)
    else:
        return np.array([[[0]]])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int check_two_edges(np.ndarray[DTYPE_int_t, ndim=3] edge1,
                        np.ndarray[DTYPE_int_t, ndim=3] edge2,
                        int verbose):
    cdef int flag1,flag2
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    
    if verbose>0:
        print('         check_two_edges()')
    else:
        pass
    
    flag1=0
    for i1 in range(2):
        for i2 in range(2):
            flag1+=check_two_vertices(edge1[i1],edge2[i2],verbose-1)
    if flag1==2:
        if verbose>0:
            print('          equivalent')
        else:
            pass
        return 0
    else:
        # edge1   A==B
        # edge2   C==D
        # check, point C and A==B
        flag1=point_on_segment(edge2[0],edge1[0],edge1[1])
        if   flag1==2:  # C is not on a line passing through A and B.
            if verbose>0:
                print('          inequivalent')
            else:
                pass
            return 2
        else:
            # check, point D and A==B
            flag2=point_on_segment(edge2[0],edge1[0],edge1[1])
            if   flag2==2:  # C is not on a line passing through A and B.
                if verbose>0:
                    print('          inequivalent')
                else:
                    pass
                return 2
            else:
                if   flag1== 0 and flag2== 0: #   A==CD=B
                    if verbose>1:
                        print('          A==CD=B')
                    else:
                        pass
                    return 1
                elif flag1==-1 and flag2== 0: # C A==D==B
                    if verbose>1:
                        print('          C A==D==B')
                    else:
                        pass
                    return 2
                elif flag1==-1 and flag2== 1: # C A=====B D
                    if verbose>1:
                        print('          C A=====B D')
                    else:
                        pass
                    return -1
                elif flag1== 0 and flag2== 1: #   A==C==B D
                    if verbose>1:
                        print('          A==C==B D')
                    else:
                        pass
                    return 2
                elif flag1== 0 and flag2==-1: # D A==C==B
                    if verbose>1:
                        print('          D A==C==B')
                    else:
                        pass
                    return 2
                elif flag1== 1 and flag2==-1: # D A=====B C
                    if verbose>1:
                        print('          D A=====B C')
                    else:
                        pass
                    return 2
                elif flag1== 1 and flag2== 0: #   A==D==B C
                    if verbose>1:
                        print('          A==D==B C')
                    else:
                        pass
                    return 2
                else:
                    return 2

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray triangles_to_edges(np.ndarray[DTYPE_int_t, ndim=4] triangles,int verbose):
    # parameter, set of triangles
    # returns, set of edges
    cdef int i1,i2,ji,j2
    cdef list combination
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    
    if verbose>0:
        print('       triangles_to_edges()')
    else:
        pass
    
    tmp1b=np.array([0])
    combination=[[0,1],[0,2],[1,2]]
    for i1 in range(len(triangles)):
        for i2 in range(3):
            j1=combination[i2][0]
            j2=combination[i2][1]
            tmp1a=np.append(triangles[i1][j1],triangles[i1][j2])
            if len(tmp1b)==1:
                tmp1b=tmp1a
            else:
                tmp1b=np.append(tmp1b,tmp1a)
    return tmp1b.reshape(int(len(tmp1b)/36),2,6,3)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray surface_cleaner(np.ndarray[DTYPE_int_t, ndim=4] surface,
                            int num_cycle,
                            int verbose):
    
    # 同一平面上にある三角形を求め、グループ分けし、各グループにおいて、以下を行う．
    # 各グループにおいて、三角形の３辺が、他のどの三角形とも共有していない辺を求める
    # そして、２つの辺が１つの辺にまとめられるのであれば、まとめる
    # 辺の集合をアウトプット
    
    cdef int flag,counter1,counter2,counter3
    cdef int i0,i1,i2,i3,i4,i5
    cdef list list_0,list_1,list_2,skip_list,combination,skip
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c,tmp1d,tmp1e
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b,tmp3c
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,tmp4b,tmp4c
    
    if verbose>0:
        print('       surface_cleaner()')
    else:
        pass
    
    obj_edge_all=np.array([[[[0]]]])
    combination=[[0,1],[0,2],[1,2]]
    
    tmp4a=surface
    tmp4b=triangles_to_edges(tmp4a,verbose-1)
    tmp1c=np.array([0])

    if verbose>0:
        print('            number of edges, %d'%(len(tmp4b)))
    else:
        pass
        
    for i2 in range(len(tmp4a)):
        for i3 in range(len(combination)):
            tmp1a=np.append(tmp4a[i2][combination[i3][0]],tmp4a[i2][combination[i3][1]])
            counter1=0
            for i4 in range(len(tmp4b)):
                flag=check_two_edges(tmp1a.reshape(2,6,3),tmp4b[i4],verbose-1)
                if flag==0: # equivalent
                    counter1+=1
                else:
                    pass
                if counter1==2:
                    break
                else:
                    pass
            if counter1==1:
                if len(tmp1c)==1:
                    tmp1c=tmp1a
                else:
                    tmp1c=np.append(tmp1c,tmp1a)
            else:
                pass
    if verbose>0:
        print('            number of independent edges, %d'%(int(len(tmp1c)/36)))
    else:
        pass
    
    # ２つの辺が１つの辺にまとめられるのであれば、まとめる
    tmp4c=tmp1c.reshape(int(len(tmp1c)/36),2,6,3)
    for i0 in range(num_cycle):
        tmp1e=np.array([0])
        skip_list=[-1]
        for i2 in range(len(tmp4c)-1):
            for i3 in range(i2+1,len(tmp4c)):
                counter1=0
                for i4 in skip_list:
                    if i2==i4 or i3==i4:
                        counter1+=1
                        break
                    else:
                        pass
                if counter1==0:
                    tmp3a=two_segment_into_one(tmp4c[i2],tmp4c[i3],verbose-1)
                    if len(tmp3a)!=1:
                        skip_list.append(i2)
                        skip_list.append(i3)
                        if len(tmp1e)==1:
                            tmp1e=np.append(tmp3a[0],tmp3a[1])
                        else:
                            tmp1e=np.append(tmp1e,tmp3a[0])
                            tmp1e=np.append(tmp1e,tmp3a[1])
                        break
                    else:
                        pass
                else:
                    pass
        for i2 in range(len(tmp4c)):
            counter1=0
            for i3 in skip_list:
                if i2==i3:
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                if len(tmp1e)==1:
                    tmp1e=tmp4c[i2].reshape(36) # 2*6*3=36
                else:
                    tmp1e=np.append(tmp1e,tmp4c[i2])
            else:
                pass
        if verbose>0:
            print('            %d cycle %d -> %d'%(i0,len(tmp4c),int(len(tmp1e)/36)))
        else:
            pass
        if len(tmp4c)==int(len(tmp1e)/36):
            break
        else:
            pass
        tmp4c=tmp1e.reshape(int(len(tmp1e)/36),2,6,3)

    # 再度、tmp4cに含まれる辺のうち、重複していない辺を求める
    skip=[-1]
    tmp1c=np.array([0])
    counter3=0
    for i1 in range(0,len(tmp4c)-1):
        counter1=0
        counter2=0
        for i2 in skip:
            if i1==i2:
                counter2+=1
                break
            else:
                pass
        #
        if counter2==0:
            for i3 in range(i1+1,len(tmp4c)):
                if check_two_edges(tmp4c[i1],tmp4c[i3],verbose-1)==0: # equivalent
                    counter1+=1
                    skip.append(i3)
                else:
                    pass
        else:
            counter1+=1
        #
        if counter1==0:
            if counter3==0:
                tmp1c=tmp4c[i1].reshape(36)
                counter3+=1
            else:
                tmp1c=np.append(tmp1c,tmp4c[i1])
        else:
            pass
    
    counter1=0
    #print(skip)
    #print(len(tmp4c)-1)
    for i1 in skip:
        if i1==len(tmp4c)-1:
            counter1+=1
            break
        else:
            pass
    if counter1==0:
        tmp1c=np.append(tmp1c,tmp4c[len(tmp4c)-1])
    else:
        pass
        
    if verbose>0:
        print('            number of independent edges, %d'%(int(len(tmp1c)/36)))
    else:
        pass
    

    return tmp1c.reshape(int(len(tmp1c)/36), 2,6,3)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray shift_object(np.ndarray[DTYPE_int_t, ndim=4] obj,
                            np.ndarray[DTYPE_int_t, ndim=2] shift,
                            int vorbose):
    cdef int i1,i2,i3
    cdef long n1,n2,n3
    cdef long v0,v1,v2,v3,v4,v5
    cdef double vol1,vol2
    cdef list a
    cdef np.ndarray[DTYPE_int_t, ndim=4] obj_new
    
    a=[]
    v0,v1,v2=obj_area_6d(obj)
    vol1=(v0+SIN*v1)/v2
    
    if vorbose>0:
        print('        shift_object()')
        if vorbose>1:
            if vorbose>0:
                print('         volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol1))
            else:
                pass
    else:
        pass
    
    for i1 in range(len(obj)):
        for i2 in range(3):
            for i3 in range(6):
                n1,n2,n3=add(obj[i1][i2][i3][0],obj[i1][i2][i3][1],obj[i1][i2][i3][2],shift[i3][0],shift[i3][1],shift[i3][2])
                a.append(n1)
                a.append(n2)
                a.append(n3)
    obj_new=np.array(a).reshape(len(obj),3,6,3)
    
    v3,v4,v5=obj_area_6d(obj_new)
    vol2=(v3+SIN*v4)/v5
    if v0==v3 and v1==v4 and v2==v5 or abs(vol1-vol2)<vol1*TOL:
        if vorbose>0:
            print(' succeeded')
        else:
            pass
        return obj_new
    else:
        print(' fail')
        return np.array([[[[0]]]])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_obj_outline(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                    int verbose):
    #
    # remove doubling segment in a set of triangle in the OD (dim4)
    #
    cdef int i,j,k,val,counter1,counter2,counter3
    #cdef int num1
    cdef list edge,comb
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1k,tmp1j
    #cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    
    if verbose>0:
        print('      generator_obj_outline()')
    else:
        pass
    
    #  edge: 0-1,0-2,1-2
    comb=[[0,1],[0,2],[1,2]]
    
    tmp3=find_unique_segments(obj)
    
    counter2=0
    counter3=0
    for j in range(len(tmp3)): # j-th edge in the unique triangle list 'tmp3'
        tmp1j=tmp3[j].reshape(36)
        counter1=0
        for i in range(0,len(obj)): # i-th triangle in 'obj'
            for k in range(len(comb)): # k-th edge of i-th triangle
                tmp1i=np.append(obj[i][comb[k][0]],obj[i][comb[k][1]])
                val=equivalent_edge(tmp1i,tmp1j) # val=0 (equivalent), 1 (non equivalent)
                if val==0:
                    counter1+=1
                    if counter1==2:
                        break
                    else:
                        pass
                else:
                    pass
        if counter1==1:
            if counter2==0:
                tmp1a=tmp1j
            else:
                tmp1a=np.append(tmp1a,tmp1j)
            counter2+=1
        elif counter1==2:
            pass
        elif counter1>2:
            counter3+=1
        else:
            pass
    if counter3>0:
        if verbose>0:
            print('      ERROR_001 %d: check your model.'%(counter3))
        else:
            pass
        return np.array([[[[0]]]])
    else:
        tmp4a=tmp1a.reshape(int(len(tmp1a)/36),2,6,3)
        if verbose>0:
            print('       Number of triangles on POD surface:%d'%(counter2))
        else:
            pass
        return tmp4a

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray find_unique_segments(np.ndarray[DTYPE_int_t, ndim=4] obj):
    cdef int i,j,k,val,counter1,counter2,counter3
    cdef list edge,comb
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1k,tmp1j
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3
    
    # initialization of tmp2 by 1st triangle
    #  edge: 0-1,0-2,1-2
    comb=[[0,1],[0,2],[1,2]]
    #
    # Three edges of 1-th triangle
    for k in range(len(comb)):
        if k==0:
            tmp2=np.vstack([obj[0][comb[k][0]],obj[0][comb[k][1]]])
        else:
            tmp2=np.vstack([tmp2,obj[0][comb[k][0]]])
            tmp2=np.vstack([tmp2,obj[0][comb[k][1]]])
    tmp3=tmp2.reshape(3,12,3)
    
    for i in range(1,len(obj)): # i-th triangle
        for k in range(len(comb)): # k-th edge of i-th triangle
            tmp1k=np.append(obj[i][comb[k][0]],obj[i][comb[k][1]])
            counter1=0
            for j in range(len(tmp3)): # j-th edge in list 'tmp3'
                tmp1j=tmp3[j].reshape(36)
                val=equivalent_edge(tmp1k,tmp1j) # val=0 (equivalent), 1 (non equivalent)
                if val==0:
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                tmp3=np.vstack([tmp3,tmp1k.reshape(1,12,3)])
            else:
                pass
    return tmp3

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int equivalent_edge(np.ndarray[DTYPE_int_t, ndim=1] edge1,\
                        np.ndarray[DTYPE_int_t, ndim=1] edge2):
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3
    tmp1=np.append(edge1,edge2)
    tmp3=remove_doubling_dim3_in_perp_space(tmp1.reshape(4,6,3))
    if len(tmp3)!=2:
        return 1 # not equivalent
    else:
        return 0 # two edges are equivalent

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray remove_doubling_dim4_in_perp_space(np.ndarray[DTYPE_int_t, ndim=4] obj):
    # remove 6d coordinates which is doubled in perpendicular space
    cdef int num
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    num=len(obj[0])
    tmp3a=obj.reshape(len(obj)*num,6,3)
    tmp3b=remove_doubling_dim3_in_perp_space(tmp3a)
    return tmp3b

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray remove_doubling_dim3_in_perp_space(np.ndarray[DTYPE_int_t, ndim=3] obj):
    # remove 6d coordinates which is doubled in perpendicular space
    cdef int i1,i2,counter1,num
    cdef np.ndarray[DTYPE_int_t,ndim=1] v4,v5,v6
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    num=len(obj)
    if num>1:
        v4,v5,v6=projection3(obj[0][0],obj[0][1],obj[0][2],obj[0][3],obj[0][4],obj[0][5])
        tmp3a=np.array([[v4,v5,v6]]) # perpendicular components
        tmp3b=np.array([obj[0]]) # 6d
        for i1 in range(1,num):
            v4,v5,v6=projection3(obj[i1][0],obj[i1][1],obj[i1][2],obj[i1][3],obj[i1][4],obj[i1][5])
            counter1=0
            for i2 in range(len(tmp3a)):
                if np.all(v4==tmp3a[i2][0]) and np.all(v5==tmp3a[i2][1]) and np.all(v6==tmp3a[i2][2]):
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                tmp3a=np.vstack([tmp3a,[[v4,v5,v6]]])
                tmp3b=np.vstack([tmp3b,[obj[i1]]])
            else:
                pass
        return tmp3b
    else:
        return obj

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray remove_doubling_dim4(np.ndarray[DTYPE_int_t, ndim=4] obj):
    cdef int num
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    num=len(obj[0])
    tmp3a=obj.reshape(len(obj)*num,6,3)
    #tmp3b=remove_doubling_dim3(tmp3a)
    return remove_doubling_dim3(tmp3a)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray remove_doubling_dim3(np.ndarray[DTYPE_int_t, ndim=3] obj):
    cdef int i,j,counter,num
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    tmp3a=np.array([obj[0]])
    for i in range(1,len(obj)):
        #num=len(tmp3a)
        counter=0
        for j in range(0,len(tmp3a)):
            if np.all(obj[i]==tmp3a[j]):
                counter+=1
                break
            else:
                pass
        if counter==0:
            tmp3a=np.vstack([tmp3a,[obj[i]]])
        else:
            pass
    return tmp3a

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list obj_area_6d(np.ndarray[DTYPE_int_t, ndim=4] obj):
    cdef int i
    cdef long v1,v2,v3,w1,w2,w3
    w1,w2,w3=0,0,1
    for i in range(len(obj)):
        [v1,v2,v3]=triangle_area_6d(obj[i])
        w1,w2,w3=add(w1,w2,w3,v1,v2,v3)
    return [w1,w2,w3]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list triangle_area_6d(np.ndarray[DTYPE_int_t, ndim=3] triangle):
    #cdef long v1,v2,v3
    cdef np.ndarray[DTYPE_int_t,ndim=1] x1i,y1i,z1i,x2i,y2i,z2i,x3i,y3i,z3i,x4i,y4i,z4i
    x1i,y1i,z1i=projection3(triangle[0][0],triangle[0][1],triangle[0][2],triangle[0][3],triangle[0][4],triangle[0][5])
    x2i,y2i,z2i=projection3(triangle[1][0],triangle[1][1],triangle[1][2],triangle[1][3],triangle[1][4],triangle[1][5])
    x3i,y3i,z3i=projection3(triangle[2][0],triangle[2][1],triangle[2][2],triangle[2][3],triangle[2][4],triangle[2][5])
    return triangle_area(np.array([x1i,y1i,z1i]),np.array([x2i,y2i,z2i]),np.array([x3i,y3i,z3i]))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list triangle_area(np.ndarray[DTYPE_int_t, ndim=2] v0,
                        np.ndarray[DTYPE_int_t, ndim=2] v1,
                        np.ndarray[DTYPE_int_t, ndim=2] v2):
    cdef long a1,a2,a3,b1,b2,b3,c1,c2,c3
    cdef np.ndarray[DTYPE_int_t, ndim=2] a,b,c

    [a1,a2,a3]=sub(v1[0][0],v1[0][1],v1[0][2],v0[0][0],v0[0][1],v0[0][2])
    [b1,b2,b3]=sub(v1[1][0],v1[1][1],v1[1][2],v0[1][0],v0[1][1],v0[1][2])
    [c1,c2,c3]=sub(v1[2][0],v1[2][1],v1[2][2],v0[2][0],v0[2][1],v0[2][2])
    a=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
    #
    [a1,a2,a3]=sub(v2[0][0],v2[0][1],v2[0][2],v0[0][0],v0[0][1],v0[0][2])
    [b1,b2,b3]=sub(v2[1][0],v2[1][1],v2[1][2],v0[1][0],v0[1][1],v0[1][2])
    [c1,c2,c3]=sub(v2[2][0],v2[2][1],v2[2][2],v0[2][0],v0[2][1],v0[2][2])
    b=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
    #
    c=outer_product(b,a) # cross product
    #print(c)
    #[a1,a2,a3]=dot_product(c[0],c[1],c[2],c[0],c[1],c[2])
    a1=c[2][0]
    a2=c[2][1]
    a3=c[2][2]
    #
    # avoid a negative value
    #
    if a1+a2*SIN<0.0: # to avoid negative volume...
        [a1,a2,a3]=mul(a1,a2,a3,-1,0,2)
    else:
        [a1,a2,a3]=mul(a1,a2,a3,1,0,2)
    return [a1,a2,a3]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list get_points_inside_obj(np.ndarray[DTYPE_int_t, ndim=4] obj, list step, list nstep):
    cdef int i1,i2,i3,verbose
    cdef double x,y,z
    cdef np.ndarray[DTYPE_double_t, ndim=1] point
    cdef list xyz
    xyz=[]
    verbose=0
    for i1 in range(0,nstep[0]+1):
        x=0.0+i1*step[0]
        for i2 in range(0,nstep[1]+1):
            y=0.0+i2*step[1]
            for i3 in range(0,nstep[2]+1):
                z=0.0+i3*step[2]
                point=np.array([x,y,z])
                if inout_occupation_domain_numerical(obj,point,verbose)==0:
                    #print('%8.6f %8.6f %8.6f'%(x,z,y))
                    xyz.append([x,y,z])
                else:
                    pass
    return xyz
