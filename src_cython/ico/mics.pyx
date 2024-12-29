#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
cimport numpy as np
cimport cython

from pyqcstrc.ico.math1 cimport coplanar_check, projection,projection3, add, triangle_area, centroid
from pyqcstrc.ico.numericalc cimport point_on_segment, projection_numerical, tetrahedron_volume_6d_numerical, obj_volume_6d_numerical, inside_outside_tetrahedron
from pyqcstrc.ico.utils cimport equivalent_edge,generator_xyz_dim4_tmp,remove_doubling_dim3_in_perp_space, remove_doubling_dim4_in_perp_space, obj_volume_6d, tetrahedron_volume_6d, generator_xyz_dim4_tetrahedron, generator_edge, generator_edge_1, generator_surface_1,equivalent_triangle,equivalent_triangle_1
from pyqcstrc.ico.intsct cimport tetrahedralization_points, tetrahedralization, intersection_using_tetrahedron_4, intersection_two_tetrahedron_4, intersection_tetrahedron_obj_4

cdef np.float64_t TAU=(1+np.sqrt(5))/2.0
cdef np.float64_t TOL=1e-6 # tolerance

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

    comb=[[0,1,0,1],[0,1,1,0],[1,0,0,1],[1,0,1,0]]
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

cdef np.ndarray check_two_triangles(np.ndarray[DTYPE_int_t, ndim=3] triange_1,
                                    np.ndarray[DTYPE_int_t, ndim=3] triange_2,
                                    int verbose):
    
    # ２つの三角形をチェックする
    # 同一の場合、
    # １つの頂点を共有し、同一平面にある場合
    # ２つの頂点を共有し、同一平面にある場合
    #
    cdef int i,i1,i2,i3,i4,i5,i6,i7,i8,counter
    cdef np.ndarray[DTYPE_int_t,ndim=1] t1,t2,t3,a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3,e1,e2,e3,f1,f2,f3
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a,tmp2b,tmp2c,tmp2d,tmp2e,tmp2f
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    cdef list comb1
    
    if verbose>0:
        print('            check_two_triangles()')
    else:
        pass
    tmp3a=triange_1
    tmp3b=triange_2
    
    tmp1a=np.append(triange_1,triange_2)
    tmp3b=remove_doubling_dim3_in_perp_space(tmp1a.reshape(int(len(tmp1a)/18),6,3))
    if coplanar_check(tmp3b)==1: # 2つの三角形は同一平面上
        if len(tmp3b)==3: # 2つの三角形は同一
            if verbose>0:
                print('              two triangles are identical')
            else:
                pass
                return triange_1
        elif len(tmp3b)==4: # 2つの三角形は２頂点共有
            comb1=[[0,1,3],[0,2,1],[1,2,0]]
            for i1 in range(len(comb1)):
                [i3,i4,i7]=comb1[i1]
                #t1,t2,t3,a1,a2,a3=projection(tmp3a[i3][0],tmp3a[i3][1],tmp3a[i3][2],tmp3a[i3][3],tmp3a[i3][4],tmp3a[i3][5])
                #t1,t2,t3,c1,c2,c3=projection(tmp3a[i4][0],tmp3a[i4][1],tmp3a[i4][2],tmp3a[i4][3],tmp3a[i4][4],tmp3a[i4][5])
                a1,a2,a3=projection3(tmp3a[i3][0],tmp3a[i3][1],tmp3a[i3][2],tmp3a[i3][3],tmp3a[i3][4],tmp3a[i3][5])
                c1,c2,c3=projection3(tmp3a[i4][0],tmp3a[i4][1],tmp3a[i4][2],tmp3a[i4][3],tmp3a[i4][4],tmp3a[i4][5])
                tmp2a=np.array([a1,a2,a3])
                tmp2c=np.array([c1,c2,c3])
                counter=0
                for i2 in range(len(comb1)):
                    [i5,i6,i8]=comb1[i2]
                    #t1,t2,t3,b1,b2,b3=projection(tmp3b[i5][0],tmp3b[i5][1],tmp3b[i5][2],tmp3b[i5][3],tmp3b[i5][4],tmp3b[i5][5])
                    #t1,t2,t3,d1,d2,d3=projection(tmp3b[i6][0],tmp3b[i6][1],tmp3b[i6][2],tmp3b[i6][3],tmp3b[i6][4],tmp3b[i6][5])
                    b1,b2,b3=projection3(tmp3b[i5][0],tmp3b[i5][1],tmp3b[i5][2],tmp3b[i5][3],tmp3b[i5][4],tmp3b[i5][5])
                    d1,d2,d3=projection3(tmp3b[i6][0],tmp3b[i6][1],tmp3b[i6][2],tmp3b[i6][3],tmp3b[i6][4],tmp3b[i6][5])
                    tmp2b=np.array([b1,b2,b3])
                    tmp2d=np.array([d1,d2,d3])
                    if (np.all(tmp2a==tmp2b) and np.all(tmp2c==tmp2d)) or (np.all(tmp2a==tmp2d) and np.all(tmp2c==tmp2b)):
                        tmp1a=np.append(tmp3a[i7],tmp3b[i8],tmp3a[i3],tmp3a[i4])
                        counter=1
                        break
                    else:
                        pass
                if counter==1:
                    break
                else:
                    print('ERROR')
                    pass
            # 2つの三角形が一つの三角形としてまとめられる場合をチェック
            # (case 1)
            tmp3a=tmp1a.reshape(4,6,3)
            tmp3b=np.append(tmp3a[0],tmp3a[2])
            tmp3c=np.append(tmp3a[1],tmp3a[2])
            tmp3d=two_segment_into_one(tmp3b,tmp3c,verbose-1)
            if len(tmp3d)!=1: # 2つの三角形が一つの三角形としてまとめられる
                tmp1a=np.append(tmp3a[i7],tmp3b[i8],tmp3a[i4])
                tmp3a=tmp1a.reshape(3,6,3)
                if verbose>1:
                    print('              two triangles merged into one')
                else:
                    pass
            # (case 2)
            else:
                tmp3b=np.append(tmp3a[0],tmp3a[3])
                tmp3c=np.append(tmp3a[1],tmp3a[3])
                tmp3d=simplification_convex_polyhedron(tmp3b,tmp3c,1,verbose-1)
                if len(tmp3d)!=1: # 2つの三角形が一つの三角形としてまとめられる
                    tmp1a=np.append(tmp3a[i7],tmp3b[i8],tmp3a[i3])
                    tmp3a=tmp1a.reshape(3,6,3)
                    if verbose>1:
                        print('              two triangles merged into one')
                    else:
                        pass
                else: # まとめられない
                    if verbose>1:
                        print('              two triangles are sharing one edge')
                    else:
                        pass
            return tmp3a

        elif len(tmp3b)==5: # 2つの三角形が1頂点共有
            comb1=[[0,1,3],[0,2,1],[1,2,0]]
            for i1 in range(len(comb1)):
                [i3,i5,i6]=comb1[i1]
                #t1,t2,t3,a1,a2,a3=projection(tmp3a[i3][0],tmp3a[i3][1],tmp3a[i3][2],tmp3a[i3][3],tmp3a[i3][4],tmp3a[i3][5])
                a1,a2,a3=projection3(tmp3a[i3][0],tmp3a[i3][1],tmp3a[i3][2],tmp3a[i3][3],tmp3a[i3][4],tmp3a[i3][5])
                tmp2a=np.array([a1,a2,a3])
                counter=0
                for i2 in [0,1,2]:
                    [i4,i7,i8]=comb1[i2]
                    #t1,t2,t3,b1,b2,b3=projection(tmp3b[i4][0],tmp3b[i4][1],tmp3b[i4][2],tmp3b[i4][3],tmp3b[i4][4],tmp3b[i4][5])
                    b1,b2,b3=projection3(tmp3b[i4][0],tmp3b[i4][1],tmp3b[i4][2],tmp3b[i4][3],tmp3b[i4][4],tmp3b[i4][5])
                    tmp2b=np.array([b1,b2,b3])
                    if np.all(tmp2a==tmp2b):
                        tmp1a=np.append(tmp3a[i5],tmp3a[i6],tmp3b[i7],tmp3b[i8],tmp3a[i3])
                        counter=1
                        break
                    else:
                        pass
                if counter==1:
                    break
                else:
                    print('ERROR')
                    pass
            if verbose>1:
                print('              two triangles are sharing one vertex')
            else:
                pass
            return tmp1a.reshape(5,6,3)
        else:
            if verbose>1:
                print('              not coplaner')
            else:
                pass
                return np.array([[[0]]])

cdef int coplanar_check_two_triangles(np.ndarray[DTYPE_int_t, ndim=3] triange1,
                                        np.ndarray[DTYPE_int_t, ndim=3] triange2,
                                        int verbose):
    cdef int i1
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    tmp1a=np.append(triange1,triange2)
    tmp3a=remove_doubling_dim3_in_perp_space(tmp1a.reshape(int(len(tmp1a)/18),6,3))
    if verbose>0:
        print('        coplanar_check_two_triangles()')
    else:
        pass
    if coplanar_check(tmp3a)==1:
        if verbose>0:
            print('         coplanar')
        else:
            pass
        return 1 # coplanar
    else:
        if verbose>0:
            print('         not coplanar')
        else:
            pass
            return 0

cdef list generate_obj_surface(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                int num_cycle,
                                int verbose):

    cdef long v0,v1,v2,v3,v4,v5
    cdef double vol0,vol1
    cdef list surface_list
    cdef np.ndarray[DTYPE_int_t, ndim=4] obj_surface
    
    if verbose>0:
        print('      generate_obj_surface()')
    else:
        pass
        
    v0,v1,v2=obj_volume_6d(obj)
    #vol0=(v0+TAU*v1)/float(v2)
    vol1=obj_volume_6d_numerical(obj)
    if verbose>1:
        #print '       initial volume: %d %d %d (%10.8f) (%10.8f)'%(v0,v1,v2,vol1)
        print('       initial volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol1))
    else:
        pass
    
    #obj_surface=generator_surface(obj)
    obj_surface=generator_surface_1(obj,verbose-1)
    surface_list=surface_cleaner(obj_surface,num_cycle,verbose-1)
    
    return surface_list

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
    """
    # numerical calc
    cdef double x0,y0,z0,x1,y1,z1
    cdef list tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t,ndim=1] t1,t2,t3,a1,a2,a3,b1,b2,b3
    if verbose>0:
        print('          check_two_vertices()')
    else:
        pass
    tmp1a=projection_numerical((vertex1[0][0]+TAU*vertex1[0][1])/(vertex1[0][2]),\
                                    (vertex1[1][0]+TAU*vertex1[1][1])/(vertex1[1][2]),\
                                    (vertex1[2][0]+TAU*vertex1[2][1])/(vertex1[2][2]),\
                                    (vertex1[3][0]+TAU*vertex1[3][1])/(vertex1[3][2]),\
                                    (vertex1[4][0]+TAU*vertex1[4][1])/(vertex1[4][2]),\
                                    (vertex1[5][0]+TAU*vertex1[5][1])/(vertex1[5][2]))
    x0,y0,z0=tmp1a[3],tmp1a[4],tmp1a[5]
    tmp1b=projection_numerical((vertex2[0][0]+TAU*vertex2[0][1])/(vertex2[0][2]),\
                                    (vertex2[1][0]+TAU*vertex2[1][1])/(vertex2[1][2]),\
                                    (vertex2[2][0]+TAU*vertex2[2][1])/(vertex2[2][2]),\
                                    (vertex2[3][0]+TAU*vertex2[3][1])/(vertex2[3][2]),\
                                    (vertex2[4][0]+TAU*vertex2[4][1])/(vertex2[4][2]),\
                                    (vertex2[5][0]+TAU*vertex2[5][1])/(vertex2[5][2]))
    x1,y1,z1=tmp1b[3],tmp1b[4],tmp1b[5]
    if abs(x0-x1)<TOL and abs(y0-y1)<TOL and abs(z0-z1)<TOL:
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
    """
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
                
cdef np.ndarray merge_two_edges(np.ndarray[DTYPE_int_t, ndim=3] edge1,
                                np.ndarray[DTYPE_int_t, ndim=3] edge2,
                                int verbose):
    cdef int flag1,flag2
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a

    if verbose>0:
        print('           merge_two_edges()')
    else:
        pass
    
    flag1=0
    for i1 in range(2):
        for i2 in range(2):
            flag1+=check_two_vertices(edge1[i1],edge1[i2],verbose)
    if flag1==2:
        if verbose>0:
            print('           merged')
        else:
            pass
        return edge1
    else:
        # edge1   A==B
        # edge2   C==D
        # check, point C and A==B
        flag1=point_on_segment(edge2[0],edge1[0],edge1[1])
        if   flag1==2:  # C is not on a line passing through A and B.
            if verbose>0:
                print('           cannot merged')
            else:
                pass
            return np.array([0]).reshape(1,1,1)
        else:
            # check, point D and A==B
            flag2=point_on_segment(edge2[0],edge1[0],edge1[1])
            if   flag1==2:  # C is not on a line passing through A and B.
                if verbose>0:
                    print('           cannot merged')
                else:
                    pass
                return np.array([0]).reshape(1,1,1)
            else:
                if verbose>0:
                    print('           merged')
                else:
                    pass
                if   flag1== 0 and flag2== 0: #   A==CD=B
                    return edge1
                elif flag1==-1 and flag2== 0: # C A==D==B
                    tmp1a=np.append(edge2[0],edge1[1])
                    return tmp1a.reshape(2,6,3)
                elif flag1==-1 and flag2== 1: # C A=====B D
                    tmp1a=np.append(edge2[0],edge2[1])
                    return tmp1a.reshape(2,6,3)
                elif flag1== 0 and flag2== 1: #   A==C==B D
                    tmp1a=np.append(edge1[0],edge2[1])
                    return tmp1a.reshape(2,6,3)
                elif flag1== 0 and flag2==-1: # D A==C==B
                    tmp1a=np.append(edge2[1],edge1[1])
                    return tmp1a.reshape(2,6,3)
                elif flag1== 1 and flag2==-1: # D A=====B C
                    tmp1a=np.append(edge2[1],edge2[0])
                    return tmp1a.reshape(2,6,3)
                elif flag1== 1 and flag2== 0: #   A==D==B C
                    tmp1a=np.append(edge1[0],edge1[0])
                    return tmp1a.reshape(2,6,3)
                    
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


# surface_cleaner()と比べて、計算効率は悪い？
cdef list surface_cleaner_1(np.ndarray[DTYPE_int_t, ndim=4] surface,
                                int num_cycle,
                                int verbose):
    
    # 同一平面上にある三角形を求め、グループ分けする
    # 各グループにおいて、以下を行う．
    # 各グループにおいて、三角形の３辺が、他のどの三角形とも共有していない辺を求める
    # そして、２つの辺が１つの辺にまとめられるのであれば、まとめる
    # 辺の集合をアウトプット
    
    cdef int flag,counter1,counter2,counter3,counter4,num
    cdef int i0,i1,i2,i3,i4,i5
    cdef list list_0,list_1,list_2,skip_list,combination
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c,tmp1d,tmp1e
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b,tmp3c
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,tmp4b,tmp4c
    
    if verbose>0:
        print('       surface_cleaner_1()')
    else:
        pass
    
    obj_edge_all=np.array([[[[0]]]])
    
    # 同一平面上にある三角形の集合
    list_0=[]
    list_2=[]
    skip_list=[-1]
    for i1 in range(len(surface)-1):
        list_1=[]
        counter2=0
        for i2 in range(i1+1,len(surface)):
            counter1=0
            for i3 in skip_list:
                if i1==i3 or i2==i3:
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                flag=coplanar_check_two_triangles(surface[i1],surface[i2],verbose-1)
                if verbose>1:
                    print('         %3d %3d %3d'%(i1,i2,flag))
                else:
                    pass
                if flag==1: # coplanar
                    if len(list_1)==0:
                        list_1.append(i1)
                    else:
                        pass
                    skip_list.append(i2)
                    list_1.append(i2)
                else:
                    pass
                counter2+=1
            else:
                pass
        if counter2!=0:
            if len(list_1)!=0:
                list_0.append(list_1)
            else:
                list_0.append([i1])
        else:
            pass
    # check the last triangle in 'surface'
    counter1=0
    for i1 in skip_list:
        if i1==len(surface)-1:
            counter1+=1
            break
        else:
            pass
    if counter1==0:
        list_0.append([len(surface)-1])
    else:
        pass
    
    if verbose>0:
        print('        number of set of coplanar triangles, %d'%(len(list_0)))
    else:
        pass

    for i1 in range(len(list_0)):
        counter2=0 # counter for tmp1c
        counter3=0 # counter for tmp1a
        if verbose>0:
            print('          %d-th set of triangles'%(i1+1))
            print('            number of trianges, %d'%(len(list_0[i1])))
        for i2 in list_0[i1]:
            if counter3==0:
                tmp1a=surface[i2].reshape(54) #3*6*3
                counter3+=1
            else:
                tmp1a=np.append(tmp1a,surface[i2])
        
        #同一平面上にある三角形について、どの三角形とも共有していない独立な辺を求める．
        # O(n^2)なので改善が必要
        tmp4a=tmp1a.reshape(int(len(tmp1a)/54),3,6,3) # set of triangles
        if len(list_0[i1])!=1:
            tmp4b=triangles_to_edges(tmp4a,verbose-1)
            if verbose>0:
                print('            number of edges, %d'%(len(tmp4b)))
            else:
                pass
            for i2 in range(len(tmp4b)):
                counter1=0
                for i3 in range(len(tmp4b)):
                    if i2!=i3:
                        if check_two_edges(tmp4b[i2],tmp4b[i3],verbose-1)==0: # equivalent
                        #if equivalent_edge(tmp4b[i2],tmp4b[i3])==0: # equivalent
                            counter1+=1
                            break
                        else:
                            pass
                    else:
                            pass
                if counter1==0:
                    if counter2==0:
                        tmp1c=tmp4b[i2].reshape(36)
                        counter2+=1
                    else:
                        tmp1c=np.append(tmp1c,tmp4b[i2])
                else:
                    pass
            
            # ２つの辺が１つの辺にまとめられるのであれば、まとめる
            if counter2!=0:
                tmp4c=tmp1c.reshape(int(len(tmp1c)/36),2,6,3)
                if verbose>0:
                    print('            number of reduced edges, %d'%(len(tmp4c)))
                else:
                    pass
                
                for i0 in range(num_cycle):
                    num=len(tmp4c)
                    skip_list=[]
                    counter4=0
                    for i2 in range(num-1):
                        counter3=0
                        for i3 in range(i2+1,num):
                            tmp3a=two_segment_into_one(tmp4c[i2],tmp4c[i3],verbose-1)
                            if len(tmp3a)!=1:
                                tmp1e=np.append(tmp3a[0],tmp3a[1])
                                counter3+=1
                                break
                            else:
                                pass
                        if counter3!=0:
                            counter4+=1
                            break
                        else:
                            pass
                    if counter4!=0:
                        skip_list=[i2,i3]
                        for i4 in range(len(tmp4c)):
                            if i4==skip_list[0]:
                                pass
                            elif i4==skip_list[1]:
                                pass
                            else:
                                tmp1e=np.append(tmp1e,tmp4c[i4])
                        tmp4c=tmp1e.reshape(int(len(tmp1e)/36),2,6,3)
                    else:
                        pass
                    if verbose>0:
                        print('            %d cycle %d -> %d'%(i0+1,num,len(tmp4c)))
                    else:
                        pass
                    
            else:
                pass
        else: # 同一平面に三角形が一つだけある場合
            if verbose>0:
                print('            number of edges, 3')
            else:
                pass
            tmp4c=triangles_to_edges(tmp4a,verbose-1)
        
        # Merge
        list_2.append(tmp4c)
        
    return list_2
    
# surface_cleaner_1()と比べて、計算効率はよい？
cdef list surface_cleaner(np.ndarray[DTYPE_int_t, ndim=4] surface,
                            int num_cycle,
                            int verbose):
    
    # 同一平面上にある三角形を求め、グループ分けする
    # 各グループにおいて、以下を行う．
    # 各グループにおいて、三角形の３辺が、他のどの三角形とも共有していない辺を求める
    # そして、２つの辺が１つの辺にまとめられるのであれば、まとめる
    # 辺の集合をアウトプット
    
    cdef int flag,counter1,counter2
    cdef int i0,i1,i2,i3,i4,i5
    cdef list list_0,list_1,list_2,skip_list,combination
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c,tmp1d,tmp1e
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b,tmp3c
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,tmp4b,tmp4c
    
    if verbose>0:
        print('       surface_cleaner()')
    else:
        pass
    
    obj_edge_all=np.array([[[[0]]]])
    combination=[[0,1],[0,2],[1,2]]

    # 同一平面上にある三角形を求め、集合Aとする
    list_0=[]
    list_2=[]
    skip_list=[-1]
    for i1 in range(len(surface)-1):
        tmp1a=np.array([0])
        tmp1b=np.array([0])
        tmp1c=np.array([0])
        list_1=[]
        counter2=0
        for i2 in range(i1+1,len(surface)):
            counter1=0
            for i3 in skip_list:
                if i1==i3 or i2==i3:
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                flag=coplanar_check_two_triangles(surface[i1],surface[i2],verbose-1)
                if verbose>1:
                    print('         %3d %3d %3d'%(i1,i2,flag))
                else:
                    pass
                if flag==1: # coplanar
                    if len(list_1)==0:
                        list_1.append(i1)
                    else:
                        pass
                    skip_list.append(i2)
                    list_1.append(i2)
                else:
                    pass
                counter2+=1
            else:
                pass
        if counter2!=0:
            if len(list_1)!=0:
                list_0.append(list_1)
            else:
                list_0.append([i1])
        else:
            pass
    # check the last triangle in 'surface'
    counter1=0
    for i1 in skip_list:
        if i1==len(surface)-1:
            counter1+=1
            break
        else:
            pass
    if counter1==0:
        list_0.append([len(surface)-1])
    else:
        pass
    
    if verbose>0:
        print('        number of set of coplanar triangles, %d'%(len(list_0)))
    else:
        pass
    
    tmp1d=np.array([0])
    for i1 in range(len(list_0)):
        tmp1a=np.array([0])
        tmp1c=np.array([0])
        tmp1e=np.array([0])
        if verbose>0:
            print('          %d-th set of triangles'%(i1+1))
            print('            number of trianges, %d'%(len(list_0[i1])))
            #print '            ',list_0[i1]
        for i2 in list_0[i1]:
            if len(tmp1a)==1:
                tmp1a=surface[i2].reshape(54) #3*6*3
            else:
                tmp1a=np.append(tmp1a,surface[i2])
        
        #同一平面上にある三角形について、どの三角形とも共有していない独立な辺を求める．
        tmp4a=tmp1a.reshape(int(len(tmp1a)/54),3,6,3) # set of triangles
        if len(list_0[i1])!=1:
            tmp4b=triangles_to_edges(tmp4a,verbose-1)
            if verbose>0:
                print('            number of edges, %d'%(len(tmp4b)))
            else:
                pass
            for i2 in range(len(tmp4a)):
                for i3 in range(len(combination)):
                    j1=combination[i3][0]
                    j2=combination[i3][1]
                    tmp1a=np.append(tmp4a[i2][j1],tmp4a[i2][j2])
                    tmp3a=tmp1a.reshape(2,6,3)
                    counter1=0
                    for i4 in range(len(tmp4b)):
                        flag=check_two_edges(tmp3a,tmp4b[i4],verbose-1)
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
            if len(tmp1c)!=1:
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
                                #tmp3a=merge_two_edges(tmp4c[i2],tmp4c[i3],verbose)
                                if len(tmp3a)!=1:
                                    #if len(skip_list)==1:
                                    #    skip_list.append(i2)
                                    #else:
                                    #    pass
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
        
        else: # 同一平面に三角形が一つだけある場合
            if verbose>0:
                print('            number of independent edges, 3')
            else:
                pass
            tmp4c=triangles_to_edges(tmp4a,verbose-1)
        
        # Merge
        list_2.append(tmp4c)
        
    return list_2
    
cdef np.ndarray chech_four_edges_forms_tetrahedron(np.ndarray[DTYPE_int_t, ndim=3] edges1,\
                                            np.ndarray[DTYPE_int_t, ndim=3] edges2,\
                                            np.ndarray[DTYPE_int_t, ndim=3] edges3,\
                                            np.ndarray[DTYPE_int_t, ndim=3] edges4,\
                                            np.ndarray[DTYPE_int_t, ndim=3] edges5,\
                                            np.ndarray[DTYPE_int_t, ndim=3] edges6,\
                                            int verbose):
    # 与えられた4つの辺が4面体を成すかを判定
    # Judgement: whether given four edges form a tetrahedron or not.
    if verbose>0:
        print('       chech_four_edges_forms_tetrahedron()')
    else:
        pass
    cdef np.ndarray[DTYPE_int_t, ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t, ndim=3] tmp3a,tmp3b
    cdef np.ndarray[DTYPE_int_t, ndim=4] tmp4a
    
    tmp1a=np.append(edges1,edges2)
    tmp1a=np.append(tmp1a,edges3)
    tmp1a=np.append(tmp1a,edges4)
    tmp1a=np.append(tmp1a,edges5)
    tmp1a=np.append(tmp1a,edges6)
    tmp3b=tmp1a.reshape(int(len(tmp1a)/18),6,3)
    
    tmp3a=remove_doubling_dim3_in_perp_space(tmp3b)
    
    if len(tmp3a)==4: 
        # 与えられた4つの辺の各頂点について、独立なものが4つだけの場合、4面体を成す
        # Given four edges form a tetrahedron
        tmp4a=tetrahedralization_points(tmp3a,verbose-1)
        if verbose>0:
            print('        found')
        else:
            pass
        return tmp4a
    else:
        if verbose>0:
            print('        not found')
        else:
            pass
        return np.array([[[[0]]]])
    
cdef np.ndarray search_polyhedron_in_surfaces(np.ndarray[DTYPE_int_t, ndim=4] edges,int verbose):
    # 与えられた面から、4面体を探す
    return 0

cdef np.ndarray search_tetrahedron_in_edges(np.ndarray[DTYPE_int_t, ndim=4] edges,int verbose):
    # 与えられた辺の中で、4面体を探す
    cdef int flag,i1,i2,i3,i4,i5,i6
    cdef double vol1
    cdef np.ndarray[DTYPE_int_t, ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t, ndim=4] tmp4a
    
    if verbose>0:
        print('          search_tetrahedron_in_edges()')
    else:
        pass

    if verbose>0:
        print('           number of egses, %d'%(len(edges)))
    else:
        pass
    
    tmp1b=np.array([0])
    if len(edges)>5:
        for i1 in range(len(edges)-5):
            for i2 in range(i1+1,len(edges)-4):
                for i3 in range(i1+2,len(edges)-3):
                    for i4 in range(i1+3,len(edges)-2):
                        for i5 in range(i1+3,len(edges)-1):
                            for i6 in range(i1+3,len(edges)):
                                tmp4a=chech_four_edges_forms_tetrahedron(edges[i1],edges[i2],edges[i3],edges[i4],edges[i5],edges[i6],verbose+1)
                                if len(tmp4a)!=1:
                                    if verbose>0:
                                        vol1=tetrahedron_volume_6d_numerical(tmp4a)
                                        print('            volume, %10.8f'%(vol1))
                                    else:
                                        pass
                                    if len(tmp1b)==1:
                                        tmp1b=tmp4a.reshape(72)
                                    else: 
                                        tmp1b=np.append(tmp1b,tmp4a)
                                else:
                                    pass
        if len(tmp1b)!=1:
            tmp4a=tmp1b.reshape(int(len(tmp1b)/72),4,6,3)
            if verbose>0:
                print('          number of tetrahedron, %d'%(len(tmp4a)))
                vol1=obj_volume_6d_numerical(tmp4a)
                print('            total volume, %10.8f'%(vol1))
            else:
                pass
            return tmp1b.reshape(int(len(tmp1b)/72),4,6,3)
        else:
            return np.array([[[[0]]]])
    else:
        return np.array([[[[0]]]])
    
cdef np.ndarray extract_common_vertex(np.ndarray[DTYPE_int_t, ndim=4] obj,int verbose):
    # 全ての四面体に共通する頂点を得る
    cdef int flag1,counter1,counter2
    cdef int i1,i2,i3
    cdef np.ndarray[DTYPE_int_t, ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t, ndim=3] tmp3a
    
    if verbose>0:
        print('       extract_common_vertex()')
    else:
        pass
    
    tmp1a=np.array([0])
    tmp3a=remove_doubling_dim4_in_perp_space(obj)
    for i1 in range(len(tmp3a)):
        counter2=0
        for i2 in range(len(obj)):
            counter1=0
            for i3 in range(4):
                flag1=check_two_vertices(tmp3a[i1],obj[i2][i3],verbose-1)
                if flag1==1: # equivalent
                    counter1+=1
                    break
                else:
                    pass
            if counter1!=0:
                pass
            else:
                counter2+=1
        if counter2==0:
            if len(tmp1a)==1:
                tmp1a=tmp3a[i1].reshape(18)
            else:
                tmp1a=np.append(tmp1a,tmp3a[i1])
        else:
            pass
    if len(tmp1a)!=1:
        if verbose>0:
            print('        number of common vertex, %d'%(int(len(tmp1a)/18)))
        else:
            pass
        return tmp1a.reshape(int(len(tmp1a)/18),6,3)
    else:
        if verbose>0:
            print('        number of common vertex, 0')
        else:
            pass
        
        return np.array([[[0]]])

cdef np.ndarray extract_surface_without_specific_vertx(np.ndarray[DTYPE_int_t, ndim=4] triangles,
                                                        np.ndarray[DTYPE_int_t, ndim=3] points,
                                                        int verbose):
    # 三角形の集まりから、特定の頂点を含まない三角形を取り出す．
    cdef int i1,i2,i3,flag,counter1
    cdef np.ndarray[DTYPE_int_t, ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t, ndim=3] tmp3a
    
    if verbose>0:
        print('       extract_surface_without_specific_vertx()')
    else:
        pass
        
    tmp1a=np.array([0])
    for i1 in range(len(points)):
        for i2 in range(len(triangles)):
            counter1=0
            for i3 in range(3): 
                flag=check_two_vertices(points[i1],triangles[i2][i3],verbose-1)
                if flag==1: # equivalent
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                if len(tmp1a)==1:
                    tmp1a=triangles[i2].reshape(54) # 3*6*3
                else:
                    tmp1a=np.append(tmp1a,triangles[i2])
            else:
                pass
    if len(tmp1a)!=1:
        if verbose>0:
            print('        number of triangles, %d'%(int(len(tmp1a)/54)))
        else:
            pass
        return tmp1a.reshape(int(len(tmp1a)/54),3,6,3)
    else:
        if verbose>0:
            print('        number of triangles, 0')
        else:
            pass
        return np.array([[[[0]]]])

cpdef np.ndarray simplification_convex_polyhedron(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                                int num_cycle,
                                                int verbose,
                                                int option):
    
    # tetrahedralization for convex polyhedron
    cdef int i1
    cdef long v0,v1,v2,v3,v4,v5
    cdef double vol0,vol1,vol2
    cdef list surface_list
    cdef np.ndarray[DTYPE_int_t, ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t, ndim=3] tmp3a
    cdef np.ndarray[DTYPE_int_t, ndim=4] tmp4a

    if verbose>0:
        print('      simplification_convex_polyhedron()')
    else:
        pass
    
    v0,v1,v2=obj_volume_6d(obj)
    vol1=obj_volume_6d_numerical(obj)
    if verbose>0:
        print('       initial volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol1))
    else:
        pass
    
    tmp1a=np.array([0])
    tmp4a=np.array([[[[0]]]])
    surface_list=generate_obj_surface(obj,num_cycle,verbose-1)
    for i1 in range(len(surface_list)):
        if len(tmp1a)==1:
            tmp4a=surface_list[i1]
            tmp1a=tmp4a.reshape(len(tmp4a)*36) # 2*6*3
        else:
            tmp1a=np.append(tmp1a,surface_list[i1])

    tmp3a=tmp1a.reshape(int(len(tmp1a)/18),6,3)
    tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
    tmp4a=tetrahedralization_points(tmp3a,verbose-1)
    
    v3,v4,v5=obj_volume_6d(tmp4a)
    vol2=obj_volume_6d_numerical(tmp4a)
        
    if v3==v0 and v4==v1 and v5==v2 or abs(vol1-vol2)<vol2*TOL:
        if verbose>1:
            print('        succdeded, simplified volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol2))
        else:
            pass
        return tmp4a
        
    else:
        if option==0:
            if verbose>1:
                print('        fail, initial obj returned')
            else:
                pass
            return obj
        else:
            if verbose>1:
                print('        fail, worng obj returned')
            else:
                pass
            return tmp4a

cpdef np.ndarray generate_convex_hull(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                        np.ndarray[DTYPE_int_t, ndim=2] coordinate,
                                        int num_cycle,
                                        int verbose):
    
    cdef int i1,i2,i3,i4,
    cdef int counter0,counter1
    cdef int num0,num1
    cdef long v0,v1,v2
    cdef double vol0
    cdef list skip_list,surface_list
    cdef np.ndarray[DTYPE_int_t, ndim=1] edge_new,removed_vertices
    cdef np.ndarray[DTYPE_int_t, ndim=3] tmp3a,tmp3b
    cdef np.ndarray[DTYPE_int_t, ndim=4] obj_surface,obj_surface_new,obj_edge
    cdef np.ndarray[DTYPE_int_t, ndim=4] tmp4a
    
    if verbose>0:
        print('      generate_convex_hull()')
    else:
        pass
    
    v0,v1,v2=obj_volume_6d(obj)
    vol0=(v0+TAU*v1)/v2
    if verbose>0:
        print('         initial volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol0))
    else:
        pass
    
    obj_surface=generator_surface_1(obj,verbose-1)

    surface_list=surface_cleaner(obj_surface,num_cycle,verbose-1) # こちらのほうが計算効率がよい？
    #surface_list=surface_cleaner_1(obj_surface,num_cycle,verbose-1)
    
    if len(surface_list)>0:
        tmp4a=surface_list[0]
        tmp1a=tmp4a.reshape(len(tmp4a)*36)# 2*6*3=36
        for i1 in range(1,len(surface_list)):
            tmp4a=surface_list[i1]
            tmp1a=np.append(tmp1a,tmp4a)
        obj_surface_new=tmp1a.reshape(int(len(tmp1a)/36),2,6,3)
    else:
        obj_surface_new=surface_list[0]
    
    tmp3b=remove_doubling_dim4_in_perp_space(obj_surface_new)
    if coordinate.tolist()!=[[0]]:
        tmp3b=np.vstack([tmp3b,[coordinate]])
    else:
        pass
    return tetrahedralization_points(tmp3b,verbose-1)
    
cpdef np.ndarray simplification_obj_smart(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                        int num_cycle,
                                        int verbose):
                                        
    # 複数の四面体が一つの頂点を共有して集まっている場合の四面体分割
    cdef int i1
    cdef long v0,v1,v2,v3,v4,v5
    cdef double vol0,vol1,vol2
    cdef list surface_list
    cdef np.ndarray[DTYPE_int_t, ndim=1] tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t, ndim=3] vertex_common
    cdef np.ndarray[DTYPE_int_t, ndim=3] tmp3a
    cdef np.ndarray[DTYPE_int_t, ndim=4] tmp4a
    cdef np.ndarray[DTYPE_int_t, ndim=4] obj_surface,obj_surface_new,obj_tetrahedron
    
    if verbose>0:
        print('      simplification_obj_smart()')
    else:
        pass
        
    v0,v1,v2=obj_volume_6d(obj)
    vol0=(v0+TAU*v1)/float(v2)
    vol1=obj_volume_6d_numerical(obj)
    if verbose>0:
        print('       initial volume: %d %d %d (%10.8f) (%10.8f)'%(v0,v1,v2,vol0,vol1))
    else:
        pass
    
    #obj_surface=generator_surface(obj,verbose-1)
    obj_surface=generator_surface_1(obj,verbose-1)
    
    vertex_common=extract_common_vertex(obj,verbose-1)
    
    obj_surface_new=extract_surface_without_specific_vertx(obj_surface,vertex_common,verbose-1)
        
    surface_list=surface_cleaner(obj_surface_new,num_cycle,verbose-1)
    
    tmp1a=np.array([0])
    if len(vertex_common)==1 and vertex_common.tolist()!=[[[0]]]:
        for i1 in range(len(surface_list)):
            tmp4a=surface_list[i1]
            tmp1b=np.append(tmp4a,vertex_common[0])
            tmp3a=tmp1b.reshape(int(len(tmp1b)/18),6,3)
            tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
            tmp4a=tetrahedralization_points(tmp3a,verbose-1)
            if len(tmp1a)==1:
                tmp1a=tmp4a.reshape(len(tmp4a)*72)
            else:
                tmp1a=np.append(tmp1a,tmp4a)
        if len(tmp1a)!=1:
            if verbose>0:
                print('        number of tetrahedron in reduced obj, %d'%(int(len(tmp1a)/72)))
            return tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
        else:
            if verbose>0:
                print('         fail, initial obj returned')
            else:
                pass
                return obj
    else:
        if verbose>0:
            print('         fail, initial obj returned')
        else:
            pass
        return obj
    
    # ここまで
    """
    # return obj_edge
    tmp3a=remove_doubling_dim4_in_perp_space(obj_edge)
    if len(tmp3a)>=4:
        tmp4a=tetrahedralization_points(tmp3a)
        v3,v4,v5=obj_volume_6d(tmp4a)
        vol2=(v3+TAU*v4)/float(v5)
        vol3=obj_volume_6d_numerical(tmp4a)
        if verbose>=2:
            print '         volume of reduced obj: %d %d %d (%10.8f) (%10.8f)'%(v3,v4,v5,vol2,vol3)
        else:
            pass
        if v3==v0 and v4==v1 and v5==v2 or abs(vol1-vol3)<vol3*TOL:
            if verbose>=2:
                print '         succdeded'
            else:
                pass
            return tmp4a
        else:
            print '         fail'
            if verbose>=1:
                print '         try to get correct obj'
            else:
                pass
            return tmp4a
    else:
        if verbose>=2:
            print '         fail, initial obj returned'
        else:
            pass
        return obj
    """

cpdef np.ndarray simplification_obj_edges_using_parents(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                                    np.ndarray[DTYPE_int_t, ndim=4] obj2,
                                                    np.ndarray[DTYPE_int_t, ndim=4] obj3,
                                                    int num_cycle,
                                                    int verbose_level):
    
    cdef int i1,i2,i3,i4,num,counter1
    cdef long v0,v1,v2,v3,v4,v5,w0,w1,w2,w3,w4,w5
    cdef double vol0,vol1,vol2,vol3,vol4
    cdef list skip_list
    cdef np.ndarray[DTYPE_int_t, ndim=1] tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t, ndim=1] edge_new,removed_vertices
    cdef np.ndarray[DTYPE_int_t, ndim=1] a1,a2,a3,a4,a5,a6,b4,b5,b6,c4,c5,c6
    cdef np.ndarray[DTYPE_int_t, ndim=3] tmp3a,tmp3b,tmp3c
    cdef np.ndarray[DTYPE_int_t, ndim=4] obj_surface,obj_edge,tmp4a,tmp4b,tmp4c,tmp4d
    
    if verbose_level>0:
        print('      simplification_obj_edges_using_parents()')
    else:
        pass
        
    v0,v1,v2=obj_volume_6d(obj)
    vol0=(v0+TAU*v1)/float(v2)
    vol4=obj_volume_6d_numerical(obj)
    if verbose_level>0:
        print('         initial volume: %d %d %d (%10.8f)(%10.8f)'%(v0,v1,v2,vol0,vol4))
    else:
        pass
    
    #obj_surface=generator_surface(obj,verbose_level-1)
    obj_surface=generator_surface_1(obj,verbose_level-1)
    
    obj_edge=generator_edge(obj_surface,verbose_level-1)
    
    for i4 in range(num_cycle):
        skip_list=[-1] # initialize
        
        edge_new=np.array([0])
        removed_vertices=np.array([0])
        for i1 in range(len(obj_edge)-1):
            for i2 in range(i1,len(obj_edge)):
                for i3 in skip_list:
                    if i1!=i3 and i2!=i3:
                        #tmp1a,tmp1b=two_segment_into_one(obj_edge[i1],obj_edge[i2])
                        tmp3c=two_segment_into_one(obj_edge[i1],obj_edge[i2],verbose_level-1)
                        if len(tmp3c)!=1:
                            skip_list.append(i1)
                            skip_list.append(i2)
                            if len(edge_new)==1:
                                #edge_new=tmp1a
                                edge_new=np.append(tmp3c[0],tmp3c[1])
                            else:
                                #edge_new=np.append(edge_new,tmp1a)
                                edge_new=np.append(edge_new,tmp3c[0])
                                edge_new=np.append(edge_new,tmp3c[1])
                            if len(removed_vertices)==1:
                                #removed_vertices=tmp1b
                                removed_vertices=tmp3c[2].reshape(18) # 6*3=36
                            else:
                                #removed_vertices=np.append(removed_vertices,tmp1b)
                                removed_vertices=np.append(removed_vertices,tmp3c[2])
                            break
                        else:
                            pass
                    else:
                        pass
        
        for i1 in range(len(obj_edge)):
            counter1=0
            for i2 in skip_list:
                if i1==i2:
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                if len(edge_new)==1:
                    edge_new=obj_edge[i1].reshape(36) # 2*6*3=36
                else:
                    edge_new=np.append(edge_new,obj_edge[i1])
            else:
                pass
        if len(edge_new)!=1: 
            num=len(obj_edge)
            if len(removed_vertices)!=1:
                obj_edge=edge_new.reshape(int(len(edge_new)/36),2,6,3)
                tmp3a=removed_vertices.reshape(int(len(removed_vertices)/18),6,3)
                edge_new=np.array([0])
                # remove unnecessary line segments
                for i1 in range(len(obj_edge)):
                    counter1=0
                    for i2 in range(len(tmp3a)):
                        #a1,a2,a3,a4,a5,a6=projection(obj_edge[i1][0][0],obj_edge[i1][0][1],obj_edge[i1][0][2],obj_edge[i1][0][3],obj_edge[i1][0][4],obj_edge[i1][0][5])
                        #a1,a2,a3,b4,b5,b6=projection(obj_edge[i1][1][0],obj_edge[i1][1][1],obj_edge[i1][1][2],obj_edge[i1][1][3],obj_edge[i1][1][4],obj_edge[i1][1][5])
                        #a1,a2,a3,c4,c5,c6=projection(tmp3a[i2][0],tmp3a[i2][1],tmp3a[i2][2],tmp3a[i2][3],tmp3a[i2][4],tmp3a[i2][5])
                        a4,a5,a6=projection3(obj_edge[i1][0][0],obj_edge[i1][0][1],obj_edge[i1][0][2],obj_edge[i1][0][3],obj_edge[i1][0][4],obj_edge[i1][0][5])
                        b4,b5,b6=projection3(obj_edge[i1][1][0],obj_edge[i1][1][1],obj_edge[i1][1][2],obj_edge[i1][1][3],obj_edge[i1][1][4],obj_edge[i1][1][5])
                        c4,c5,c6=projection3(tmp3a[i2][0],tmp3a[i2][1],tmp3a[i2][2],tmp3a[i2][3],tmp3a[i2][4],tmp3a[i2][5])

                        if (np.all(a4==c4) and np.all(a5==c5) and np.all(a6==c6)) or (np.all(b4==c4) and np.all(b5==c5) and np.all(b6==c6)):
                            counter1+=1
                            break
                        else:
                            pass
                    if counter1==0:
                        if len(edge_new)==1:
                            edge_new=obj_edge[i1].reshape(36)
                        else:
                            edge_new=np.append(edge_new,obj_edge[i1])
                    else:
                        pass
                tmp4b=edge_new.reshape(int(len(edge_new)/36),2,6,3)
                if verbose_level>0:
                    print('       cycle %d, %d -> %d'%(i4,num,len(tmp4b)))
                else:
                    pass
                obj_edge=tmp4b
            else:
                break
            if num==len(tmp4b):
                break
            else:
                pass
        else:
            pass
    # return obj_edge
    tmp3b=remove_doubling_dim4_in_perp_space(obj_edge)
    if len(tmp3b)>=4:
        tmp4a=tetrahedralization_points(tmp3b,verbose_level-1)
        v3,v4,v5=obj_volume_6d(tmp4a)
        vol1=(v3+TAU*v4)/v5
        vol3=obj_volume_6d_numerical(tmp4a)
        if verbose_level>0:
            print('         volume of reduced obj: %d %d %d (%10.8f)(%10.8f)'%(v3,v4,v5,vol1,vol3))
        else:
            pass
    
        if v3==v0 and v4==v1 and v5==v2 or abs(vol3-vol4)<vol3*TOL:
            print('         succdeded')
            return tmp4a
        else:
            if verbose_level>0:
                print('         fail')
                print('         try to get correct obj')
            else:
                pass
            tmp1a=np.array([0])
            for i1 in range(len(tmp4a)):
                tmp4b=tmp4a[i1].reshape(1,4,6,3)
                w0,w1,w2=obj_volume_6d(tmp4b)
                tmp4c=intersection_using_tetrahedron_4(tmp4b,obj2,verbose_level-1,1)
                if tmp4c.tolist()!=[[[[0]]]]:
                    tmp4c=intersection_using_tetrahedron_4(tmp4c,obj3,verbose_level-1,1)
                else:
                    pass
                if tmp4c.tolist()!=[[[[0]]]]:
                    w3,w4,w5=obj_volume_6d(tmp4c)
                    if w0==w3 and w1==w4 and w2==w5:
                        if len(tmp1a)==1:
                            tmp1a=tmp4b.reshape(72)
                        else:
                            tmp1a=np.append(tmp1a,tmp4b)
                    else:
                        #tmp4c=simplification(tmp4c,5,0,0,2,0,0,1)
                        tmp4d=object_subtraction_2(tmp4b,tmp4c,verbose_level-1)
                        tmp4c=object_subtraction_2(tmp4b,tmp4d,verbose_level-1)
                        if len(tmp1a)==1:
                            tmp1a=tmp4c.reshape(len(tmp4c)*72)
                        else:
                            tmp1a=np.append(tmp1a,tmp4c)
                else:
                    pass
            tmp4a=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
            v3,v4,v5=obj_volume_6d(tmp4a)
            vol2=(v3+TAU*v4)/v5
            vol3=obj_volume_6d_numerical(tmp4a)
            if verbose_level>0:
                print('         volume of reduced obj: %d %d %d (%10.8f)(%10.8f)'%(v3,v4,v5,vol2,vol3))
            else:
                pass
            if v3==v0 and v4==v1 and v5==v2 or abs(vol3-vol4)<vol3*TOL:
                if verbose_level>0:
                    print('         succdeded')
                else:
                    pass
                return tmp4a
            else:
                #if abs(vol1-vol3)<1e-8:
                if abs(vol1-vol3)<vol1*TOL:
                    if verbose_level>0:
                        print('         succdeded')
                    else:
                        pass
                    return tmp4a
                else:
                    if verbose_level>0:
                        print('         fail, initial obj returned')
                    else:
                        pass
                    return obj
                    #return tmp4a
                    #return np.array([0]).reshape(1,1,1,1)
    else:
        if verbose_level>0:
            print('         fail, initial obj returned')
        else:
            pass
        return obj

# NEW
cpdef np.ndarray simplification_obj_edges_1(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                            int num_cycle,
                                            int verbose):
    
    # tetrahedralization for convex polyhedron
    cdef int i1
    cdef long v0,v1,v2,v3,v4,v5
    cdef long w0,w1,w2,w3,w4,w5,
    cdef double vol0,vol1,vol2
    cdef list surface_list
    cdef np.ndarray[DTYPE_int_t, ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t, ndim=3] tmp3a
    cdef np.ndarray[DTYPE_int_t, ndim=4] tmp4a,tmp4b,tmp4c,tmp4d

    if verbose>0:
        print('      simplification_obj_edges_1()')
    else:
        pass
    
    v0,v1,v2=obj_volume_6d(obj)
    #vol0=(v0+TAU*v1)/float(v2)
    vol1=obj_volume_6d_numerical(obj)
    if verbose>0:
        #print '       initial volume: %d %d %d (%10.8f) (%10.8f)'%(v0,v1,v2,vol0,vol1)
        print('       initial volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol1))
    else:
        pass
    
    tmp1a=np.array([0])
    tmp4a=np.array([[[[0]]]])
    
    surface_list=generate_obj_surface(obj,num_cycle,verbose-1)
    
    for i1 in range(len(surface_list)):
        if len(tmp1a)==1:
            tmp4a=surface_list[i1]
            tmp1a=tmp4a.reshape(len(tmp4a)*36) # 2*6*3
        else:
            tmp1a=np.append(tmp1a,surface_list[i1])
    
    tmp3a=tmp1a.reshape(int(len(tmp1a)/18),6,3)
    tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
    tmp4a=tetrahedralization_points(tmp3a,verbose-1)
    
    v3,v4,v5=obj_volume_6d(tmp4a)
    #vol0=(v0+TAU*v1)/float(v2)
    vol2=obj_volume_6d_numerical(tmp4a)
    
    #if (v3==v0 and v4==v1 and v5==v2) or abs(vol1-vol2)<TOL*vol2:
    #if abs(vol1-vol2)/vol2<TOL:
    if v3==v0 and v4==v1 and v5==v2:
        if verbose>0:
            #print '       succdeded, simplified volume: %d %d %d (%10.8f) (%10.8f)'%(v0,v1,v2,vol0,vol2)
            print('       succdeded, simplified volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol2))
        else:
            pass
        return tmp4a
    else:
        if verbose>0:
            print('         fail')
        else:
            pass
        vol3=0.0
        tmp1a=np.array([0])
        for i1 in range(len(tmp4a)):
            tmp4b=tmp4a[i1].reshape(1,4,6,3)
            w0,w1,w2=obj_volume_6d(tmp4b)
            tmp4c=intersection_using_tetrahedron_4(tmp4b,obj,verbose-1,1) # option=1
            #tmp4c=intersection_using_tetrahedron_4(tmp4b,obj,0,verbose-1,1) # option=0
            if tmp4c.tolist()!=[[[[0]]]]:
                w3,w4,w5=obj_volume_6d(tmp4c)
                #
                vol3+=obj_volume_6d_numerical(tmp4c)
                #
                if w0==w3 and w1==w4 and w2==w5:
                    if len(tmp1a)==1:
                        tmp1a=tmp4b.reshape(72)
                    else:
                        tmp1a=np.append(tmp1a,tmp4b)
                else:
                    #tmp4c=simplification(tmp4c,5,0,0,2,0,0,1)
                    #tmp4d=object_subtraction_2(tmp4b,tmp4c,verbose_level)
                    #tmp4c=object_subtraction_2(tmp4b,tmp4d,verbose_level)
                    #
                    #tmp4d=object_subtraction_dev(tmp4b,tmp4c,obj,verbose)
                    #tmp4c=object_subtraction_dev(tmp4b,tmp4d,obj,verbose)
                    if len(tmp1a)==1:
                        tmp1a=tmp4c.reshape(len(tmp4c)*72)
                    else:
                        tmp1a=np.append(tmp1a,tmp4c)
            else:
                pass
        tmp4a=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
        v3,v4,v5=obj_volume_6d(tmp4a)
        #vol2=(v3+TAU*v4)/float(v5)
        #vol3=obj_volume_6d_numerical(tmp4a)
        print('         volume of reduced obj: %d %d %d (%10.8f)'%(v3,v4,v5,vol3))
        if v3==v0 and v4==v1 and v5==v2 or abs(vol3-vol1)/vol3<TOL:
            print('         succdeded')
            return tmp4a
        else:
            if abs(vol1-vol2)<TOL:
                print('         succdeded')
                return tmp4a
            else: 
                print('         fail, initial obj returned')
                return obj

cpdef np.ndarray simplification_obj_edges(np.ndarray[DTYPE_int_t, ndim=4] obj, int num_cycle, int verbose_level):
    
    cdef int i1,i2,i3,i4,num,counter1
    cdef long v0,v1,v2,v3,v4,v5,w0,w1,w2,w3,w4,w5
    cdef double vol0,vol1,vol2,vol3,vol4
    cdef list skip_list
    cdef np.ndarray[DTYPE_int_t, ndim=1] tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t, ndim=1] edge_new,removed_vertices
    cdef np.ndarray[DTYPE_int_t, ndim=1] a1,a2,a3,a4,a5,a6,b4,b5,b6,c4,c5,c6
    cdef np.ndarray[DTYPE_int_t, ndim=3] tmp3a,tmp3b,tmp3c
    cdef np.ndarray[DTYPE_int_t, ndim=4] obj_surface,obj_edge,tmp4a,tmp4b,tmp4c,tmp4d
    
    if verbose_level>0:
        print('      simplification_obj_edges()')
    else:
        pass
        
    v0,v1,v2=obj_volume_6d(obj)
    vol0=(v0+TAU*v1)/(v2)
    vol4=obj_volume_6d_numerical(obj)
    if verbose_level>0:
        print('         initial volume: %d %d %d (%10.8f)(%10.8f)'%(v0,v1,v2,vol0,vol4))
    else:
        pass
    
    #obj_surface=generator_surface(obj)
    obj_surface=generator_surface_1(obj,verbose_level-1)
    
    obj_edge=generator_edge(obj_surface,verbose_level-1)
    
    for i4 in range(num_cycle):
        skip_list=[-1] # initialize
        
        edge_new=np.array([0])
        removed_vertices=np.array([0])
        for i1 in range(len(obj_edge)-1):
            for i2 in range(i1,len(obj_edge)):
                for i3 in skip_list:
                    if i1!=i3 and i2!=i3:
                        tmp3c=two_segment_into_one(obj_edge[i1],obj_edge[i2],verbose_level-1)
                        if len(tmp3c)!=1:
                            skip_list.append(i1)
                            skip_list.append(i2)
                            if len(edge_new)==1:
                                edge_new=np.append(tmp3c[0],tmp3c[1])
                            else:
                                edge_new=np.append(edge_new,tmp3c[0])
                                edge_new=np.append(edge_new,tmp3c[1])
                            if len(removed_vertices)==1:
                                removed_vertices=tmp3c[2].reshape(18) # 6*3=36
                            else:
                                removed_vertices=np.append(removed_vertices,tmp3c[2])
                            break
                        else:
                            pass
                    else:
                        pass
        
        for i1 in range(len(obj_edge)):
            counter1=0
            for i2 in skip_list:
                if i1==i2:
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                if len(edge_new)==1:
                    edge_new=obj_edge[i1].reshape(36) # 2*6*3=36
                else:
                    edge_new=np.append(edge_new,obj_edge[i1])
            else:
                pass
        if len(edge_new)!=1: 
            num=len(obj_edge)
            if len(removed_vertices)!=1:
                obj_edge=edge_new.reshape(int(len(edge_new)/36),2,6,3)
                tmp3a=removed_vertices.reshape(int(len(removed_vertices)/18),6,3)
                edge_new=np.array([0])
                # remove unnecessary line segments
                for i1 in range(len(obj_edge)):
                    counter1=0
                    for i2 in range(len(tmp3a)):
                        #a1,a2,a3,a4,a5,a6=projection(obj_edge[i1][0][0],obj_edge[i1][0][1],obj_edge[i1][0][2],obj_edge[i1][0][3],obj_edge[i1][0][4],obj_edge[i1][0][5])
                        #a1,a2,a3,b4,b5,b6=projection(obj_edge[i1][1][0],obj_edge[i1][1][1],obj_edge[i1][1][2],obj_edge[i1][1][3],obj_edge[i1][1][4],obj_edge[i1][1][5])
                        #a1,a2,a3,c4,c5,c6=projection(tmp3a[i2][0],tmp3a[i2][1],tmp3a[i2][2],tmp3a[i2][3],tmp3a[i2][4],tmp3a[i2][5])
                        a4,a5,a6=projection3(obj_edge[i1][0][0],obj_edge[i1][0][1],obj_edge[i1][0][2],obj_edge[i1][0][3],obj_edge[i1][0][4],obj_edge[i1][0][5])
                        b4,b5,b6=projection3(obj_edge[i1][1][0],obj_edge[i1][1][1],obj_edge[i1][1][2],obj_edge[i1][1][3],obj_edge[i1][1][4],obj_edge[i1][1][5])
                        c4,c5,c6=projection3(tmp3a[i2][0],tmp3a[i2][1],tmp3a[i2][2],tmp3a[i2][3],tmp3a[i2][4],tmp3a[i2][5])
                        if (np.all(a4==c4) and np.all(a5==c5) and np.all(a6==c6)) or (np.all(b4==c4) and np.all(b5==c5) and np.all(b6==c6)):
                            counter1+=1
                            break
                        else:
                            pass
                    if counter1==0:
                        if len(edge_new)==1:
                            edge_new=obj_edge[i1].reshape(36)
                        else:
                            edge_new=np.append(edge_new,obj_edge[i1])
                    else:
                        pass
                tmp4b=edge_new.reshape(int(len(edge_new)/36),2,6,3)
                if verbose_level>0:
                    print('       cycle %d, %d -> %d'%(i4,num,len(tmp4b)))
                else:
                    pass
                obj_edge=tmp4b
            else:
                break
            if num==len(tmp4b):
                break
            else:
                pass
        else:
            pass
    # return obj_edge
    tmp3b=remove_doubling_dim4_in_perp_space(obj_edge)
    if len(tmp3b)>=4:
        tmp4a=tetrahedralization_points(tmp3b,verbose_level-1)
        v3,v4,v5=obj_volume_6d(tmp4a)
        vol1=(v3+TAU*v4)/v5
        vol3=obj_volume_6d_numerical(tmp4a)
        if verbose_level>0:
            print('         volume of reduced obj: %d %d %d (%10.8f)(%10.8f)'%(v3,v4,v5,vol1,vol3))
        else:
            pass
    
        if v3==v0 and v4==v1 and v5==v2 or abs(vol3-vol4)<vol3*TOL:
            if verbose_level>0:
                print('         succdeded')
            else:
                pass
            return tmp4a
        else:
            if verbose_level>0:
                print('         fail, try to get correct obj')
            else:
                pass
            tmp1a=np.array([0])
            for i1 in range(len(tmp4a)):
                tmp4b=tmp4a[i1].reshape(1,4,6,3)
                w0,w1,w2=obj_volume_6d(tmp4b)
                tmp4c=intersection_using_tetrahedron_4(tmp4b,obj,verbose_level-1,1)
                if tmp4c.tolist()!=[[[[0]]]]:
                    w3,w4,w5=obj_volume_6d(tmp4c)
                    if w0==w3 and w1==w4 and w2==w5:
                        if len(tmp1a)==1:
                            tmp1a=tmp4b.reshape(72)
                        else:
                            tmp1a=np.append(tmp1a,tmp4b)
                    else:
                        #tmp4c=simplification(tmp4c,5,0,0,2,0,0,1)
                        tmp4d=object_subtraction_2(tmp4b,tmp4c,verbose_level-1)
                        tmp4c=object_subtraction_2(tmp4b,tmp4d,verbose_level-1)
                        if len(tmp1a)==1:
                            tmp1a=tmp4c.reshape(len(tmp4c)*72)
                        else:
                            tmp1a=np.append(tmp1a,tmp4c)
                else:
                    pass
            tmp4a=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
            v3,v4,v5=obj_volume_6d(tmp4a)
            vol2=(v3+TAU*v4)/float(v5)
            vol3=obj_volume_6d_numerical(tmp4a)
            #print('         volume of reduced obj: %d %d %d (%10.8f)(%10.8f)'%(v3,v4,v5,vol2,vol3))
            if v3==v0 and v4==v1 and v5==v2 or abs(vol3-vol4)<vol3*TOL:
                if verbose_level>0:
                    print('          succdeded')
                else:
                    pass
                return tmp4a
            else:
                if abs(vol1-vol2)<1e-8:
                    if verbose_level>0:
                        print('          succdeded')
                    else:
                        pass
                    return tmp4a
                else: 
                    if verbose_level>2:
                        print('          fail, initial obj returned')
                    else:
                        pass
                    return obj
                    #return tmp4a
                    #return np.array([0]).reshape(1,1,1,1)
    else: 
        if verbose_level>0:
            print('         fail, initial obj returned')
        else:
            pass
        return obj

cdef np.ndarray merge_4_tetrahedra_in_obj(np.ndarray[DTYPE_int_t, ndim=4] obj,int verbose):
    # this simplificates obj (set of tetrahedra)
    cdef int i1,i2,i3,i4,i5,counter0,counter1,counter2,counter3,counter4,counter5,num1,num2,num3
    cdef long v1,v2,v3
    cdef double vol1,vol2
    cdef list a1
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    
    if len(obj)>=4:
        
        if verbose>0:
            print('         merge_4_tetrahedra_in_obj()')
        else:
            pass
    
        # volume of initial obj
        #v1,v2,v3=obj_volume_6d(obj)
        vol1=obj_volume_6d_numerical(obj)
        if verbose>0:
            #print '         volume initial obj = %d %d %d (%8.6f)'%(v1,v2,v3,(v1+TAU*v2)/float(v3))
            print('         volume initial obj = %10.8f'%(vol1))
        else:
            pass
    
        a1=[]
        counter4=0
        tmp1a=np.array([0])
        tmp1b=np.array([0])
        for i1 in range(len(obj)-3):
            counter3=0
            for i2 in range(i1+1,len(obj)-2):
                counter2=0
                for i3 in range(i2+1,len(obj)-1):
                    counter1=0
                    for i4 in range(i3+1,len(obj)):
                        counter0=0
                        if len(a1)!=0:
                            for i5 in a1: # skip tetrahedron already merged
                                if i1!=i5 and i2!=i5 and i3!=i5 and i4!=i5:
                                    pass
                                else:
                                    counter0+=1
                                    break
                        if counter0==0:
                            tmp3a=merge_four_tetrahedra(obj[i1],obj[i2],obj[i3],obj[i4],verbose)
                            if tmp3a.tolist()!=[[[0]]]:
                                if verbose>0:
                                    print('         %d %d %d %d (merged)'%(i1,i2,i3,i4))
                                else:
                                    pass
                                a1.append(i1)
                                a1.append(i2)
                                a1.append(i3)
                                a1.append(i4)
                                if len(tmp1a)==1:
                                    tmp1a=tmp3a.reshape(72)
                                else:
                                    tmp1a=np.append(tmp1a,tmp3a)
                                counter1+=1
                                counter2+=1
                                counter3+=1
                                break
                            else:
                                if verbose>0:
                                    print('         %d %d %d %d'%(i1,i2,i3,i4))
                                else:
                                    pass
                        else:
                            pass
                    if counter1!=0:
                        break
                    else:
                        pass
                if counter2!=0:
                    break
                else:
                    pass
            if counter3!=0:
                break
            else:
                pass
        if len(a1)!=0:
            for i1 in range(len(obj)):
                counter1=0
                for i2 in a1: # skip tetrahedron already merged
                    if i1==i2:
                        counter1+=1
                        break
                    else:
                        pass
                if counter1==0:
                    tmp1a=np.append(tmp1a,obj[i1])
                else:
                    pass
        else:
            tmp1a=obj[0].reshape(72)
            for i1 in range(1,len(obj)):
                tmp1a=np.append(tmp1a,obj[i1])
        

        tmp4a=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
    
        # volume of simplized obj
        #w1,w2,w3=obj_volume_6d(tmp4a)
        vol2=obj_volume_6d_numerical(tmp4a)
        if verbose>0:
            #print '         volume simplified obj = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/float(w3))
            print('         volume simplified obj = %10.8f'%(vol2))
        else:
            pass
    
        #if v1==w1 and v2==w2 and v3==w3:
        if abs(vol1-vol2)<TOL:
            return tmp4a
        else:
            if verbose>0:
                print('         fail')
            else:
                pass    
            return np.array([[[[0]]]])
    else:
        return obj

cdef np.ndarray check_two_tetrahedra(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_1, \
                                        np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_2):
    # check whether tetrahedron_1 and _2 are sharing a triangle surface or not.
    cdef int flag,i1,i2,counter1
    cdef list comb
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a,tmp2b
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    
    # three triangles of tetrahedron: 1-2-3, 1-2-4, 1-3-4, 2-3-4
    comb=[[0,1,2],[0,1,3],[0,2,3],[1,2,3]] 
    # other vertx: 3,2,1,0
    comb1=[3,2,1,0] 
    
    # Four triangles of tetrahedron_1
    for i1 in range(len(comb)): # i1-th triangle of tetrahedron1
        tmp1a=np.append(tetrahedron_1[comb[i1][0]],tetrahedron_1[comb[i1][1]])
        tmp1a=np.append(tmp1a,tetrahedron_1[comb[i1][2]])
        counter1=0
        for i2 in range(len(comb)): # i2-th triangle of tetrahedron2
            tmp1b=np.append(tetrahedron_2[comb[i2][0]],tetrahedron_2[comb[i2][1]])
            tmp1b=np.append(tmp1b,tetrahedron_2[comb[i2][2]])
            # equivalent_triangle()
            # flag=0 (equivalent), 1 (non equivalent)
            #flag=equivalent_triangle(tmp1a,tmp1b)
            flag=equivalent_triangle_1(tmp1a,tmp1b)
            if flag==0:
                counter1+=1
                tmp3a=tmp1a.reshape(3,6,3) # common triangle
                tmp2a=tetrahedron_1[comb1[i1]] # a vertx of tetrahedron_1 which is not vertices of the common triangle.
                tmp2b=tetrahedron_2[comb1[i2]] # a vertx of tetrahedron_2 which is not vertices of the common triangle.
                break
            else:
                pass
        if counter1==1:
            break
        else:
            pass
    if counter1==1:
        tmp1a=np.append(tmp2a,tmp2b)
        tmp1a=np.append(tmp3a,tmp1a)
        return tmp1a.reshape(int(len(tmp1a)/18),6,3)
    else:
        return np.array([[[0]]])
            
cdef np.ndarray merge_three_tetrahedra(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_1, \
                                        np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_2, \
                                        np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_3, \
                                        int verbose):
    # merge three tetrahedra
    cdef int flag,i1,i2,counter1,counter2,counter3
    cdef long a1,b1,c1,a2,b2,c2,a3,b3,c3,w1,w2,w3,v1,v2,v3
    cdef list comb
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b,tmp3c,tmp3d,tmp3e,tmp3f
    
    if verbose>0:
        print('         merge_three_tetrahedra()')
    else:
        pass
    # volume
    a1,b1,c1=tetrahedron_volume_6d(tetrahedron_1)
    a2,b2,c2=tetrahedron_volume_6d(tetrahedron_2)
    a3,b3,c3=tetrahedron_volume_6d(tetrahedron_3)
    v1,v2,v3=add(a1,b1,c1,a2,b2,c2)
    v1,v2,v3=add(v1,v2,v3,a3,b3,c3)
        
    tmp1b=np.append(tetrahedron_1,tetrahedron_2)
    tmp1b=np.append(tmp1b,tetrahedron_3)
    tmp3e=remove_doubling_dim3_in_perp_space(tmp1b.reshape(int(len(tmp1b)/18),6,3))
    
    flag=0
    tmp3d=np.array([[[0]]])
    tmp3a=check_two_tetrahedra(tetrahedron_1,tetrahedron_2)
    if tmp3a.tolist()!=[[[0]]]:
        tmp3b=check_two_tetrahedra(tetrahedron_1,tetrahedron_3)
        if tmp3b.tolist()!=[[[0]]]:
            tmp3c=check_two_tetrahedra(tetrahedron_2,tetrahedron_3)
            if tmp3c.tolist()!=[[[0]]]:
                flag=1
            else:
                pass
        else:
            pass
    else:
        pass
    
    if verbose>0:
        print('         flag=1')
    else:
        pass

    tmp1b=np.array([0])
    tmp1c=np.array([0])
    if flag==1:
        tmp1a=np.append(tmp3a[3],tmp3a[4])
        tmp1a=np.append(tmp1a,tmp3b[3])
        tmp1a=np.append(tmp1a,tmp3b[4])
        tmp1a=np.append(tmp1a,tmp3c[3])
        tmp1a=np.append(tmp1a,tmp3c[4])
        tmp3f=tmp1a.reshape(int(len(tmp1a)/18),6,3)
        tmp3d=remove_doubling_dim3_in_perp_space(tmp3f)
        if verbose>0:
            print('         len(tmp3d)=',len(tmp3d))
        else:
            pass
            
        for i1 in range(len(tmp3e)):
            counter1=0
            for i2 in range(len(tmp3d)):
                if np.all(tmp3e[i1]==tmp3d[i2]):
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                if len(tmp1b)==1:
                    tmp1b=tmp3e[i1].reshape(18)
                else:
                    tmp1b=np.append(tmp1b,tmp3e[i1])
            else:
                pass
        
        #print 'len(tmp1b)/18=',len(tmp1b)/18
        
        tmp3f=np.array([[[0]]])
        tmp3e=tmp1b.reshape(int(len(tmp1b)/18),6,3)
        counter2=0
        counter3=0
        for i1 in range(len(tmp3e)): # len(tmp3e)=2
            tmp1b=np.append(tmp3d,tmp3e[i1])
            if coplanar_check(tmp1b.reshape(int(len(tmp1b)/18),6,3))==0: # not coplanar
                tmp3f=tmp1b.reshape(int(len(tmp1b)/18),6,3)
                counter2+=1
            else:
                counter3+=1
        if counter2==1 and counter3==1: 
            if verbose>0:
                w1,w2,w3=tetrahedron_volume_6d(tmp3f)
                print('         volume merged tet = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/w3))
                print('         volume sum 3  tet = %d %d %d (%8.6f)'%(v1,v2,v3,(v1+TAU*v2)/v3))
                # sum of two tetrahedra is forming a tetrahedron
                print('         merged')
            else:
                pass
            return tmp3f.reshape(4,6,3)
        else:
            return np.array([[[0]]])
    else:
        return np.array([[[0]]])

cdef np.ndarray merge_four_tetrahedra(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_1,\
                                        np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_2,\
                                        np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_3,\
                                        np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_4,\
                                        int verbose):
    # merge four tetrahedra
    cdef int flag,i1,i2,counter1,counter2,counter3
    cdef long a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,w1,w2,w3,v1,v2,v3
    cdef list comb
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b,tmp3c,tmp3d,tmp3e,tmp3f,tmp3g,tmp3h
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,tmp4b
    
    if verbose>0:
        print('         merge_four_tetrahedra()')
    else:
        pass
    """
    # volume
    a1,b1,c1=tetrahedron_volume_6d(tetrahedron_1)
    a2,b2,c2=tetrahedron_volume_6d(tetrahedron_2)
    a3,b3,c3=tetrahedron_volume_6d(tetrahedron_3)
    a4,b4,c4=tetrahedron_volume_6d(tetrahedron_4)
    v1,v2,v3=add(a1,b1,c1,a2,b2,c2)
    v1,v2,v3=add(v1,v2,v3,a3,b3,c3)
    v1,v2,v3=add(v1,v2,v3,a4,b4,c4)
    
    tmp1b=np.append(tetrahedron_1,tetrahedron_2)
    tmp1b=np.append(tmp1b,tetrahedron_3)
    tmp1b=np.append(tmp1b,tetrahedron_4)
    # all vertices of four tetrahedra
    tmp3e=remove_doubling_dim3_in_perp_space(tmp1b.reshape(len(tmp1b)/18,6,3))
    """
    flag=0
    tmp3d=np.array([[[0]]])
    tmp3a=check_two_tetrahedra(tetrahedron_1,tetrahedron_2)
    if tmp3a.tolist()!=[[[0]]]:
        tmp3b=check_two_tetrahedra(tetrahedron_1,tetrahedron_3)
        if tmp3b.tolist()!=[[[0]]]:
            tmp3c=check_two_tetrahedra(tetrahedron_1,tetrahedron_4)
            if tmp3c.tolist()!=[[[0]]]:
                tmp3d=check_two_tetrahedra(tetrahedron_2,tetrahedron_3)
                if tmp3d.tolist()!=[[[0]]]:
                    tmp3e=check_two_tetrahedra(tetrahedron_2,tetrahedron_4)
                    if tmp3e.tolist()!=[[[0]]]:
                        tmp3f=check_two_tetrahedra(tetrahedron_3,tetrahedron_4)
                        if tmp3f.tolist()!=[[[0]]]:
                            #
                            tmp1a=np.append(tmp3a[0],tmp3a[1])
                            tmp1a=np.append(tmp1a,tmp3a[2])
                            tmp1a=np.append(tmp1a,tmp3b[0])
                            tmp1a=np.append(tmp1a,tmp3b[1])
                            tmp1a=np.append(tmp1a,tmp3b[2])
                            tmp1a=np.append(tmp1a,tmp3c[0])
                            tmp1a=np.append(tmp1a,tmp3c[1])
                            tmp1a=np.append(tmp1a,tmp3c[2])
                            tmp1a=np.append(tmp1a,tmp3d[0])
                            tmp1a=np.append(tmp1a,tmp3d[1])
                            tmp1a=np.append(tmp1a,tmp3d[2])
                            tmp1a=np.append(tmp1a,tmp3e[0])
                            tmp1a=np.append(tmp1a,tmp3e[1])
                            tmp1a=np.append(tmp1a,tmp3e[2])
                            tmp1a=np.append(tmp1a,tmp3f[0])
                            tmp1a=np.append(tmp1a,tmp3f[1])
                            tmp1a=np.append(tmp1a,tmp3f[2])
                            tmp3g=tmp1a.reshape(int(len(tmp1a)/18),6,3)
                            tmp3h=remove_doubling_dim3_in_perp_space(tmp3g)
                            #
                            flag=1
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
    else:
        pass
    
    if flag==1:
        tmp1a=np.array([0])
        for i1 in range(len(tmp3h)):
            counter1=0
            for i2 in range(len(tmp3g)):
                if np.all(tmp3h[i1]==tmp3g[i2]):
                    counter1+=1
                else:
                    pass
            #print '%d %d'%(i1,counter1)
            if counter1==6:
                pass
            elif counter1!=6:
                if len(tmp1a)==1:
                    tmp1a=tmp3h[i1].reshape(18)
                else:
                    tmp1a=np.append(tmp1a,tmp3h[i1])
        
        if verbose>1:
            tmp4b=tmp1a.reshape(4,6,3)
            vol2=tetrahedron_volume_6d_numerical(tmp4b)

            tmp1a=np.append(tetrahedron_1,tetrahedron_2)
            tmp1a=np.append(tmp1a,tetrahedron_3)
            tmp1a=np.append(tmp1a,tetrahedron_4)
            tmp4a=tmp1a.reshape(4,4,6,3)
            # Total volume of five tetrahedra
            vol1=obj_volume_6d_numerical(tmp4a)
            
            print('         volume merged tet = %10.8f'%(vol1))
            print('         volume sum 4  tet = %10.8f'%(vol2))
            # sum of two tetrahedra is forming a tetrahedron
            print('         merged')
        else:
            pass
        return tmp4b
    else:
        return np.array([[[0]]])

cdef np.ndarray merge_3_tetrahedra_in_obj(np.ndarray[DTYPE_int_t, ndim=4] obj,int verbose):
    # this simplificates obj (set of tetrahedra)
    cdef int i1,i2,i3,i4,i5,counter1,counter2,counter3,counter4,counter5,num1,num2
    cdef long v1,v2,v3
    cdef double vol1,vol2
    cdef list a1
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    
    # volume of initial obj
    #v1,v2,v3=obj_volume_6d(obj)
    vol1=obj_volume_6d_numerical(obj)
    if verbose>0:
        #print '         volume initial obj = %d %d %d (%8.6f)'%(v1,v2,v3,(v1+TAU*v2)/float(v3))
        print('         volume initial obj = %10.8f'%(vol1))
    else:
        pass
    
    if len(obj)>=3:
        if verbose>0:
            print('        merge_3_tetrahedra_in_obj()')
        else:
            pass
        
        a1=[]
        counter4=0
        tmp1a=np.array([0])
        tmp1b=np.array([0])
        for i1 in range(len(obj)-2):
            counter3=0
            for i2 in range(i1+1,len(obj)-1):
                counter2=0
                for i3 in range(i2+1,len(obj)):
                    counter1=0
                    if len(a1)!=0:
                        for i4 in a1: # skip tetrahedron already merged
                            if i1!=i4 and i2!=i4 and i3!=i4:
                                pass
                            else:
                                counter1+=1
                                break
                    else:
                        pass
                    if counter1==0:
                        tmp3a=merge_three_tetrahedra(obj[i1],obj[i2],obj[i3],verbose)
                        if tmp3a.tolist()!=[[[0]]]:
                            if verbose>1:
                                print('         %d %d %d (merged)'%(i1,i2,i3))
                            else:
                                pass
                            a1.append(i1)
                            a1.append(i2)
                            a1.append(i3)
                            if len(tmp1a)==1:
                                tmp1a=tmp3a.reshape(72)
                            else:
                                tmp1a=np.append(tmp1a,tmp3a)
                            counter2+=1
                        else:
                            if verbose>1:
                                print('         %d %d %d'%(i1,i2,i3))
                            else:
                                pass
                    else:
                        pass
                if counter2!=0:
                    break
                else:
                    pass
                    
        if len(a1)!=0:
            for i1 in range(len(obj)):
                counter1=0
                for i2 in a1: # skip tetrahedron already merged
                    if i1==i2:
                        counter1+=1
                        break
                    else:
                        pass
                if counter1==0:
                    tmp1a=np.append(tmp1a,obj[i1])
                else:
                    pass
        else:
            tmp1a=obj[0].reshape(72)
            for i1 in range(1,len(obj)):
                tmp1a=np.append(tmp1a,obj[i1])
        
        tmp4a=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
    
        # volume of simplized obj
        #w1,w2,w3=obj_volume_6d(tmp4a)
        vol2=obj_volume_6d_numerical(tmp4a)
        if verbose>0:
            #print '         volume simplified obj = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/float(w3))
            print('         volume simplified obj = %10.8f'%(vol2))
        else:
            pass
    
        #if v1==w1 and v2==w2 and v3==w3:
        if abs(vol1-vol2)<TOL:
            return tmp4a
        else:
            if verbose>0:
                print('         fail')
            else:
                pass    
            return np.array([[[[0]]]])
    else:
        return obj

cdef np.ndarray merge_2_tetrahedra_in_obj(np.ndarray[DTYPE_int_t, ndim=4] obj,int verbose):
    # this simplificates obj (set of tetrahedra)
    cdef int i1,i2,i3,i4,counter1,counter2,counter3,counter4,num
    cdef long v1,v2,v3,w1,w2,w3
    cdef double vol1,vol2
    cdef list a1
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    
    if len(obj)>=2:
        if verbose>0:
            print('         merge_2_tetrahedra_in_obj()')
        else:
            pass
    
        # volume of initial obj
        v1,v2,v3=obj_volume_6d(obj)
        vol1=obj_volume_6d_numerical(obj)
        if verbose>0:
            print('         volume initial obj = %d %d %d (%10.8f)'%(v1,v2,v3,(v1+TAU*v2)/v3))
            print('                            = %10.8f'%(vol1))
        else:
            pass
    
        a1=[]
        counter3=0
        tmp1a=np.array([0])
        tmp1b=np.array([0])
        for i1 in range(len(obj)-1):
            counter2=0
            for i2 in range(i1+1,len(obj)):
                counter1=0
                if len(a1)!=0:
                    for i3 in a1: # skip tetrahedron already merged
                        if i1!=i3 and i2!=i3:
                            pass
                        else:
                            counter1+=1
                            break
                else:
                    pass
                if counter1==0:
                    tmp3a=merge_two_tetrahedra(obj[i1],obj[i2],verbose)
                    if tmp3a.tolist()!=[[[0]]]:
                        if verbose>1:
                            print('         %d %d (merged)'%(i1,i2))
                        else:
                            pass
                        a1.append(i1)
                        a1.append(i2)
                        if len(tmp1a)==1:
                            tmp1a=tmp3a.reshape(72)
                        else:
                            tmp1a=np.append(tmp1a,tmp3a)
                        counter2+=1
                        break
                    else:
                        if verbose>1:
                            print('         %d %d'%(i1,i2) )
                        else:
                            pass
                else:
                    pass
                if counter2!=0:
                    break
                else:
                    pass
        
        if len(a1)!=0:
            for i1 in range(len(obj)):
                counter1=0
                for i2 in a1: # skip tetrahedron already merged
                    if i1==i2:
                        counter1+=1
                        break
                    else:
                        pass
                if counter1==0:
                    tmp1a=np.append(tmp1a,obj[i1])
                else:
                    pass
        else:
            tmp1a=obj[0].reshape(72)
            for i1 in range(1,len(obj)):
                tmp1a=np.append(tmp1a,obj[i1])
        
        tmp4a=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
    
        # volume of simplized obj
        w1,w2,w3=obj_volume_6d(tmp4a)
        vol2=obj_volume_6d_numerical(tmp4a)
        if verbose>0:
            print('         volume simplified obj = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/w3))
            print('                               = %10.8f'%(vol2))
        else:
            pass
    
        #if v1==w1 and v2==w2 and v3==w3:
        if abs(vol1-vol2)<TOL:
            return tmp4a
        else:
            if verbose>0:
                print('         fail')
            else:
                pass    
            return np.array([[[[0]]]])
    else:
        return obj

# new
cdef np.ndarray merge_two_tetrahedra(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_1, \
                                        np.ndarray[DTYPE_int_t, ndim=3] tetrahedron_2, \
                                        int verbose):
    # merge two tetrahedra
    cdef int flag,i1,i2,counter1
    cdef long a1,b1,c1,a2,b2,c2,w1,w2,w3,v1,v2,v3
    cdef list comb
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a,tmp2b
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    
    if verbose>0:
        print('         merge_two_tetrahedra()')
    else:
        pass
    # volume
    a1,b1,c1=tetrahedron_volume_6d(tetrahedron_1)
    a2,b2,c2=tetrahedron_volume_6d(tetrahedron_2)
    v1,v2,v3=add(a1,b1,c1,a2,b2,c2)
    
    tmp1a=np.array([0])
    tmp3a=np.array([0]).reshape(1,1,1)
    tmp3a=check_two_tetrahedra(tetrahedron_1,tetrahedron_2)
    if tmp3a.tolist()!=[[[0]]]:
        tmp1a=np.append(tmp3a[3],tmp3a[4])
        counter1=0
        for i1 in range(3):
            flag=point_on_segment(tmp3a[i1],tmp3a[3],tmp3a[4])
            # a vertex of common triangle is on the line segment between tmp2a and tmp2b
            # this means that sum of tetrahedron_1 and _2 forms a tetrahedron.
            if flag==0:
                counter1+=1
            else:
                tmp1a=np.append(tmp1a,tmp3a[i1])
        if counter1==1:
            if  verbose>0:
                w1,w2,w3=tetrahedron_volume_6d(tmp1a.reshape(4,6,3))
                print('         volume merged tet = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/w3))
                print('         volume sum 2  tet = %d %d %d (%8.6f)'%(v1,v2,v3,(v1+TAU*v2)/v3))
                # sum of two tetrahedra is forming a tetrahedron
                print('         merged')
            else:
                pass
            return tmp1a.reshape(4,6,3)
        elif  counter1==0:
            # sum of two tetrahedra is not tetrahedron
            return np.array([[[0]]])
        else:
            # something strange
            return np.array([[[0]]])
    else:
        # no common triangle
        return np.array([[[0]]])

cdef np.ndarray do_merge_tetrahedra_in_obj(np.ndarray[DTYPE_int_t, ndim=4] obj, int numbre_of_cycle, int numbre_of_tetrahedron, int verbose):
    cdef int i1
    cdef np.ndarray[DTYPE_int_t, ndim=4] tmp4a,tmp4b
    
    if verbose>0:
        print('        do_merge_tetrahedra_in_obj()')
    else:
        pass
    
    tmp4a=np.array([[[[0]]]])
    tmp4a=obj
    if len(obj)>=numbre_of_tetrahedron:
        # merging two tetrahedra
        for i1 in range(numbre_of_cycle):
            #
            if numbre_of_tetrahedron==2:
                tmp4b=merge_2_tetrahedra_in_obj(tmp4a,verbose-1)
            elif numbre_of_tetrahedron==3:
                tmp4b=merge_3_tetrahedra_in_obj(tmp4a,verbose-1)
            elif  numbre_of_tetrahedron==4:
                tmp4b=merge_4_tetrahedra_in_obj(tmp4a,verbose-1)
            else:
                pass
            #
            if tmp4b.tolist()!=[[[[0]]]]:
                if verbose>1:
                    print('        simplification: %d -> %d'%(len(tmp4a),len(tmp4b)))
                else:
                    pass
                if len(obj)==len(tmp4b) or len(tmp4b)<numbre_of_tetrahedron:
                    tmp4a=tmp4b
                    break
                else:
                    tmp4a=tmp4b
            else:
                if verbose>0:
                    print('      simplification: fail')
                    print('      return previous obj')
                else:
                    pass
                break
        if len(tmp4a)==len(tmp4b) or len(tmp4b)<numbre_of_tetrahedron:
            if verbose>0:
                print('        succeed: %d to %d'%(len(obj),len(tmp4b)))
            else:
                pass
            tmp4a=tmp4b
        else:
            pass
        return tmp4a
    else:
        return np.array([[[[0]]]])

cpdef np.ndarray simplification(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                int num2_of_cycle,
                                int num3_of_cycle,
                                int num4_of_cycle,
                                int num2_of_shuffle,
                                int num3_of_shuffle, 
                                int num4_of_shuffle,
                                int verbose_level):
    cdef int i1,index_min
    cdef list b
    cdef np.ndarray[DTYPE_int_t, ndim=1] index_list,tmp1
    cdef np.ndarray[DTYPE_int_t, ndim=4] obj1,obj2,obj_new
    
    # verbose_level  0 > 4 (no comment, verbose)
    if verbose_level>=0:
        print('    simplification()')
    else:
        pass
    
    # objに収納されている4面体をまとめる操作は、4面体の順番に大きく影響される.
    # 順番をランダムに変え、最も単純化できたものを採用する.
    
    if num2_of_shuffle>0:
        b=[]
        for i1 in range(num2_of_shuffle):
            obj1=shuffle_obj(obj)
            obj2=do_merge_tetrahedra_in_obj(obj1,num2_of_cycle,2,verbose_level-1)
            if len(obj2)!=1:
                obj_new=obj2
            else:
                obj_new=obj1
            make_tmp_pod(obj_new,i1)
            b.append(len(obj_new))
        #print(b)
        index_min=b.index(min(b))
        obj_min=read_tmp_pod(index_min)
    else:
        pass
    
    if num3_of_shuffle>0:
        b=[]
        for i1 in range(num3_of_shuffle):
            #obj1=shuffle_obj(obj)
            obj1=shuffle_obj(obj_min)
            obj2=do_merge_tetrahedra_in_obj(obj1,num3_of_cycle,3,verbose_level-1)
            if len(obj2)!=1:
                obj_new=obj2
            else:
                obj_new=obj1
            make_tmp_pod(obj_new,i1)
            b.append(len(obj_new))
        index_min=b.index(min(b))
        obj_min=read_tmp_pod(index_min)
    else:
        pass
    
    if num4_of_shuffle>0:
        b=[]
        for i1 in range(num4_of_shuffle):
            #obj1=shuffle_obj(obj)
            obj1=shuffle_obj(obj_min)
            obj2=do_merge_tetrahedra_in_obj(obj1,num4_of_cycle,4,verbose_level-1)
            if len(obj2)!=1:
                obj_new=obj2
            else:
                obj_new=obj1
            make_tmp_pod(obj_new,i1)
            b.append(len(obj_new))
        index_min=b.index(min(b))
        obj_min=read_tmp_pod(index_min)
    else:
        pass
    
    if num2_of_shuffle==0 and num3_of_shuffle==0 and num4_of_shuffle==0:
        obj_min=obj
    else:
        pass
    
    return obj_min

cdef np.ndarray shuffle_obj(np.ndarray[DTYPE_int_t, ndim=4] obj):
    cdef int i1
    cdef list a
    cdef np.ndarray[DTYPE_int_t, ndim=1] index_list,tmp1
    cdef np.ndarray[DTYPE_int_t, ndim=4] obj_new
    
    index_list = np.arange(len(obj))
    np.random.shuffle(index_list)
    a=index_list.tolist()
    
    for i1 in range(len(a)):
        if i1==0:
            tmp1=obj[a[0]].reshape(72)
        else:
            tmp1=np.append(tmp1,obj[a[i1]])
    obj_new=tmp1.reshape(int(len(tmp1)/72),4,6,3)
    return obj_new

######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################


cdef int on_out_surface(np.ndarray[DTYPE_int_t, ndim=2] point,
                        np.ndarray[DTYPE_int_t, ndim=3] triangle):
    
    cdef np.ndarray[DTYPE_int_t, ndim=1] m1,m2,m3,m4,m5,m6
    cdef np.ndarray[DTYPE_int_t, ndim=2] p1,p2,p3
    cdef double volume0,volume1

    #m1,m2,m3,m4,m5,m6=projection(point[0],point[1],point[2],point[3],point[4],point[5])
    m4,m5,m6=projection3(point[0],point[1],point[2],point[3],point[4],point[5])
    p0=np.array([m4,m5,m6])
    #m1,m2,m3,m4,m5,m6=projection(triangle[0][0],triangle[0][1],triangle[0][2],triangle[0][3],triangle[0][4],triangle[0][5])
    m4,m5,m6=projection3(triangle[0][0],triangle[0][1],triangle[0][2],triangle[0][3],triangle[0][4],triangle[0][5])
    p1=np.array([m4,m5,m6])
    #m1,m2,m3,m4,m5,m6=projection(triangle[1][0],triangle[1][1],triangle[1][2],triangle[1][3],triangle[1][4],triangle[1][5])
    m4,m5,m6=projection3(triangle[1][0],triangle[1][1],triangle[1][2],triangle[1][3],triangle[1][4],triangle[1][5])
    p2=np.array([m4,m5,m6])
    #m1,m2,m3,m4,m5,m6=projection(triangle[2][0],triangle[2][1],triangle[2][2],triangle[2][3],triangle[2][4],triangle[2][5])
    m4,m5,m6=projection3(triangle[2][0],triangle[2][1],triangle[2][2],triangle[2][3],triangle[2][4],triangle[2][5])
    p3=np.array([m4,m5,m6])
    volume0=triangle_area(p1,p2,p3)
    volume1=triangle_area(p0,p2,p3)+triangle_area(p1,p0,p3)+triangle_area(p1,p2,p0)
    if abs(volume0-volume1)< TOL:
        return 0
    else:
        return 1

cdef int make_tmp_pod(np.ndarray[DTYPE_int_t, ndim=4] pod, int number):
    #generator_xyz_dim4_tmp(pod,number)
    generator_xyz_dim4_tetrahedron(pod,'./junk%d'%(number),0)
    return 0

cdef np.ndarray read_tmp_pod(int number):
    cdef int a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6
    cdef int i,num
    cdef np.ndarray[DTYPE_int_t, ndim=1] tmp1
    
    #print('%d'%(number))
    #file_name='./tmp%d.xyz'%(number)
    f1=read_file('./junk%d.xyz'%(number))
    f0=f1[0].split()
    num=int(f0[0])
    for i in range(2,num+2):
        fi=f1[i]
        fi=fi.split()
        a1=int(fi[10])
        b1=int(fi[11])
        c1=int(fi[12])
        a2=int(fi[13])
        b2=int(fi[14])
        c2=int(fi[15])
        a3=int(fi[16])
        b3=int(fi[17])
        c3=int(fi[18])
        a4=int(fi[19])
        b4=int(fi[20])
        c4=int(fi[21])
        a5=int(fi[22])
        b5=int(fi[23])
        c5=int(fi[24])
        a6=int(fi[25])
        b6=int(fi[26])
        c6=int(fi[27])
        if i==2:
            tmp1=np.array([a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6])
        else:
            tmp1=np.append(tmp1,[a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6])
    return tmp1.reshape(int(len(tmp1)/72),4,6,3)

cdef list read_file(str fname):
    cdef list line
    try:
        f=open(fname,'r')
    except IOError, e:
        print(e)
        sys.exit(0)
    line=[]
    while 1:
        a=f.readline()
        if not a:
            break
        line.append(a[:-1])
    return line

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    ###################################      object_subtraction     ######################################
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################


cpdef np.ndarray object_subtraction_2(np.ndarray[DTYPE_int_t, ndim=4] obj1,\
                                        np.ndarray[DTYPE_int_t, ndim=4] obj2,\
                                        int verbose):
    # get A and not B = A and not (A and B)
    # obj1: A
    # obj2: A and B
    # obj3: B
    cdef int i1,i2,i3,i4,i5,flag1,counter1,counter2
    cdef long v01,v02,v03
    cdef float vol
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2c
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    cdef np.ndarray[DTYPE_int_t,ndim=4] obj2_surface
    
    if verbose>0:
        print('    object_subtraction_2()')
    #if verbose==1:
    #    print '  get A not B = A not (A and B)'
    #    print '  obj1: A'
    #    print '  obj2: A and B'
    #else:
    #    pass

    #
    tmp1a=np.array([0])
    if obj2.tolist()!=[[[[0]]]]:
    
        # get surfaces of onj2
        #obj2_surface=generator_surface(obj2,verbose-1)
        obj2_surface=generator_surface_1(obj2,verbose-1)
        #
        tmp3a=obj2_surface.reshape(len(obj2_surface)*3,6,3)
        tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
    
        for i1 in range(len(obj1)):
            tmp1b=np.array([0])
            for i2 in range(len(tmp3a)):
                flag1=inside_outside_tetrahedron(tmp3a[i2],obj1[i1][0],obj1[i1][1],obj1[i1][2],obj1[i1][3])
                if flag1==0: # inside
                    if len(tmp1b)==1:
                        tmp1b=tmp3a[i2].reshape(18)
                    else:
                        tmp1b=np.append(tmp1b,tmp3a[i2])
                else:
                    pass
            if len(tmp1b)!=1:
                # Tetrahedralization
                tmp4a=tetrahedralization(obj1[i1],tmp1b.reshape(int(len(tmp1b)/18),6,3),verbose-1)
                if tmp4a.tolist!=[[[[0]]]]:
                    for i3 in range(len(tmp4a)):
                        #"""
                        ### Algorithm 1 ###
                        # geometric center, centroid of the tetrahedron, tmp2c
                        tmp2c=centroid(tmp4a[i3])
                        counter2=0
                        for i5 in range(len(obj2)):
                            # check tmp2c is out of obj2 or not
                            flag1=inside_outside_tetrahedron(tmp2c,obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
                            if flag1==0: # inside
                                counter2+=1
                                break
                            else:
                                pass
                        if counter2==0:
                            if len(tmp1a)==1:
                                tmp1a=tmp4a[i3].reshape(72)
                            else:
                                tmp1a=np.append(tmp1a,tmp4a[i3])
                        else:
                            pass
                        #"""
                        """
                        ### Algorithm 2 ###
                        counter1=0
                        for i4 in range(4):
                            for i5 in range(len(obj2)):
                                flag1=inside_outside_tetrahedron(tmp4a[i3][i4],obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
                                if flag1==0: # inside
                                    counter1+=1
                                    break
                                else:
                                    pass
                        if counter1!=4:
                            if len(tmp1a)==1:
                                tmp1a=tmp4a[i3].reshape(72)
                            else:
                                tmp1a=np.append(tmp1a,tmp4a[i3])
                        else:
                            # geometric center, centroid of the tetrahedron, tmp2c
                            tmp2c=centroid(tmp4a[i3])
                            counter2=0
                            for i5 in range(len(obj2)):
                                # check tmp2c is out of obj2 or not
                                flag1=inside_outside_tetrahedron(tmp2c,obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
                                if flag1==0: # inside
                                    counter2+=1
                                    break
                                else:
                                    pass
                            if counter2==0:
                                if len(tmp1a)==1:
                                    tmp1a=tmp4a[i3].reshape(72)
                                else:
                                    tmp1a=np.append(tmp1a,tmp4a[i3])
                            else:
                                pass
                        """    
                else:
                    pass
            else:
                if len(tmp1a)==1:
                    tmp1a=obj1[i1].reshape(72)
                else:
                    tmp1a=np.append(tmp1a,obj1[i1])
        if len(tmp1a)!=1:
            tmp4a=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
            if verbose>0:
                v01,v02,v03=obj_volume_6d(tmp4a)
                vol=(v01+v02*TAU)/v03
                print('      obj1 NOT obj2, volume = %d %d %d (%10.6f)'%(v01,v02,v03,vol))
            else:
                pass
            return tmp4a
        else:
            #return tmp1a.reshape(1,1,1,1)
            return np.array([[[[0]]]])
    else:
        return obj1

cpdef np.ndarray object_subtraction(np.ndarray[DTYPE_int_t, ndim=4] obj1,
                                    np.ndarray[DTYPE_int_t, ndim=4] obj2,
                                    int verbose):
    # This get an object = obj1 and not obj2 = obj1 and not (obj1 and obj2)
    cdef int i1,i2,i3,num
    cdef int counter1,counter2,counter3,counter4
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c,tmp1d
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,tmp4b,tmp4c,tmp4d
    cdef list indx1
    counter2=0
    counter3=0
    counter4=0
    
    if verbose>0:
        print('       Object_subtraction()')
    else:
        pass
    for i1 in range(len(obj1)):
        counter1=0
        indx1=[]
        tmp3a=obj1[i1]
        #####
        for i2 in range(len(obj2)):
            tmp3b=obj2[i2]
            #print 'hallo1'
            tmp4a,tmp4b=intersection_two_tetrahedron_4(tmp3a,tmp3b,verbose-1) # two objects after tetrahedralization
            if tmp4a.tolist()!=[[[[0]]]]:
                if counter3==0:
                    tmp1a=tmp3a.reshape(72) # 4*6*3=72
                else:
                    tmp1a=np.append(tmp1a,tmp3a)
                counter1+=1
                indx1.append(i2)
            else:
                counter1+=0
        ######
        # tmp1c: set of tetrahedra in obj1 that are outseide of obj2
        #print 'hallo2'
        if counter1==0: # i1-th tetrahedron of obj1 is outside of obj2
            if counter2==0:
                tmp1c=tmp3a.reshape(72) # 72=4*6*3
            else:
                tmp1c=np.append(tmp1c,tmp3a)
            counter2+=1
        else:
            counter4+=1
        #####
        # if i1-th tetrahedron in obj1 (tmp3a) intersects with obj2
        #print 'hallo3'
        if counter1!=0:
            #####
            # generate set of tetrahedra in obj2 that intersect with tmp4b
            # -->> tmp4d:
            #print 'hallo4'
            for i3 in range(len(indx1)):
                if i3==0:
                    num=indx1[0]
                    tmp1d=obj2[num]
                else:
                    num=indx1[i3]
                    tmp1d=np.append(tmp1d,obj2[num])
            tmp4d=tmp1d.reshape(int(len(tmp1d)/72),4,6,3)
            #
            #
            # intersection_tetrahedron_object(tetrahedron,object)
            # intersection between a tetrahedron and an object (set of tetrahedra)
            #
            # tetrahedron = tmp3a
            # object = tmp4d
            #
            #print 'hallo5'
            tmp4c=intersection_tetrahedron_obj_4(tmp3a,tmp4d,0,verbose-1)
            if counter3==0:
                tmp1b=tmp4c.reshape(int(len(tmp4c)/72),4,6,3)
            else:
                tmp1b=np.append(tmp1b,tmp4c)
            counter3+=1
            # merge two sets (tmp1c,tmp1b) of tetrahedra that are outseide of obj2
            tmp1c=np.append(tmp1c,tmp1b)
        else: #if i1 tetrahedron in obj1 (tmp3a) does NOT intersect with obj2
            pass
    #print 'hallo6'    
    if counter4+counter3!=0:
        return tmp1c.reshape(int(len(tmp1c)/72),4,6,3)
    else: # if (obj1 and not obj2) is empty
        return np.array([[[[0]]]])

# NEW
cpdef np.ndarray object_subtraction_new(np.ndarray[DTYPE_int_t, ndim=4] obj1,\
                                        np.ndarray[DTYPE_int_t, ndim=4] obj2,\
                                        np.ndarray[DTYPE_int_t, ndim=4] obj3,\
                                        int verbose):
    # get A and not B = A and not (A and B)
    # obj1: A
    # obj2: A and B
    # obj3: B
    cdef int i1,i2,counter0,counter1,counter2,counter5,flag,flag1,flag2,num
    cdef long w1,w2,w3,v01,v02,v03
    cdef float vol
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,v1,v2,v3,v4,v5,v6
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a,tmp2b
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b,tmp3c,tetrahedron
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,obj2_surface,obj3_surface
        
    counter0=0
    counter1=0
    if verbose>0:
        print('       object_subtraction_new()')
    else:
        pass
    #if verbose==1:
    #    print ' Subtraction by object_subtraction_new(obj1, obj2, obj3)'
    #    print '  get A not B = A not (A and B)'
    #    print '  obj1: A'
    #    print '  obj2: A and B'
    #    print '  obj3: B'
    #else:
    #    pass
    #
    if verbose>0:
        print('          (1) A and B')
    else:
        pass
    #obj2_surface=generator_surface(obj2verbose-1)
    obj2_surface=generator_surface_1(obj2,verbose-1)
    
    if verbose>0:
        print('          (2) B')
    else:
        pass
    #obj3_surface=generator_surface(obj3)
    #obj3_surface=generator_surface(obj3,verbose-1)
    obj3_surface=generator_surface_1(obj3,verbose-1)
    
    """
    #tmp3b=remove_doubling_dim4(obj2_surface)
    tmp3b=remove_doubling_dim4(obj3_surface)
    for i1 in range(len(tmp3b)):
        v1,v2,v3,v4,v5,v6=projection(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
        print 'Zz %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
    """
    if verbose>0:
        print('   (3) A not B')
    else:
        pass
    if obj2.tolist()!=[[[[0]]]]:
        ##########
        # get vertices of obj2_surface which are on obj3_surface (points B)
        #    see subtraction_tetrahedron_object()
        tmp3b=obj2_surface.reshape(len(obj2_surface)*3,6,3)
        #tmp3b=remove_doubling_dim3_in_perp_space(tmp3b)
        counter0=0
        for i1 in range(len(tmp3b)):
            tmp2b=tmp3b[i1]
            for i2 in range(len(obj3_surface)):
                tmp3c=obj3_surface[i2]
                flag=on_out_surface(tmp2b,tmp3c)
                if flag==0: # on surface
                    if counter0==0:
                        tmp1b=tmp2b.reshape(18)
                    else:
                        tmp1b=np.append(tmp1b,tmp2b)
                    counter0+=1
                else: # out
                    pass
        if counter0!=0:
            tmp3b=tmp1b.reshape(int(len(tmp1b)/18),6,3)
        else:
            pass
        ########### 
        #   TEMP  #
        ###########
        tmp3b=remove_doubling_dim3_in_perp_space(tmp3b)
        #print '       number of points B = %d'%(len(tmp3b))
        #for i1 in range(len(tmp3b)):
        #    v1,v2,v3,v4,v5,v6=projection(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
        #    print 'Yy %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
        ###########
        #         #
        ###########
        counter1=0
        for i1 in range(len(obj1)):
            #print '       %d-th tetrahedron in A'%(i1)
            tetrahedron=obj1[i1]
            ############
            #  subtraction_tetrahedron_object()
            #  This returns vertecies which correspond to "A and not B = A and not (A and B)".
            ############
            tmp3a=subtraction_tetrahedron_object(tetrahedron,obj2,tmp3b,obj2_surface,obj3_surface,verbose-1)
            if tmp3a.tolist()!=[[[0]]]:
                if counter1==0:
                    tmp1a=tmp3a.reshape(len(tmp3a)*18) # 18=6*3
                else:
                    tmp1a=np.append(tmp1a,tmp3a)
                counter1+=1
            else:
                pass
            ############
            """
            ############
            #  subtraction_tetrahedron_object_dev()
            #  under construction
            #  This returns set of tetrahedra for "A and not B = A and not (A and B)".
            ############
            tmp4a=subtraction_tetrahedron_object_dev(tetrahedron,obj2,tmp3b,obj2_surface,obj3_surface,path)
            if tmp4a.tolist()!=[[[[0]]]]:
                if counter1==0:
                    tmp1a=tmp4a.reshape(len(tmp4a)*72) # 72=4*6*3
                else:
                    tmp1a=np.append(tmp1a,tmp4a)
                counter1+=1
            else:
                pass
            ############
            """
        if counter1!=0:
            #return tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
            tmp3a=tmp1a.reshape(int(len(tmp1a)/18),6,3)
            tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
            #for i1 in range(len(tmp3a)):
            #    v1,v2,v3,v4,v5,v6=projection(tmp3a[i1][0],tmp3a[i1][1],tmp3a[i1][2],tmp3a[i1][3],tmp3a[i1][4],tmp3a[i1][5])
            #    print 'Yy %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
            # return tmp4a
            ######################
            #                    #
            # Tetrahedralization #
            #                    #
            ######################
            counter3=0
            tmp4a=tetrahedralization_points(tmp3a,verbose-1)
            if tmp4a.tolist()!=[[[[0]]]]:
                for i1 in range(len(tmp4a)):
                    tmp3a=tmp4a[i1]
                    counter2=0
                    #print 'tmp3a = ',tmp3a
                    for i2 in range(4):
                        tmp2a=tmp3a[i2]
                        counter1=0
                        for i3 in range(len(obj2)):
                            tmp3b=obj2[i3]
                            flag=inside_outside_tetrahedron(tmp2a,tmp3b[0],tmp3b[1],tmp3b[2],tmp3b[3])
                            if flag==0: # inside
                                counter1+=1
                                break
                            else: # outside 
                                pass
                        if counter1==0:
                            counter2+=1
                        else:
                            pass
                    if counter2<4: # outside
                        if counter3==0:
                            tmp1a=tmp3a.reshape(72) # 72=4*6*3
                        else:
                            tmp1a=np.append(tmp1a,tmp3a)
                        counter3+=1
                    else:
                        pass
                tmp4a=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
                w1,w2,w3=obj_volume_6d(tmp4a)
                vol=(w1+w2*TAU)/w3
                if verbose>0:
                    print('          A not B obj, volume = %d %d %d (%10.6f)'%(w1,w2,w3,vol))
                else:
                    pass
                """
                #
                # check each tetrahedron in "tmp4a" is really inside "obj1" and out of "obj3"
                print '   checking common obj'
                counter5=0
                num=len(tmp4a)
                for i1 in range(len(tmp4a)):
                    flag1=tetrahedron_inside_obj(tmp4a[i1],obj1)
                    flag2=tetrahedron_inside_obj(tmp4a[i1],obj3)
                    if flag1==0 and flag2==1:
                        if counter5==0:
                            tmp1a=tmp4a[i1].reshape(72)
                        else:
                            tmp1a=np.append(tmp1a,tmp4a[i1])
                        counter5+=1
                    else:
                        pass
                tmp4a=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
                print '   numbre of tetrahedron: %d -> %d'%(num,counter5)
                v01,v02,v03=obj_volume_6d(tmp4a)
                vol=(v01+v02*TAU)/float(v03)
                print '   A not B obj, volume = %d %d %d (%10.6f)'%(v01,v02,v03,vol)
                """
                return tmp4a
            else:
                return np.array([[[[0]]]])
        else:
            #return np.array([0],dtype=np.int).reshape(1,1,1,1)
            return np.array([[[[0]]]])
    else:
        return obj1    

# working
cpdef np.ndarray object_subtraction_dev1(np.ndarray[DTYPE_int_t, ndim=4] obj1,\
                                        np.ndarray[DTYPE_int_t, ndim=4] obj2,\
                                        np.ndarray[DTYPE_int_t, ndim=4] obj3,\
                                        int verbose):
    # get A and not B = A and not (A and B)
    # obj1: A
    # obj2: A and B
    cdef int flag,flag1,counter1,counter2,counter3
    cdef int i1,i2,i3
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    cdef np.ndarray[DTYPE_int_t,ndim=3] vertx_obj2
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,tmp4b,tmp4c
    cdef np.ndarray[DTYPE_int_t,ndim=4] surface_obj3
    
    print(' object_subtraction_dev1()')
    
    #if verbose==1:
    #    print '  get A not B = A not (A and B)'
    #    print '  obj1: A'
    #    print '  obj2: A and B'
    #    print '  obj3: B'
    #else:
    #    pass

    tmp1b=np.array([0])
    #tmp4b=generator_surface(obj2,verbose-1)
    #tmp4b=generator_surface(obj2,verbose-1)
    tmp4b=generator_surface_1(obj2,verbose-1)
    
    vertx_obj2=remove_doubling_dim3_in_perp_space(tmp4b.reshape(len(tmp4b)*3,6,3)) # vertices of obj2
    #surface_obj3=generator_surface(obj3,verbose-1) # surface of obj3
    #surface_obj3=generator_surface(obj3,verbose-1)# surface of obj3
    surface_obj3=generator_surface_1(obj3,verbose-1)# surface of obj3
    
    # get vertices of obj2 which on the surface of obj3
    tmp1a=np.array([0])
    counter2=0
    for i2 in range(len(vertx_obj2)):
        counter1=0
        for i3 in range(len(surface_obj3)):
            flag=on_out_surface(vertx_obj2[i2],surface_obj3[i3])
            if flag==0: # on
                counter1+=1
                break
            else:
                pass
        if counter1!=0:
            if counter2==0:
                tmp1a=vertx_obj2[i2].reshape(18)
            else:
                tmp1a=np.append(tmp1a,vertx_obj2[i2])
            counter2+=1
        else:
            pass

    vertx_obj2=remove_doubling_dim3_in_perp_space(tmp1a.reshape(int(len(tmp1a)/18),6,3)) # vertices of obj2
    #print 'len(vertx_obj2)=',len(vertx_obj2)
    
    counter3=0
    for i1 in range(len(obj1)):

        counter2=0

        # get vertices of i1-th tetrahedron of obj1 which are NOT inside obj3
        for i2 in range(4):
            counter1=0
            for i3 in range(len(obj3)):
                flag=inside_outside_tetrahedron(obj1[i1][i2],obj3[i3][0],obj3[i3][1],obj3[i3][2],obj3[i3][3])
                if flag==0: # insede
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                if counter2==0:
                    tmp1a=obj1[i1][i2].reshape(18)
                else:
                    tmp1a=np.append(tmp1a,obj1[i1][i2])
                counter2+=1
            else:
                pass
        
        # get vertx_obj2 (vertices of obj2 on the surface of obj3) which are inside i1-th tetrahedron in obj1
        for i2 in range(len(vertx_obj2)):
            flag=inside_outside_tetrahedron(vertx_obj2[i2],obj1[i1][0],obj1[i1][1],obj1[i1][2],obj1[i1][3])
            if flag==0: # insede
                if counter2==0:
                    tmp1a=vertx_obj2[i2].reshape(18)
                else:
                    tmp1a=np.append(tmp1a,vertx_obj2[i2])
                counter2+=1
            else:
                pass
        
        if counter2!=0:
            tmp3b=remove_doubling_dim3_in_perp_space(tmp1a.reshape(int(len(tmp1a)/18),6,3))
            tmp1a=np.array([0])
            if len(tmp3b)>=4:
                if coplanar_check(tmp3b)==0:
                    tmp4a=tetrahedralization_points(tmp3b,verbose-1)
                    counter2=0
                    for i2 in range(len(tmp4a)):
                        # geometric center, centroid of the tetrahedron, tmp2c
                        tmp2a=centroid(tmp4a[i2])
                        counter1=0
                        for i3 in range(len(obj2)):
                            # check tmp2c is out of obj2 or not
                            flag=inside_outside_tetrahedron(tmp2a,obj2[i3][0],obj2[i3][1],obj2[i3][2],obj2[i3][3])
                            if flag==0: # inside
                                counter1+=1
                                break
                            else:
                                pass
                        if counter1==0:
                            if counter2==0:
                                tmp1a=tmp4a[i2].reshape(72)
                            else:
                                tmp1a=np.append(tmp1a,tmp4a[i2])
                            counter2+=1
                        else:
                            pass
                    if counter2!=0:
                        if counter3==0:
                            tmp1b=tmp1a
                        else:
                            tmp1b=np.append(tmp1b,tmp1a)
                        counter3+=1
                    else:
                        pass
                else:
                    pass
            else:
                pass
        else:
            pass
    if counter3!=0:
        return tmp1b.reshape(int(len(tmp1b)/72),4,6,3)
    else:
        return np.array([[[[0]]]])

# working
cpdef np.ndarray object_subtraction_dev(np.ndarray[DTYPE_int_t, ndim=4] obj1,\
                                        np.ndarray[DTYPE_int_t, ndim=4] obj2,\
                                        np.ndarray[DTYPE_int_t, ndim=4] obj3,\
                                        int verbose):
    # get A and not B = A and not (A and B)
    # obj1: A
    # obj2: A and B
    cdef int flag,flag1,counter1,counter2,counter3
    cdef int i1,i2,i3
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,tmp4b,tmp4c
    
    print(' object_subtraction_dev()')
    
    #if verbose==1:
    #    print '  get A not B = A not (A and B)'
    #    print '  obj1: A'
    #    print '  obj2: A and B'
    #else:
    #    pass
    tmp1a=np.array([0])
    tmp1b=np.array([0])
    #tmp4b=generator_surface(obj2,verbose-1)
    tmp4b=generator_surface_1(obj2,verbose-1)
    tmp3a=remove_doubling_dim3_in_perp_space(tmp4b.reshape(len(tmp4b)*3,6,3)) # vertices of obj2
    for i1 in range(len(obj1)):
        # get vertices of obj2 which are inside i1-th tetrahedron of obj1
        for i2 in range(len(tmp3a)):
            flag=inside_outside_tetrahedron(tmp3a[i2],obj1[i1][0],obj1[i1][1],obj1[i1][2],obj1[i1][3])
            if flag==0: # insede
                if len(tmp1a)==1:
                    tmp1a=tmp3a[i2].reshape(18)
                else:
                    tmp1a=np.append(tmp1a,tmp3a[i2])
            else:
                pass
        # get vertices of i1-th tetrahedron of obj1 which are NOT inside obj2
        for i2 in range(4):
            counter1=0
            for i3 in range(len(obj2)):
                flag=inside_outside_tetrahedron(obj1[i1][i2],obj2[i2][0],obj2[i2][1],obj2[i2][2],obj2[i2][3])
                if flag==0: # insede
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                if len(tmp1a)==1:
                        tmp1a=obj1[i1][i2].reshape(18)
                else:
                    tmp1a=np.append(tmp1a,obj1[i1][i2])
            else:
                pass
        if tmp1a.tolist()!=[0]:
            tmp3b=remove_doubling_dim3_in_perp_space(tmp1a.reshape(int(len(tmp1a)/18),6,3))
            tmp1a=np.array([0])
            if len(tmp3b)>=4:
                tmp4a=tetrahedralization_points(tmp3b,verbose-1)
                for i2 in range(len(tmp4a)):
                    # geometric center, centroid of the tetrahedron, tmp2c
                    tmp2a=centroid(tmp4a[i2])
                    counter1=0
                    for i3 in range(len(obj2)):
                        # check tmp2c is out of obj2 or not
                        flag=inside_outside_tetrahedron(tmp2a,obj2[i3][0],obj2[i3][1],obj2[i3][2],obj2[i3][3])
                        if flag==0: # inside
                            counter1+=1
                            break
                        else:
                            pass
                    if counter1==0:
                        if len(tmp1a)==1:
                            tmp1a=tmp4a[i2].reshape(72)
                        else:
                            tmp1a=np.append(tmp1a,tmp4a[i2])
                    else:
                        pass
                if len(tmp1b)==1:
                    tmp1b=tmp1a
                else:
                    tmp1b=np.append(tmp1b,tmp1a)
            else:
                pass
        else:
            pass
    if len(tmp1b)!=1:
        tmp4a=tmp1b.reshape(int(len(tmp1b)/72),4,6,3)
        # もう一度、obj2との交点を求め、四面体分割。そして、obj2に含まれない部分を取り出す。
        # ただし、複雑に凹凸部分と重なる部分では、obj2に含まれる部分を取り除くことができないので注意
        counter1=0
        for i1 in range(len(obj1)):
            for i2 in range(len(tmp4a)):
                tmp4c=intersection_two_tetrahedron_4(obj1[i1],tmp4a[i2],verbose-1)
                if tmp4c.tolist()!=[[[[0]]]]:
                    if counter1==0:
                        tmp1c=tmp4c.reshape(len(tmp4c)*72)
                    else:
                        tmp1c=np.append(tmp1c,tmp4c)
                    counter1+=1
                else:
                    pass
        tmp1c=np.append(tmp4a,tmp1c)
        tmp3a=remove_doubling_dim3_in_perp_space(tmp1c.reshape(int(len(tmp1c)/18),6,3))
        tmp4a=tetrahedralization_points(tmp3a,verbose-1)
        tmp1a=np.array([0])
        counter1=0
        for i2 in range(len(tmp4a)):
            counter2=0
            for i3 in range(4) :
                for i4 in range(len(obj2)):
                    flag=inside_outside_tetrahedron(tmp4a[i2][i3],obj2[i4][0],obj2[i4][1],obj2[i4][2],obj2[i4][3])
                    if flag==0: # inside
                        counter2+=1
                        break
                    else:
                        pass
            if counter2==4:
                # geometric center, centroid of the tetrahedron, tmp2c
                tmp2a=centroid(tmp4a[i2])
                for i5 in range(len(obj2)):
                    flag1=inside_outside_tetrahedron(tmp2a,obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
                    if flag1==0: # inside
                        counter3+=1
                        break
                    else:
                        pass
                if counter3==0:
                    if counter1==0:
                        tmp1a=tmp4a[i2].reshape(72)
                    else:
                        tmp1a=np.append(tmp1a,tmp4a[i2])
                    counter1+=1
                else:
                    pass
            else:
                if counter1==0:
                    tmp1a=tmp4a[i2].reshape(72)
                else:
                    tmp1a=np.append(tmp1a,tmp4a[i2])
                counter1+=1
        # obj2の凹凸部分との共通部分を除くために、分割した四面体ごとに、再度、obj3との交点を求める。
        # 交点があるならば、その点を含めて四面体分割する。
        # そして、各四面体の重心を用いて、obj1 not obj 2 かを判断
        if counter1!=0:
            tmp1b=np.array([0])
            tmp4a=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
            for i3 in range(len(tmp4a)):
                tmp1c=np.array([0])
                for i4 in range(len(obj3)):
                    tmp4c=intersection_two_tetrahedron_4(tmp4a[i3],obj3[i4],verbose-1)
                    if tmp4c.tolist()!=[[[[0]]]]:
                        if len(tmp1c)==1:
                            tmp1c=tmp4c.reshape(len(tmp4c)*72)
                        else:
                            tmp1c=np.append(tmp1c,tmp4c)
                    else:
                        pass
                if len(tmp1c)!=1:
                    tmp3a=remove_doubling_dim3_in_perp_space(tmp1c.reshape(int(len(tmp1c)/18),6,3))
                    tmp4b=tetrahedralization(tmp4a[i3],tmp3a,verbose-1)
                    for i4 in range(len(tmp4b)):
                        tmp2a=centroid(tmp4b[i4])
                        counter3=0
                        for i5 in range(len(obj2)):
                            flag1=inside_outside_tetrahedron(tmp2a,obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
                            if flag1==0: # inside
                                counter3+=1
                                break
                            else:
                                pass
                        if counter3==0:
                            if len(tmp1b)==1:
                                tmp1b=tmp4b[i4].reshape(72)
                            else:
                                tmp1b=np.append(tmp1b,tmp4b[i4])
                        else:
                            pass
                else:
                    pass
            if len(tmp1b)!=1:
                return tmp1b.reshape(int(len(tmp1b)/72),4,6,3)
            else:
                return np.array([[[[0]]]])
        else:
            return np.array([[[[0]]]])
    else:
        return np.array([[[[0]]]])

# working
cpdef np.ndarray object_subtraction_dev2(np.ndarray[DTYPE_int_t, ndim=4] obj1,\
                                        np.ndarray[DTYPE_int_t, ndim=4] obj2,\
                                        np.ndarray[DTYPE_int_t, ndim=4] obj3,\
                                        int verbose):
    # get A and not B = A and not (A and B)
    # obj1: A
    # obj2: A and B
    cdef int flag,flag1,counter1,counter2,counter3
    cdef int i1,i2,i3
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    cdef np.ndarray[DTYPE_int_t,ndim=3] vertx_obj2
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,tmp4b,tmp4c
    cdef np.ndarray[DTYPE_int_t,ndim=4] surface_obj3
    
    print(' object_subtraction_dev2()')
    
    #if verbose==1:
    #    print '  get A not B = A not (A and B)'
    #    print '  obj1: A'
    #    print '  obj2: A and B'
    #    print '  obj3: B'
    #else:
    #    pass

    tmp1b=np.array([0])
    #tmp4b=generator_surface(obj2)
    tmp4b=generator_surface_1(obj2,verbose-1)
    vertx_obj2=remove_doubling_dim3_in_perp_space(tmp4b.reshape(len(tmp4b)*3,6,3)) # vertices of obj2
    #surface_obj3=generator_surface(obj3) # surface of obj3
    surface_obj3=generator_surface_1(obj3,verbose-1)
    
    # get vertices of obj2 which on the surface of obj3
    tmp1a=np.array([0])
    counter2=0
    for i2 in range(len(vertx_obj2)):
        counter1=0
        for i3 in range(len(surface_obj3)):
            flag=on_out_surface(vertx_obj2[i2],surface_obj3[i3])
            if flag==0: # on
                counter1+=1
                break
            else:
                pass
        if counter1!=0:
            if counter2==0:
                tmp1a=vertx_obj2[i2].reshape(18)
            else:
                tmp1a=np.append(tmp1a,vertx_obj2[i2])
            counter2+=1
        else:
            pass

    vertx_obj2=remove_doubling_dim3_in_perp_space(tmp1a.reshape(int(len(tmp1a)/18),6,3)) # vertices of obj2
    #print 'len(vertx_obj2)=',len(vertx_obj2)
    
    # get vertices of obj1 which are NOT inside obj3
    counter2=0
    for i1 in range(len(obj1)):
        for i2 in range(4):
            counter1=0
            for i3 in range(len(obj3)):
                flag=inside_outside_tetrahedron(obj1[i1][i2],obj3[i3][0],obj3[i3][1],obj3[i3][2],obj3[i3][3])
                if flag==0: # insede
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                if counter2==0:
                    tmp1a=obj1[i1][i2].reshape(18)
                else:
                    tmp1a=np.append(tmp1a,obj1[i1][i2])
                counter2+=1
            else:
                pass
    
    # get vertices of obj2 on the surface of obj3 which are inside obj1
    for i1 in range(len(obj1)):
        for i2 in range(len(vertx_obj2)):
            flag=inside_outside_tetrahedron(vertx_obj2[i2],obj1[i1][0],obj1[i1][1],obj1[i1][2],obj1[i1][3])
            if flag==0: # insede
                if counter2==0:
                    tmp1a=vertx_obj2[i2].reshape(18)
                else:
                    tmp1a=np.append(tmp1a,vertx_obj2[i2])
                counter2+=1
            else:
                pass
    
    if len(tmp1a)!=1:
        counter3=0
        tmp3b=remove_doubling_dim3_in_perp_space(tmp1a.reshape(int(len(tmp1a)/18),6,3))
        tmp1a=np.array([0])
        if len(tmp3b)>=4:
            if coplanar_check(tmp3b)==0:
                tmp4a=tetrahedralization_points(tmp3b,verbose-1)
                counter2=0
                for i2 in range(len(tmp4a)):
                    # geometric center, centroid of the tetrahedron, tmp2c
                    tmp2a=centroid(tmp4a[i2])
                    counter1=0
                    for i3 in range(len(obj2)):
                        # check tmp2c is out of obj2 or not
                        flag=inside_outside_tetrahedron(tmp2a,obj2[i3][0],obj2[i3][1],obj2[i3][2],obj2[i3][3])
                        if flag==0: # inside
                            counter1+=1
                            break
                        else:
                            pass
                    if counter1==0:
                        if counter2==0:
                            tmp1a=tmp4a[i2].reshape(72)
                        else:
                            tmp1a=np.append(tmp1a,tmp4a[i2])
                        counter2+=1
                    else:
                        pass
                if counter2!=0:
                    if counter3==0:
                        tmp1b=tmp1a
                    else:
                        tmp1b=np.append(tmp1b,tmp1a)
                    counter3+=1
                else:
                    pass
            else:
                pass
        if counter3!=0:
            return tmp1b.reshape(int(len(tmp1b)/72),4,6,3)
        else:
            return np.array([[[[0]]]])
    else:
        return np.array([[[[0]]]])

cdef np.ndarray object_not_object(np.ndarray[DTYPE_int_t, ndim=4] obj1,
                                    np.ndarray[DTYPE_int_t, ndim=4] obj2,
                                    int verbose):
    # get A not B
    # obj1: A
    # obj2: B
    cdef int i1
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a

    if verbose>0:
        print('       object_not_object()')
    else:
        pass
        
    tmp1a=np.array([0])
    for i1 in range(len(obj1)):
        tmp4a=tetrahedron_not_obj(obj1[i1],obj2,verbose-1)
        if tmp4a.tolist()!=[[[[0]]]]:
            if len(tmp1a)==1:
                tmp1a=tmp4a.reshape(len(tmp4a)*72)
            else:
                tmp1a=np.append(tmp1a,tmp4a)
        else:
            pass
    if len(tmp1a)!=1:
        return tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
    else:
        return np.array([[[[0]]]])

cdef np.ndarray tetrahedron_not_obj(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron,
                                    np.ndarray[DTYPE_int_t, ndim=4] obj,
                                    int verbose):
    #
    # tetrahedron_not_obj()
    #
    # Parameters:
    # (1) tetrahedron
    # (2) obj
    #
    # Teturn:
    # tetrahedron NOT obj (a set of tetrahedron)
    #
    cdef int i1,counter1
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    
    tmp1a=np.array([0])
    tmp1b=np.array([0])
    
    if verbose>0:
        print('       tetrahedron_not_obj()')
    else:
        pass
        
    for i1 in range(len(obj)):
        #tmp3a=intersection_two_tetrahedron_0(tetrahedron,obj[i1])
        tmp3a=intersection_two_tetrahedron_4(tetrahedron,obj[i1],verbose-1)
        if tmp3a.tolist()!=[[[0]]]:
            if len(tmp1b)==1:
                tmp1b=tmp3a.reshape(len(tmp3a)*18)
            else:
                tmp1b=np.append(tmp1b,tmp3a)
        else:
            pass
    if len(tmp1b)!=1:
        tmp3a=remove_doubling_dim3_in_perp_space(tmp1b.reshape(int(len(tmp1b)/18),6,3))
        tmp4a=tetrahedralization(tetrahedron,tmp3a,verbose-1)
        for i1 in range(len(tmp4a)):
            tmp2a=centroid(tmp4a[i1])
            counter1=0
            for i2 in range(len(obj)):
                flag1=inside_outside_tetrahedron(tmp2a,obj[i2][0],obj[i2][1],obj[i2][2],obj[i2][3])
                if flag1==0: # inside
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                if len(tmp1a)==1:
                    tmp1a=tmp4a[i1].reshape(72)
                else:
                    tmp1a=np.append(tmp1a,tmp4a[i1])
            else:
                pass
        if len(tmp1a)!=1:
            return tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
        else:
            return np.array([[[[0]]]])
    else:
        return np.array([[[[0]]]])

cpdef np.ndarray object_subtraction_3(np.ndarray[DTYPE_int_t, ndim=4] obj1,
                                                np.ndarray[DTYPE_int_t, ndim=4] obj2,
                                                int verbose):
    # get A and not B = A and not (A and B)
    # obj1: A
    # obj2: A and B
    # obj3: B
    cdef int i1,i2,i3,i4,i5,counter0,counter1,counter2,counter3,counter4,counter5,counter6,counter7,flag1
    cdef long v01,v02,v03
    cdef float vol
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c,tmp1d
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    cdef np.ndarray[DTYPE_int_t,ndim=4] obj2_surface
    
    if verbose>0:
        print('        object_subtraction_3()')
    #print '  get A not B = A not (A and B)'
    #print '  obj1: A'
    #print '  obj2: A and B'

    tmp1a=np.array([0])
    if obj2.tolist()!=[[[[0]]]]:
    
        # get surfaces of obj2
        #obj2_surface=generator_surface(obj2,verbose-1)
        obj2_surface=generator_surface_1(obj2,verbose-1)
        
        tmp3a=obj2_surface.reshape(len(obj2_surface)*3,6,3)
        tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
        
        counter0=0
        counter3=0
        for i1 in range(len(obj1)):
            counter2=0
            for i2 in range(4):
                counter1=0
                for i3 in range(len(obj2)):
                    flag1=inside_outside_tetrahedron(obj1[i1][i2],obj2[i3][0],obj2[i3][1],obj2[i3][2],obj2[i3][3])
                    if flag1==0: # inside
                        counter1+=1
                        break
                    else:
                        pass
                if counter1!=0:
                    if counter2==0:
                        tmp1b=obj1[i1][i2].reshape(18)
                    else:
                        tmp1b=np.append(tmp1b,obj1[i1][i2])
                    counter2+=1
                else:
                    pass
            if counter2==4: # i1-th tetrahedron is inside obj2
                tmp2c=centroid(obj1[i1])
                counter7=0
                for i3 in range(len(obj2)):
                    flag1=inside_outside_tetrahedron(tmp2c,obj2[i3][0],obj2[i3][1],obj2[i3][2],obj2[i3][3])
                    if flag1==0:
                        counter7+=1
                    else:
                        pass
                if counter7==0:
                    if counter0==0:
                        tmp1a=obj1[i1].reshape(72)
                    else:
                        tmp1a=np.append(tmp1a,obj1[i1])
                    counter0+=1
            elif counter2==0: # i1-th tetrahedron is out of obj2
                if counter0==0:
                    tmp1a=obj1[i1].reshape(72)
                else:
                    tmp1a=np.append(tmp1a,obj1[i1])
                counter0+=1
            else: # i1-th tetrahedron is partially out of obj2
                # tmp1b: vertices of i1-th tetrahedron which is out of obj2
                # get vertices of obj2 which are inside i1-th tetrahedron, and merge them into tmp1b, then do tetrahedralization
                #
                counter4=0
                for i3 in range(len(tmp3a)):
                    flag1=inside_outside_tetrahedron(tmp3a[i3],obj1[i1][0],obj1[i1][1],obj1[i1][2],obj1[i1][3])
                    if flag1==0: # inside
                        if counter4==0:
                            tmp1d=tmp3a[i3].reshape(18)
                        else:
                            tmp1d=np.append(tmp1d,tmp3a[i3])
                        counter4+=1
                    else:
                        pass
                if counter4!=0:
                    tmp1b=np.append(tmp1b,tmp1d)
                    tmp3b=tmp1b.reshape(int(len(tmp1b)/18),6,3)
                    # Tetrahedralization
                    tmp4a=tetrahedralization_points(tmp3b,verbose-1)
                    if tmp4a.tolist()!=[[[[0]]]]:
                        # check each tetrahedron in tmp4a is not inside obj2
                        for i4 in range(len(tmp4a)):
                            tmp2c=centroid(tmp4a[i4])
                            counter7=0
                            for i5 in range(len(obj2)):
                                flag1=inside_outside_tetrahedron(tmp2c,obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
                                if flag1==0: # inside
                                    pass
                                else:
                                    if counter0==0:
                                        tmp1a=tmp4a[i4].reshape(72)
                                    else:
                                        tmp1a=np.append(tmp1a,tmp4a[i4])
                                    counter0+1
                            """
                            counter5=0
                            for i5 in range(4):
                                counter6=0
                                for i6 in range(len([obj2])):
                                    flag1=inside_outside_tetrahedron(tmp4a[i4][i5],obj2[i6][0],obj2[i6][1],obj2[i6][2],obj2[i6][3])
                                    if flag1==0: # inside
                                        counter6+=1
                                        break
                                    else:
                                        pass
                                if counter6==0:
                                    counter5+=1
                                else:
                                    pass
                            if counter5!=4:
                                if counter0==0:
                                    tmp1a=tmp4a[i4].reshape(72)
                                else:
                                    tmp1a=np.append(tmp1a,tmp4a[i4])
                                counter0+=1
                            else:
                                pass
                            """
                    else:
                        pass
                else:
                    pass
        if tmp1a.tolist()!=[0]:
            tmp4a=tmp1a.reshape(int(len(tmp1a)/72),4,6,3)
            v01,v02,v03=obj_volume_6d(tmp4a)
            vol=(v01+v02*TAU)/v03
            if verbose>0:
                print('          A not B obj, volume = %d %d %d (%10.6f)'%(v01,v02,v03,vol))
            return tmp4a
        else:
            return np.array([[[[0]]]])
    else:
        return obj1

# Developing
cdef np.ndarray subtraction_tetrahedron_object(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron,
                                                np.ndarray[DTYPE_int_t, ndim=4] obj2,
                                                np.ndarray[DTYPE_int_t, ndim=3] point_b,
                                                np.ndarray[DTYPE_int_t, ndim=4] obj2_surface,
                                                np.ndarray[DTYPE_int_t, ndim=4] obj3_surface,
                                                int verbose):
    # this subtractes a tetrahedron from an object (set of tetrahedra)
    # return (dim4)
    #
    cdef int flag
    cdef int i1,i2
    cdef int counter0,counter1,counter2,counter3,counter4
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c,v1,v2,v3,v4,v5,v6
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a,tmp2b
    #cdef np.ndarray[DTYPE_int_t,ndim=2] v1,v2,v3,v4,v5,v6
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b,tmp3c,obj_surface_vertex
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,obj_common

    if verbose>0:
        print('       subtraction_tetrahedron_object()')
    #print 'Subtraction_tetrahedron_object()'
    #print 'Intersection_two_tetrahedron_mod2()'
        
    #####################################
    #                                   #
    #                                   #
    #####################################
    # (1) get vertices of tetrahedron which are outside of obj2 (points A)
    counter1=0
    for i1 in range(4):
        tmp2a=tetrahedron[i1]
        counter0=0
        for i2 in range(len(obj2)):
            tmp3a=obj2[i2]
            flag=inside_outside_tetrahedron(tmp2a,tmp3a[0],tmp3a[1],tmp3a[2],tmp3a[3])
            if flag==0: # inside
                counter0+=1
                break
            else:
                pass
        if counter0==0:
            if counter1==0:
                tmp1a=tmp2a.reshape(18)
            else:
                tmp1a=np.append(tmp1a,tmp2a)
            counter1+=1
        else:
            pass
    if counter1==0: # tetrahedron is inside obj3
        #print 'inconsistency: common part is empty?'
        return np.array([[[0]]])
    else:
        #print '    points A:'
        tmp3a=tmp1a.reshape(int(len(tmp1a)/18),6,3)
        tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
        #for i1 in range(len(tmp3a)):
        #    v1,v2,v3,v4,v5,v6=projection(tmp3a[i1][0],tmp3a[i1][1],tmp3a[i1][2],tmp3a[i1][3],tmp3a[i1][4],tmp3a[i1][5])
        #    print 'Xx %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
        #pass
        #if counter0!=0:
        #    tmp1a=np.append(tmp1a,tmp1b)
        #else:
        #    pass
    #####################################
    #                                   #
    #                                   #
    #####################################    
    """  point_b
    # (2) get vertices of obj2_surface which are on obj3_surface (points B)
    tmp3b=obj2_surface.reshape(len(obj2_surface)/18,6,3)
    for i1 in range(len(tmp3b)):
        tmp2b=tmp3b[i1]
        for i2 in range(len(obj3_surface)):
            tmp3c=obj3_surface[i2]
            flag=on_out_surface(tmp2b,tmp3c)
            if flag==0: # on surface
                if counter0==0:
                    tmp1b=tmp2b.reshape(18)
                else:
                    tmp1b=np.append(tmp1b,tmp2b)
                counter0+=1
            else: # out
                pass
    print '    number of points B = %d'%(counter0)
    """
    #####################################
    #                                   #
    #                                   #
    #####################################
    # (3) get points in (points B) which are inside tetrahedron -- > (points C)
    #tmp1b=np.append(tmp1b,obj3_surface)
    tmp1b=np.append(point_b,obj3_surface)
    tmp3b=tmp1b.reshape(int(len(tmp1b)/18),6,3)
    tmp3b=remove_doubling_dim3_in_perp_space(tmp3b)
    counter2=0
    for i1 in range(len(tmp3b)):
        tmp2b=tmp3b[i1]
        flag=inside_outside_tetrahedron(tmp2b,tetrahedron[0],tetrahedron[1],tetrahedron[2],tetrahedron[3])
        if flag==0: # inside
            if counter2==0:
                tmp1b=tmp2b.reshape(18)
            else:
                tmp1b=np.append(tmp1b,tmp2b)
            counter2+=1
        else:
            pass
    tmp3b=tmp1b.reshape(int(len(tmp1b)/18),6,3)
    #tmp3b=remove_doubling_dim3_in_perp_space(tmp3b)
    #print '    points C:'
    #for i1 in range(len(tmp3b)):
    #    v1,v2,v3,v4,v5,v6=projection(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
    #    print 'Zz %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
    #####################################
    #                                   #
    #                                   #
    #####################################
    # (4) merge points A and points C
    if counter1==0 and counter2==0:
        return np.array([[[0]]])
    else:
        if counter1!=0 and counter2!=0:
            tmp1c=np.append(tmp1a,tmp1b)
            return tmp1c.reshape(int(len(tmp1c)/18),6,3)
        elif counter1!=0 and counter2==0:
            tmp1c=tmp1a
            return tmp1c.reshape(int(len(tmp1c)/18),6,3)
        elif counter1==0 and counter2!=0:
            tmp1c=tmp1b
            return tmp1c.reshape(int(len(tmp1c)/18),6,3)
        else:
            return np.array([[[0]]])

# Developing
cdef np.ndarray subtraction_tetrahedron_object_dev(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron,
                                                    np.ndarray[DTYPE_int_t, ndim=4] obj2,
                                                    np.ndarray[DTYPE_int_t, ndim=3] point_b,
                                                    np.ndarray[DTYPE_int_t, ndim=4] obj2_surface,
                                                    np.ndarray[DTYPE_int_t, ndim=4] obj3_surface,
                                                    int verbose):
    # this subtractes a tetrahedron from an object (set of tetrahedra)
    # return (dim4)
    #
    cdef int flag
    cdef int i1,i2
    cdef int counter0,counter1,counter2,counter3,counter4
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c,v1,v2,v3,v4,v5,v6
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a,tmp2b
    #cdef np.ndarray[DTYPE_int_t,ndim=2] v1,v2,v3,v4,v5,v6
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b,tmp3c,obj_surface_vertex
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,obj_common
    
    if verbose>0:
        print('         subtraction_tetrahedron_object_dev()')
    else:
        pass
    # common part between tetrahedron and obj3.
    #obj_common=intersection_using_tetrahedron(tetrahedron.reshape(1,4,6,3),obj3,path)
    #tmp1a=np.append(tmp1a,obj_common)
    #tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
    #print '    number of common tetrahedra = %d'%(len(obj_common))
    
    """
    # get vertices of common tetrahedra, which are on the obj3_surface.
    counter0=0
    tmp3b=remove_doubling_dim4(obj2)
    for i1 in range(len(tmp3b)) :
        tmp2b=tmp3b[i1]
        for i2 in range(len(obj2_surface)):
            tmp3c=obj2_surface[i2]
            flag=on_out_surface(tmp2b,tmp3c)
            if flag==0: # on surface
                if counter0==0:
                    tmp1b=tmp2b.reshape(18)
                else:
                    tmp1b=np.append(tmp1b,tmp2b)
                counter0+=1
            else: # out
                pass
    print '    number of vertices of common tetrahedra located on the surface = %d'%(counter0)
    if counter0!=0:
        tmp3b=tmp1b.reshape(len(tmp1b)/18,6,3)
        for i1 in range(len(tmp3b)):
            v1,v2,v3,v4,v5,v6=projection(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
            print 'Zz %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
    else:
        pass
    """
    #####################################
    #                                   #
    #                                   #
    #####################################
    # (1) get vertices of tetrahedron which are outside of obj2 (points A)
    counter1=0
    for i1 in range(4):
        tmp2a=tetrahedron[i1]
        for i2 in range(len(obj2)):
            tmp3a=obj2[i2]
            flag=inside_outside_tetrahedron(tmp2a,tmp3a[0],tmp3a[1],tmp3a[2],tmp3a[3])
            if flag!=0: # outside
                if counter1==0:
                    tmp1a=tmp2a.reshape(18)
                else:
                    tmp1a=np.append(tmp1a,tmp2a)
                counter1+=1
            else:
                pass
    if counter1==0: # tetrahedron is inside obj3
        #print 'inconsistency: common part is empty?'
        return np.array([[[[0]]]])
    else:
        pass
        #if counter0!=0:
        #    tmp1a=np.append(tmp1a,tmp1b)
        #else:
        #    pass
    #####################################
    #                                   #
    #                                   #
    #####################################    
    """  point_b
    # (2) get vertices of obj2_surface which are on obj3_surface (points B)
    tmp3b=obj2_surface.reshape(len(obj2_surface)/18,6,3)
    for i1 in range(len(tmp3b)):
        tmp2b=tmp3b[i1]
        for i2 in range(len(obj3_surface)):
            tmp3c=obj3_surface[i2]
            flag=on_out_surface(tmp2b,tmp3c)
            if flag==0: # on surface
                if counter0==0:
                    tmp1b=tmp2b.reshape(18)
                else:
                    tmp1b=np.append(tmp1b,tmp2b)
                counter0+=1
            else: # out
                pass
    print '    number of points B = %d'%(counter0)
    """
    #####################################
    #                                   #
    #                                   #
    #####################################
    # (3) get points in (points B) which are inside tetrahedron (points C)
    #tmp1b=np.append(tmp1b,obj3_surface)
    tmp1b=np.append(point_b,obj3_surface)
    tmp3b=tmp1b.reshape(int(len(tmp1b)/18),6,3)
    #tmp3b=remove_doubling_dim3_in_perp_space(tmp3b)
    counter2=0
    for i1 in range(len(tmp3b)):
        tmp2b=tmp3b[i1]
        flag=inside_outside_tetrahedron(tmp2b,tetrahedron[0],tetrahedron[1],tetrahedron[2],tetrahedron[3])
        if flag==0: # inside
            if counter2==0:
                tmp1b=tmp2a.reshape(18)
            else:
                tmp1b=np.append(tmp1b,tmp2b)
            counter2+=1
        else:
            pass
    tmp3b=tmp1b.reshape(int(len(tmp1b)/18),6,3)
    tmp3b=remove_doubling_dim3_in_perp_space(tmp3b)
    print('    points C:')
    #for i1 in range(len(tmp3b)):
    #    v1,v2,v3,v4,v5,v6=projection(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
    #    print 'Zz %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
    #####################################
    #                                   #
    #                                   #
    #####################################
    # (4) merge points A and points C
    if counter1==0 and counter2==0:
        return np.array([[[[0]]]])
    else:
        if counter1!=0 and counter2!=0:
            tmp1c=np.append(tmp1a,tmp1b)
        elif counter1!=0 and counter2==0:
            tmp1c=tmp1a
        elif counter1==0 and counter2!=0:
            tmp1c=tmp1b
        else:
            pass
        if verbose>0:
            print('    merge points A and points B -> tetrahedralization')
        tmp3c=tmp1c.reshape(int(len(tmp1c)/18),6,3)
        tmp3c=remove_doubling_dim3_in_perp_space(tmp3c)
        if len(tmp3c)>=4:
            tmp4c=tetrahedralization_points(tmp3c,verbose-1)
            if verbose>0:
                print('    number of tetrahedron = %d'%(len(tmp4c)))
            #####################################
            #                                   #
            #                                   #
            #####################################
            # (5) get tetrahedra in tmp4c which are outside of obj2
            counter3=0
            for i1 in range(len(tmp4c)):
                counter4=0
                for i2 in range(4):
                    tmp2c=tmp4c[i1][i2]
                    for i3 in range(len(obj2)):
                        tmp3a=obj2[i3]
                        flag=inside_outside_tetrahedron(tmp2c,tmp3a[0],tmp3a[1],tmp3a[2],tmp3a[3])
                        if flag==0: # inside
                            counter4+=1
                            break
                        else:
                            pass
                if counter4<4:
                    if counter3==0:
                        tmp1c=tmp4c[i1].reshape(72)
                    else:
                        tmp1c=np.append(tmp1c,tmp4c[i1])
                    counter3+=1
                else:
                    pass
            if verbose>0:
                print('    number of tetrahedron outside of obj2 = %d'%(counter3))
            if counter3!=0:
                return tmp1c.reshape(int(len(tmp1c)/72),4,6,3)
            else:
                return np.array([[[[0]]]])
        else:
            return np.array([[[[0]]]])

cdef int print_egdes_xyz_dim3(np.ndarray[DTYPE_int_t, ndim=4] egdes):
    cdef int i1
    cdef np.ndarray[DTYPE_int_t,ndim=1] a4,a5,a6
    for i1 in range(len(egdes)):
        for i2 in range(2):
            a4,a5,a6=projection3(egdes[i1][i2][0],egdes[i1][i2][1],egdes[i1][i2][2],egdes[i1][i2][3],egdes[i1][i2][4],egdes[i1][i2][5])
            print('Xx %8.6f %8.6f %8.6f'%\
            ((a4[0]+a4[1]*TAU)/(a4[2]),(a5[0]+a5[1]*TAU)/(a5[2]),(a6[0]+a6[1]*TAU)/(a6[2])))
    return 0
