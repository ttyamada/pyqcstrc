#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
from pyqcstrc.dode2.math12 import (dot_product, 
                                    outer_product, 
                                    projection, 
                                    projection3, 
                                    add, 
                                    sub, 
                                    mul, 
                                    div,
                                    )
from pyqcstrc.dode2.numericalc12 import (point_on_segment, 
                                        inout_occupation_domain_numerical,
                                        numerical_value,
                                        )
DTYPE_double = np.float64
DTYPE_int = np.int64

SIN=np.sqrt(3)/2.0
TOL=1e-6 # tolerance

def check_two_vertices(vt1,vt2):
    a=projection3(vt1)
    b=projection3(vt2)
    if np.all(a==b):
        return 1
    else:
        return 0

def two_segment_into_one(line_segment_1,line_segment_2):
    
    comb=[[0,1,0,1],[0,1,1,0],[1,0,0,1]]
    #comb=[[0,1,0,1],[0,1,1,0],[1,0,0,1],[1,0,1,0]]
    counter=0
    
    for i1 in range(len(comb)):
        edge1a=line_segment_1[comb[i1][0]]
        edge1b=line_segment_1[comb[i1][1]]
        edge2a=line_segment_2[comb[i1][2]]
        edge2b=line_segment_2[comb[i1][3]]
        if check_two_vertices(edge1a,edge2a)==1: # equivalent
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
        return 

def check_two_edges(edge1,edge2,verbose=1):
    
    if verbose>0:
        print('         check_two_edges()')
    else:
        pass
    
    flag1=0
    for i1 in range(2):
        for i2 in range(2):
            flag1+=check_two_vertices(edge1[i1],edge2[i2])
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

def triangles_to_edges(triangles):
    # parameter, set of triangles
    # returns, set of edges
    out=np.zeros(3,2,6,3)
    combination=[[0,1],[0,2],[1,2]]
    for i1 in range(len(triangles)):
        for i2 in range(3):
            j1=combination[i2][0]
            j2=combination[i2][1]
            out[i1][0]=triangles[i1][j1]
            out[i1][1]=triangles[i1][j2]
    return out

def surface_cleaner(surface,num_cycle):
    # 同一平面上にある三角形を求め、グループ分けし、各グループにおいて、以下を行う．
    # 各グループにおいて、三角形の３辺が、他のどの三角形とも共有していない辺を求める
    # そして、２つの辺が１つの辺にまとめられるのであれば、まとめる
    # 辺の集合をアウトプット
    
    obj_edge_all=np.array([[[[0]]]])
    combination=[[0,1],[0,2],[1,2]]
    
    tmp4a=surface
    tmp4b=triangles_to_edges(tmp4a,verbose-1)
    tmp1c=np.array([0])
    
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
        
        if counter2==0:
            for i3 in range(i1+1,len(tmp4c)):
                if check_two_edges(tmp4c[i1],tmp4c[i3],verbose-1)==0: # equivalent
                    counter1+=1
                    skip.append(i3)
                else:
                    pass
        else:
            counter1+=1
        
        if counter1==0:
            if counter3==0:
                tmp1c=tmp4c[i1].reshape(36)
                counter3+=1
            else:
                tmp1c=np.append(tmp1c,tmp4c[i1])
        else:
            pass
    
    counter1=0
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
    
    return tmp1c.reshape(int(len(tmp1c)/36), 2,6,3)

def shift_object(obj,shift):
    """shift an object
    """
    if obj.ndim==4:
        obj_new=np.zeros(obj.shape,dtype=np.int64)
        i1=0
        for triangle in obj:
            i2=0
            for vertex in tetrahedron:
                obj_new[i1][i2]=add_vectors(vertex,shift)
                i2+=1
            i1+=1
        return obj_new
    else:
        print('object has an incorrect shape!')
        return 

def generator_obj_outline(obj):
    """remove doubling segment in a set of triangle in the OD (dim4)
    """
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
        return 
    else:
        return tmp1a.reshape(int(len(tmp1a)/36),2,6,3)

def find_unique_segments(obj):
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

def equivalent_edge(edge1,edge2):
    tmp=np.vstack([edge1,edge2])
    if len(remove_doubling_in_perp_space(tmp))!=2:
        return 1 # not equivalent
    else:
        return 0 # two edges are equivalent

#----------------------------
# Remove doubling
#----------------------------
def remove_doubling(vts):
    """Remove doubling 6d coordinates
    
    Parameters
    ----------
    obj: array
        set of 6-dimensional vectors in TAU-style
    
    Returns
    -------
    obj: array
        set of 6-dimensional vectors in TAU-style
    """
    ndim=vts.ndim
    if ndim==4:
        n1,n2,_,_=vts.shape
        num=n1*n2
        vts=vts.reshape(num,6,3)
        return np.unique(vts,axis=0)
    elif ndim==3:
        return np.unique(vts,axis=0)
    else:
        print('ndim should be 3 or 4.')
        return 

def remove_doubling_in_perp_space(vts):
    """Remove 6d coordinates which is doubled in Eperp.
    
    Parameters
    ----------
    vts: array
        set of 6-dimensional vectors in TAU-style
    
    Returns
    -------
    vts: array
        set of 6-dimensional vectors in TAU-style
    """
    ndim=vts.ndim
    if ndim==4:
        n1,n2,_,_=vts.shape
        num=n1*n2
        vst=vts.reshape(num,6,3)
    elif ndim==3:
        #num,_,_=vts.shape
        pass
    
    # first run remove_doubling()
    vts=remove_doubling(vts)
    num=len(vts)
    
    # then, remove doubling in perp space.
    a=np.zeros((num,3,3),dtype=np.int64)
    for i in range(num):
        a[i]=projection3(vts[i])
    b=np.unique(a,return_index=True,axis=0)[1]
    num=len(b)
    a=np.zeros((num,6,3),dtype=np.int64)
    for i in range(num):
        a[i]=vts[b[i]]
    return a







def obj_area_6d(obj):
    """return area of an ocupation domain
    """
    w=np.array([0,0,1])
    for triangle in obj:
        v=triangle_area_6d(triangle)
        w=add(w,v)
    return w

def triangle_area_6d(triangle):
    """return area of a triangle in SIN-style
    """
    
    def triangle_area(v0,v1,v2):
        v1=sub(v1[0],v0[0])
        v2=sub(v1[1],v0[1])
        v3=sub(v1[2],v0[2])
        a=np.vstack([v1,v2,v3])
        #
        w1=sub(v2[0],v0[0])
        w2=sub(v2[1],v0[1])
        w3=sub(v2[2],v0[2])
        b=np.vstack([w1,w2,w3])
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
    
    a=projection3(triangle[0])
    b=projection3(triangle[1])
    c=projection3(triangle[2])
    return triangle_area(a,b,c)

def get_points_inside_obj(obj, step, nstep):
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
