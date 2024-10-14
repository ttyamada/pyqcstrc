#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import itertools
from pyqcstrc.dode2.math1 import (add, 
                                matrixpow, 
                                dot_product, 
                                dot_product_1, 
                                sub_vectors, 
                                add_vectors,
                                )
from pyqcstrc.dode2.utils import (remove_doubling_in_perp_space, 
                                remove_doubling,
                                )
from pyqcstrc.dode2.numericalc import (projection_numerical,
                                projection3_numerical,
                                numerical_vector,
                                length_numerical,
                                )
import numpy as np

EPS=1e-6
V0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)

def symop_obj(symop,obj,centre):
    """ Apply a symmetric operation on an object around given centre. in TAU-style
    
    """
    ndim=obj.ndim
    if ndim==3:
        return symop_vecs(symop,obj,centre)
    elif ndim==4:
        obj1=np.zeros(obj.shape,dtype=np.int64)
        i=0
        for vts in obj:
            obj1[i]=symop_vecs(symop,vts,centre)
            i+=1
        return obj1
    elif ndim==2:
        return symop_vec(symop,obj,centre)
    else:
        print('object has an incorrect shape!')
        return 

def symop_vecs(symop,vts,centre):
    """ Apply a symmetric operation on set of vectors around given centre. in TAU-style
    
    """
    out=np.zeros(vts.shape,dtype=np.int64)
    i=0
    for vt in vts:
        out[i]=symop_vec(symop,vt,centre)
        i+=1
    return out

def symop_vec(symop,vt,centre):
    """ Apply a symmetric operation on a vector around given centre. in TAU-style
    """
    vt=sub_vectors(vt,centre)
    vt=dot_product_1(symop,vt)
    return add_vectors(vt,centre)

def generator_obj_symmetric_obj(obj,centre,pg='-12m2'):
    """
    """
    if obj.ndim==3 or obj.ndim==4:
        if np.all(centre==V0):
            mop=dodesymop_array(pg)
        else:
            lst_site_symmetry=site_symmetry(centre)
            mop=[]
            tmp=dodesymop_array(pg)
            for i in lst_site_symmetry:
                mop.append(tmp[i])
        num=len(mop)
        shape=tuple([num])
        a=np.zeros(shape+obj.shape,dtype=np.int64)
        for i,op in enumerate(mop):
            a[i]=symop_obj(op,obj,centre)
        if obj.ndim==4:
            n1,n2,_,_=obj.shape
            a=a.reshape(num*n1,n2,6,3)
        return a
    else:
        print('object has an incorrect shape!')
        return

def generator_obj_symmetric_triangle(obj,centre,pg='-12m2'):
    """
    """
    return generator_obj_symmetric_obj(obj,centre,pg)
    
#def generator_obj_symmetric_tetrahedron(obj,centre):
#    return generator_obj_symmetric_obj(obj,centre)

def generator_obj_symmetric_vector_specific_symop(obj,centre,index_of_symmetry_operation,pg='-12m2'):
    """
    vector: triangles
    (6,3)
    """
    # using specific symmetry operations
    if obj.ndim==2:
        #mop=dodesymop(pg)
        mop=dodesymop_array(pg)
        shape=tuple([len(index_of_symmetry_operation)])
        a=np.zeros(shape+obj.shape,dtype=np.int64)
        j=0
        for i1 in index_of_symmetry_operation:
            a[j]=symop_obj(mop[i1],obj,centre)
            j+=1
        return a
    else:
        print('object has an incorrect shape!')
        return

def generator_obj_symmetric_triangle_specific_symop(obj,centre,index_of_symmetry_operation,pg='-12m2'):
    """
    triangle: triangles
    (3,6,3)
    """
    # using specific symmetry operations
    if obj.ndim==3:
        #mop=dodesymop(pg)
        mop=dodesymop_array(pg)
        shape=tuple([len(index_of_symmetry_operation)])
        a=np.zeros(shape+obj.shape,dtype=np.int64)
        j=0
        for i1 in index_of_symmetry_operation:
            a[j]=symop_obj(mop[i1],obj,centre)
            j+=1
        return a
    else:
        print('object has an incorrect shape!')
        return

def generator_obj_symmetric_obj_specific_symop(obj,centre,index_of_symmetry_operation,pg='-12m2'):
    """
    obj: a set of triangles
    (n,3,6,3)
    """
    # using specific symmetry operations
    if obj.ndim==4:
        #mop=dodesymop(pg)
        mop=dodesymop_array(pg)
        shape=tuple([len(index_of_symmetry_operation)])
        a=np.zeros(shape+obj.shape,dtype=np.int64)
        j=0
        for i1 in index_of_symmetry_operation:
            a[j]=symop_obj(mop[i1],obj,centre)
            j+=1
        n1,n2,_,_,_=a.shape
        #return a.reshape(int(len(a)/54),3,6,3)
        return a.reshape(n1*n2,3,6,3)
    else:
        print('object has an incorrect shape!')
        return
    
def generator_obj_symmetric_triangle_0(obj,centre,symmetry_operation_index,pg='-12m2'):
    """
    """
    #mop=dodesymop(pg)
    mop=dodesymop_array(pg)
    return symop_obj(mop[symmetry_operation_index],obj,centre)

def generator_obj_symmetric_vec(vectors,centre,pg='-12m2'):
    """
    """
    return generator_obj_symmetric_obj(vectors,centre,pg)

def generator_equivalent_vectors(vectors,centre,pg='-12m2'):
    """
    """
    a=generator_obj_symmetric_obj(vectors,centre,pg)
    return remove_doubling_in_perp_space(a)

def generator_equivalent_vec(vector,centre,pg='-12m2'):
    """
    """
    a=generator_obj_symmetric_obj(vector,centre,pg)
    return remove_doubling(a)

def dodesymop(pg='-12m2'):
    """
    pg: point group, '12/mmm', '-12m2'
    """
    
    #"""
    # dodecagonal symmetry operations
    # c12
    # y, z, u, −x + z, v,
    m1=np.array([[0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [-1, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
    # mirror normal to y-axis
    # x,y − u,x − z,−u,−v
    m2=np.array([[1, 0, 0, 0, 0, 0],\
                [ 0, 1, 0,-1, 0, 0],\
                [ 1, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
    # mirror normal to z-axis
    # x,y,z,u,−v
    m3=np.array([[1, 0, 0, 0, 0, 0],\
                [ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
    # inversion
    # -x,-y,-z,-u,−v
    m4=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
    # s12
    # IR12: -y, -z, -u, x-z, -v
    m5=np.dot(m4,m1)
    
    symop=[]
    if pg=='12/mmm':
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    for i4 in range(12):
                        s1=matrixpow(m1,i4) # c12
                        s2=matrixpow(m2,i3) # mirror,y
                        s3=matrixpow(m3,i2) # mirror,z
                        s4=matrixpow(m4,i1) # inversion
                        tmp=np.dot(s2,s1)
                        tmp=np.dot(s3,tmp)
                        tmp=np.dot(s4,tmp)
                        symop.append(tmp)
    elif pg=='-12m2':
        for i1 in range(2):
            for i2 in range(12):
                s1=matrixpow(m5,i2) # s12
                s2=matrixpow(m2,i1) # mirror,y
                tmp=np.dot(s2,s1)
                symop.append(tmp)
    elif pg=='-12':
        for i1 in range(12):
            s1=matrixpow(m5,i1) # s12
            symop.append(s1)
    elif pg=='12':
        for i1 in range(12):
            s1=matrixpow(m1,i1) # c12
            symop.append(s1)
    return symop
    
def dodesymop_array(pg='-12m2'):
    """
    """
    ops=matrix_dode_sym()
    num=0
    if pg=='12/mmm':
        m1=ops[0] # c12
        m2=ops[1] # mirror,y
        m3=ops[2] # mirror,z
        m4=ops[3] # inversion
        symop=np.zeros((96,6,6),dtype=np.int64)
        for i1 in range(2):
            s4=matrixpow(m4,i1)
            for i2 in range(2):
                s3=matrixpow(m3,i2)
                for i3 in range(2):
                    s2=matrixpow(m2,i3)
                    for i4 in range(12):
                        s1=matrixpow(m1,i4)
                        tmp=np.dot(s2,s1)
                        tmp=np.dot(s3,tmp)
                        tmp=np.dot(s4,tmp)
                        symop[num]=tmp
                        num+=1
    elif pg=='12mm':
        m1=ops[0] # C12
        m2=ops[1] # mirror,y
        m3=ops[3] # inversion
        symop=np.zeros((48,6,6),dtype=np.int64)
        for i1 in range(2):
            s3=matrixpow(m3,i1)
            for i2 in range(2):
                s2=matrixpow(m2,i2)
                for i3 in range(12):
                    s1=matrixpow(m1,i3)
                    tmp=np.dot(s2,s1)
                    tmp=np.dot(s3,tmp)
                    symop[num]=tmp
                    num+=1
    elif pg=='-12m2':
        m1=ops[4] # s12
        m2=ops[1] # mirror,y
        symop=np.zeros((24,6,6),dtype=np.int64)
        for i1 in range(2):
            s2=matrixpow(m2,i1)
            for i2 in range(12):
                s1=matrixpow(m1,i2)
                tmp=np.dot(s2,s1)
                symop[num]=tmp
                num+=1
    elif pg=='-12':
        m1=ops[4] # s12
        symop=np.zeros((12,6,6),dtype=np.int64)
        for i1 in range(12):
            s1=matrixpow(m1,i1)
            symop[num]=s1
            num+=1
    elif pg=='12':
        m1=ops[0] # c12
        symop=np.zeros((12,6,6),dtype=np.int64)
        for i1 in range(12):
            s1=matrixpow(m1,i1)
            symop[num]=s1
            num+=1
    return symop
    
def matrix_dode_sym():
    # c12
    # y, z, u, −x + z, v,
    m1=np.array([[0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [-1, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
    # mirror normal to y-axis
    # x,y − u,x − z,−u,−v
    m2=np.array([[1, 0, 0, 0, 0, 0],\
                [ 0, 1, 0,-1, 0, 0],\
                [ 1, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
    # mirror normal to z-axis
    # x,y,z,u,−v
    m3=np.array([[1, 0, 0, 0, 0, 0],\
                [ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
    # inversion
    # -x,-y,-z,-u,−v
    m4=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
    # s12
    # IR12: -y, -z, -u, x-z, -v
    #m5=np.array([[0,-1, 0, 0, 0, 0],\
    #            [ 0, 0,-1, 0, 0, 0],\
    #            [ 0, 0, 0,-1, 0, 0],\
    #            [ 1, 0,-1, 0, 0, 0],\
    #            [ 0, 0, 0, 0,-1, 0],\
    #            [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
    m5=np.dot(m4,m1)
    symop=np.zeros((5,6,6),dtype=np.int64)
    symop[0]=m1
    symop[1]=m2
    symop[2]=m3
    symop[3]=m4
    symop[4]=m5
    return symop

"""
def mattrix_dode_sym():
    # dodecagonal symmetry operations
    # c12
    m1=np.array([[ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [-1, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
    # mirror
    m2=np.array([[ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 1, 0, 0, 0, 0],\
                [ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
    # inversion
    m3=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
    symop=np.zeros((3,6,6),dtype=np.int64)
    symop[0]=m1
    symop[1]=m2
    symop[2]=m3
    return symop
"""
    
def generator_symmetric_vec_specific_symop(vector,centre,index_of_symmetry_operation,pg='-12m2'):
    #mop=dodesymop(pg)
    mop=dodesymop_array(pg)
    #a=np.zeros(len(index_of_symmetry_operation))
    return symop_vec(mop[symmetry_operation_index],vector,centre)
    
def translation(ndim):
    """translational symmetry
    """
    symop=[]
    lst=[-1,0,1]
    tmp=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    symop.append(tmp)
    if ndim==4:
        for i1 in lst:
            for i2 in lst:
                for i3 in lst:
                    for i4 in lst:
                        tmp=np.array([[i1,0,1],[i2,0,1],[i3,0,1],[i4,0,1],[0,0,1],[0,0,1]])
                        symop.append(tmp)
    else:
        for i1 in lst:
            for i2 in lst:
                for i3 in lst:
                    for i4 in lst:
                        for i5 in lst:
                            tmp=np.array([[i1,0,1],[i2,0,1],[i3,0,1],[i4,0,1],[i5,0,1],[0,0,1]])
                            symop.append(tmp)
    return np.array(symop)

################ 
# numeric
################
def generator_equivalent_numeric_vector_specific_symop(vn,index_of_symmetry_operation,pg='-12m2'):
    mop=dodesymop_array(pg)
    out=np.zeros((len(index_of_symmetry_operation),6),dtype=np.float64)
    for i1 in index_of_symmetry_operation:
        out[i1]=mop[i1]@vn
    return out
    
def generator_equivalent_numeric_vectors_specific_symop(vns,index_of_symmetry_operation,pg='-12m2'):
    mop=dodesymop_array(pg)
    num1=len(index_of_symmetry_operation)
    out=np.zeros((num1,len(vns),6),dtype=np.float64)
    for i1 in range(num1):
        for vn in vns:
            out[i1]=mop[i1]@vn
    return out
    
################ 
# site symmetry
################
def site_symmetry(site,dim=5,pg='-12m2'):
    """symmetry operators of site symmetry group of 'site'.
    
    Args:
        site (numpy.ndarray):
            xyz coordinate of the site.
            The shape is (6,3).
        ndim (int):
            dimension, 4 or 5
        verbose (int)
    
    Returns:
        List of index of symmetry operators of the site symmetry group (list):
            The symmetry operators leaves xyz identical.
    """
    
    symop=dodesymop_array(pg)
    traop=translation(dim)
    
    list1=[]
    for i2,op in enumerate(symop):
        site1=symop_vec(op,site,V0)
        flag=0
        for top in traop:
            site2=add_vectors(site1,top)
            tmp=sub_vectors(site,site2)
            if length_numerical(tmp)<EPS:
                list1.append(i2)
                break
            else:
                pass
    return remove_overlaps(list1)

def equivalent_positions(site,dim=5,pg='-12m2'):
    """symmetry operators that generate equivalent positions of 'site'.
    
    Args:
        site (numpy.ndarray):
            xyz coordinate of the site.
            The shape is (6,3).
        ndim (int):
            dimension, 4 or 5
        verbose (int)
    
    Returns:
        List of index of symmetry operators of the equivalent positions of 'site'. (list):
    """
    symop=dodesymop_array(pg)
    #print('len(symop):',len(symop))
    
    lst=site_symmetry(site,dim,pg)
    #print('len(lst):',len(lst))
    a=set(range(len(symop)))
    b=set(lst)-{0}
    #return list(a-b)
    
    idx_equiv=list(a-b)
    return idx_equiv
    """
    vts=generator_obj_symmetric_vector_specific_symop(site,V0,idx_equiv,pg)
    idx_site_unit=[]
    for i,vt in enumerate(vts):
        vn=numerical_vector(vt)
        print('%d: %6.4f %6.4f %6.4f %6.4f %6.4f'%(i,vn[0],vn[1],vn[2],vn[3],vn[4]))
        #print(vn)
        if vn[0]>=0 and vn[1]>=0 and vn[2]>=0 and vn[3]>=0:
            #print(vn)
            idx_site_unit.append(i)
    #print('idx_site_unit:',idx_site_unit)
    return idx_site_unit
    """

def coset(site,dim=5,pg='-12m2'):
    """left coset decomposition
    """
    symop=dodesymop_array(pg)
    num_ord_g=len(symop)
    
    list1=site_symmetry(site,dim,pg)
    num_ord_h=len(list1)
    
    if num_ord_g==num_ord_h:
        return [0]
        
    elif num_ord_h==1:
        return list(range(num_ord_g))
        
    elif int(num_ord_g/num_ord_h)-num_ord_g/num_ord_h<EPS:
        out=[0]
        num_index=int(num_ord_g/num_ord_h)
        if num_index==1:
            return out
        else:
            # lst_index_else: list of index of symmetry operators which are not in the G.
            tmp=set(range(num_ord_g))-set(list1)
            lst_index_else=list(tmp)
            
            list4=[]
            for i2 in lst_index_else:
                lst_tmp=[]
                for i1 in list1: # i1-th symmetry operation of the site symmetry (point group, H)
                    op1=np.dot(symop[i2],symop[i1])
                    for i3,op in enumerate(symop):
                        if np.all(op==op1):
                            lst_tmp.append(i3)
                            break
                        else:
                            pass
                list4.append(lst_tmp)
            
            lst=list(range(len(list4)))
            for pair in itertools.combinations(lst,num_index-1):
                a=[]
                for i1 in pair:
                    a+=list4[i1]
                c=remove_overlaps(a)
                if len(c)==len(a):
                    break
            for i in pair:
                out.append(lst_index_else[i])
            return out
    else:
        print('error')
        return 

def coset_a(site,dim=5,pg='-12m2'):
    """left coset decomposition
    """
    print('============= in coset_a() =============')
    symop=dodesymop_array(pg)
    num_ord_g=len(symop)
    
    list1=site_symmetry(site,dim,pg)
    num_ord_h=len(list1)
    
    print('site simmetry (list1):',list1)
    
    if num_ord_g==num_ord_h:
        return [0]
        
    elif int(num_ord_g/num_ord_h)-num_ord_g/num_ord_h<EPS:
        #out=[0]
        out=[]
        num_index=int(num_ord_g/num_ord_h)
        if num_index==1:
            return out
        else:
            # lst_index_else: list of index of symmetry operators which are not in the G.
            tmp=set(range(num_ord_g))-set(list1)
            lst_index_else=list(tmp)
            print('sym op not in site sym (lst_index_else):',lst_index_else)
            list4=[]
            for i2 in lst_index_else:
                lst_tmp=[]
                for i1 in list1: # i1-th symmetry operation of the site symmetry (point group, H)
                    op1=np.dot(symop[i2],symop[i1])
                    for i3,op in enumerate(symop):
                        if np.all(op==op1):
                            lst_tmp.append(i3)
                            break
                        else:
                            pass
                list4.append(lst_tmp)
                
            for i in range(len(list4)):
                print('%d'%(lst_index_else[i]),list4[i])
                
            """
            lst=list(range(len(list4)))
            for pair in itertools.combinations(lst,num_index-1):
                a=[]
                for i1 in pair:
                    a+=list4[i1]
                c=remove_overlaps(a)
                if len(c)==len(a):
                    break
            for i in pair:
                out.append(lst_index_else[i])
            return out
            """
            
            tmp=[]
            lst=list(range(len(list4)))
            for pair in itertools.combinations(lst,num_index-1):
                a=[]
                for i1 in pair:
                    a+=list4[i1]
                c=remove_overlaps(a)
                if len(c)==len(a):
                    tmp.append(pair)
            for i in range(len(tmp)):
                out0=[0]
                for j in tmp[i]:
                    out0.append(lst_index_else[j])
                out.append(out0)
                #print('#%d'%(i))
                #print(out0)
                for j1 in out0:
                    lst_tmp=[]
                    for i1 in list1:
                        op1=np.dot(symop[j1],symop[i1])
                        for i3,op in enumerate(symop):
                            if np.all(op==op1):
                                lst_tmp.append(i3)
                                break
                            else:
                                pass
                    b=remove_overlaps(lst_tmp)
                    #print(b)
            #print(out)
            #return out[17] # 5, 17
            print('============= in coset_a() =============')
            return out
    else:
        print('error')
        return 

def coset_1(site,dim=5,pg='-12m2'):
    """
    left coset decomposition
    another vasion, without using intertools
    """
    symop=dodesymop_array(pg)
    
    list1=site_symmetry(site,dim,pg)
    
    if len(symop)==len(list1):
        return [0]
    else:
        # list2: list of index of symmetry operators which are not in the G.
        tmp=range(len(symop))
        tmp=set(tmp)-set(list1)
        #list2=list(tmp)
        #list2_new=remove_overlaps(list2)
        #list2=remove_overlaps(list(tmp))
        list2=list(tmp)
        
        list4=[]
        for i2 in list2:
            list3=[]
            for i1 in list1:
                op1=np.dot(symop[i2],symop[i1]) # left coset decomposition
                for i3,op in enumerate(symop):
                    if np.all(op==op1):
                        list3.append(i3)
                        break
                    else:
                        pass
            list4.append(list3)
            
        for i2 in range(len(list4)-1):
            a=list4[i2]
            b=[]
            d=[]
            #list5=[0] # symmetry element of identity, symop[0]
            list5=[]
            list5.append(list2[i2])
            i3=i2+1
            while i3<len(list4):
                b=list4[i3]
                if len(d)==0:
                    if find_overlaps(a,b):
                        pass
                    else:
                        d=a+b
                        list5.append(list2[i3])
                else:
                    if find_overlaps(d,b):
                        pass
                    else:
                        d=d+b
                        list5.append(list2[i3])
                i3+=1
            b=remove_overlaps(d)
            if len(symop)==len(list5)*len(list1):
                break
            else:
                pass
                
        return list5

def site_symmetry_and_coset(site,dim=5,pg='-12m2',verbose=0):
    #symmetry operators in the site symmetry group G and its left coset decomposition.
    #
    #Args:
    #    site (numpy.ndarray):
    #        xyz coordinate of the site.
    #        The shape is (6,3).
    #    ndim (int):
    #        dimension, 4 or 5
    #    verbose (int)
    #
    #Returns:
    #    List of index of symmetry operators of the site symmetry group G (list):
    #        The symmetry operators leaves xyz identical.
    #    
    #    List of index of symmetry operators in the left coset representatives of the poibt group G (list):
    #        The symmetry operators generates equivalent positions of the site xyz.
    
    symop=dodesymop_array(pg)
    traop=translation(dim)
    
    # List of index of symmetry operators of the site symmetry group G.
    # The symmetry operators leaves xyz identical.
    list1=[]
    
    # List of index of symmetry operators which are not in the G.
    list2=[]
    
    if verbose>0:
        
        xyz=projection_numerical(site)
        xe=xyz[0]
        ye=xyz[1]
        ze=xyz[2]
        xi=xyz[3]
        yi=xyz[4]
        #zi=xyz[5]
        
        sn=numerical_vector(site)
        xyzn=numerical_vector(xyz)
        xen=xyzn[0]
        yen=xyzn[1]
        zen=xyzn[2]
        xin=xyzn[3]
        yin=xyzn[4]
        print('site_symmetry()')
        print(' site coordinates: %3.2f %3.2f %3.2f %3.2f %3.2f %3.2f'%(\
                                    float(sn[0]),\
                                    float(sn[1]),\
                                    float(sn[2]),\
                                    float(sn[3]),\
                                    float(sn[4]),\
                                    float(sn[5])))
        print('         in Epar : %3.2f %3.2f %3.2f'%(float(xen),float(yen),float(zen)))
        print('         in Eperp: %3.2f %3.2f'%(float(xin),float(yin)))
    else:
        pass
    
    for i2,op in enumerate(symop):
        site1=symop_vec(op,site,V0)
        flag=0
        for top in traop:
            site2=add_vectors(site1,top)
            tmp=sub_vectors(site,site2)
            if length_numerical(tmp)<EPS:
                list1.append(i2)
                flag+=1
                break
            else:
                pass
        if flag==0:
            list2.append(i2)
    
    list1_new=remove_overlaps(list1)
    list2_new=remove_overlaps(list2)
    
    if verbose>0:
        print('     multiplicity:',len(list1_new))
        print('    site symmetry:',list1_new)
    else:
        pass
    
    if len(symop)==len(list1_new):
        list5=[0]
        if verbose>0:
            print('       left coset:',list5)
        else:
            pass
    else:
        # left coset decomposition:
        list4=[]
        for i2 in list2_new:
            list3=[]
            for i1 in list1_new:
                op1=np.dot(symop[i2],symop[i1])
                for i3,op in enumerate(symop):
                    if np.all(op1==op):
                        list3.append(i3)
                        break
                    else:
                        pass
            list4.append(list3)
        
        #print('----------------')
        #for i2 in range(len(list4)):
        #    print(list4[i2])
        #print('----------------')
        
        for i2 in range(len(list4)-1):
            a=list4[i2]
            b=[]
            d=[]
            list5=[0] # symmetry element of identity, symop[0]
            list5.append(list2_new[i2])
            i3=i2+1
            while i3<len(list4):
                b=list4[i3]
                if len(d)==0:
                    if find_overlaps(a,b):
                        pass
                    else:
                        d=a+b
                        list5.append(list2_new[i3])
                else:
                    if find_overlaps(d,b):
                        pass
                    else:
                        d=d+b
                        list5.append(list2_new[i3])
                i3+=1
            b=remove_overlaps(d)
            if len(symop)==len(list5)*len(list1_new):
                if verbose>0:
                    print('       left coset:',list5)
                else:
                    pass
                break
            else:
                pass
    
    return list1_new,list5

#################
#   Utilities
#################
def remove_overlaps(l1):
    """
    Remove overlap elements in list with set method.
    
    Args:
        l1 (list):
    
    Returns:
        l2 (list)
    """
    tmp=set(l1)
    l2=list(tmp)
    l2.sort()
    return l2
    
def find_overlaps(l1,l2):
    """find overlap or not btween list1 and list2.
    
    Args:
        l1 (list):
        l2 (list):
    
    Returns:
        True : overlaping
        False: no overlap
    """
    l3=remove_overlaps(l1+l2)
    if len(l1)+len(l2)==len(l3): # no overlap
        return False
    else:
        return True

############################
# Similarity transformation
############################
def similarity_obj(obj,m):
    """similarity transformation of a triangle
    """
    out=np.zeros(obj.shape,dtype=np.int64)
    for i1,od in enumerate(obj):
        out[i1]=similarity_triangle(od,m)
    return out

def similarity_triangle(triangle,m):
    """similarity transformation of a triangle
    """
    out=np.zeros(triangle.shape,dtype=np.int64)
    for i1,vt in enumerate(triangle):
        out[i1]=similarity_vec(vt,m)
    return out

def similarity_vec(vt,m):
    """similarity transformation of a vector
    """
    vec1=[]
    op=similarity(m)
    return dot_product_1(op,vt)
    
def similarity(m):
    """Similarity transformation of Dodecagonal QC
    """
    m1=np.array([[ 1, 0, 0, -1, 0, 0],\
                [ 1, 1, 0, 0, 0, 0],\
                [ 0, 1, 1, 1, 0, 0],\
                [ 0, 0, 1, 1, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
    return matrixpow(m1.T,m)

############################
# Group
############################
def generate_multiplication_table(a,flag,ndim):
    """
    対称行列リストaから積表を求める
    Generation of a multiplication table from symmetry elements,
    for point group (flag='PG'), 
    for space group (flag='SG').
    
    input:
    list a, list of symmetry elements
    string flag: 'SG' for space group, 'PG' for point group,
    int ndim: dimention of periodic structure, (dummy for flag == 'PG')
    
    return:
    int ndarray table
        1（群をつくらない場合）
    """
    RTOL=1e-02
    ATOL=1e-03
    
    def translation_one_unit_cel(ndim):
        """
        ユニットセル1つ分だけシフトする。
        並進を含む対称操作が等価かどうか確認するときに用いる。
    
        int ndim: 3次元結晶の場合は3, icoの場合は6, decagonal, dodecagonalの場合は5
        """
        z0 = np.zeros([6,7])
        z1 = np.zeros([2,8])
    
        t1 = np.array([[1,0,0,0,0,0]]).T
        t2 = np.array([[0,1,0,0,0,0]]).T
        t3 = np.array([[0,0,1,0,0,0]]).T
    
        a1 = [-1,0,1]
    
        lst=[]
        if ndim==6:
            t4 = np.array([[0,0,0,1,0,0]]).T
            t5 = np.array([[0,0,0,0,1,0]]).T
            t6 = np.array([[0,0,0,0,0,1]]).T
            for i1 in a1:
                for i2 in a1:
                    for i3 in a1:
                        for i4 in a1:
                            for i5 in a1:
                                for i6 in a1:
                                    t = t1*i1+t2*i2+t3*i3+t4*i4+t5*i5+t6*i6
                                    tmp = np.block([z0,t])
                                    tmp = np.block([[tmp],[z1]])
                                    lst.append(tmp)
        elif ndim==3:
            for i1 in a1:
                for i2 in a1:
                    for i3 in a1:
                        t = t1*i1+t2*i2+t3*i3
                        tmp = np.block([z0,t])
                        tmp = np.block([[tmp],[z1]])
                        lst.append(tmp)
        elif ndim==5:
             t4 = np.array([[0,0,0,1,0,0]]).T
             t5 = np.array([[0,0,0,0,1,0]]).T
             for i1 in a1:
                for i2 in a1:
                    for i3 in a1:
                        for i4 in a1:
                            for i5 in a1:
                                t = t1*i1+t2*i2+t3*i3+t4*i4+t5*i5
                                tmp = np.block([z0,t])
                                tmp = np.block([[tmp],[z1]])
                                lst.append(tmp)
        else:
            pass
        return lst
    
    def equivalent(a,b,flag=0,ndim=6):
        """
        judge whether symmetric matrices a and b are equivalen or not. 
        Input
        nd.array a: symmetry element matrix
        nd.array b: symmetry element matrix
        int flag: 
            0 with translation symmetry, 
            1 without translation symmetry
        int ndim, dimension of periodic structure
        """
        counter = 0
        if flag==0:
            for t in translation_one_unit_cel(ndim):
                if np.all(np.isclose(a, b+t, rtol=RTOL, atol=ATOL)):
                    counter+=1
                    break
        else:
            if np.all(np.isclose(a, b, rtol=RTOL, atol=ATOL)):
                counter+=1
        if counter!=0:
            return True
        else:
            return False
    
    num = len(a)
    
    #print('generate_multiplication_table()')
    # 元の通し番号と6D表現行列を出力
    #print('\nSymmetry Elements')
    #for i1 in range(num):
    #    print('# %d:'%i1)
    #    print(a[i1])
    
    if flag=='SG':
        nflag = 0
    else:
        nflag = 1
        
    # 積表を作成
    lst = []
    for i1 in range(num):
        for i2 in range(num):
            b = np.dot(a[i2],a[i1])
            counter = 0
            for i3 in range(num):
                if equivalent(b,a[i3],nflag,ndim):
                    counter+=1
                    lst.append(i3)
                    break
            if counter == 0:
                lst.append(-1) # a1とa2の積がリストaの中に含まれていない場合は-1を代入する。
    if len(lst) == num**2:
        return np.array(lst).reshape(num,num)
    else:
        print('Cannot make multiplication table.')
        return 1

def check_closure(mul_table,combination):
    # 閉包性(closure)
    counter3 = 0
    num = len(combination)
    for i2 in range(num):
        for i3 in range(num):
            counter = 0
            for i4 in combination:
                if mul_table[i2][i3] == i4:
                    counter+=1
            if counter == 0:
                counter3+=1
                break
    if counter3 == 0:
        return True # closure
    else:
        return False

def check_identity_element(combination,num_identity=0):
    # 単位元(identity element)の存在
    counter1 = 0
    for i2 in range(len(combination)):
        if combination[i2] == num_identity:
            counter1+=1
            break
    if counter1 == 1:
        return True
    else:
        return False

def check_inverse_element(mul_table,num_identity=0):
    # 逆元(inverse element)の存在
    (m,n)=mul_table.shape
    if m == n:
        counter2 = 0
        for i2 in range(m):
            counter = 0
            for i3 in range(m):
                if mul_table[i2][i3] == num_identity:
                    counter+=1
                    break
            if counter != 1:
                counter2+=1
                break
        if counter2 == 0:
            return True
        else:
            return False
    else:
        return False

def check_connectivity(mul_table,combination):
    # 結合律(connectivity)
    counter3=0
    num = len(combination)
    for i2 in range(num):
        for i3 in range(num):
            for i4 in range(num):
                ab = mul_table[i2][i3]
                ab =combination.index(ab)
                abc1 = mul_table[ab][i4]
                bc = mul_table[i3][i4]
                bc =combination.index(bc)
                abc2 = mul_table[i2][bc]
                if abc1 != abc2:
                    counter3+=1
                    break
    if counter3 == 0:
        return True
    else:
        return False

def check_group(a,flag='SG',ndim=6):
    """
    """
    table=generate_multiplication_table(a,flag,ndim)
    comb=list(range(len(a)))
    
    if check_closure(table,comb): # 閉包性のチェック
        #if check_identity_element(comb,num_identity): # 単位元の存在
        # 前処理で単位元を必ず含んでいる為、単位元の存在を再度確認する必要はない。
        num_identity=0
        if check_inverse_element(table,num_identity): # 逆元の存在
            if check_connectivity(table,comb): # 結合律
                return True
            else:
                return False
        else:
            return False
    else:
        return False

def stereographic_projection(symop,idx_site,idx_coset,vn0):
    #print('Stereographic projection')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,8))
    
    x1e=[]
    y1e=[]
    x2e=[]
    y2e=[]
    x1i=[]
    y1i=[]
    x2i=[]
    y2i=[]
    for idx in idx_site:
        sop=symop[idx]
        vn=np.dot(sop,vn0)
        p=projection_numerical(vn)
        xe=p[0]
        ye=p[1]
        ze=p[2]
        xi=p[3]
        yi=p[4]
        zi=p[5]
        dde=np.sqrt(xe**2+ye**2+ze**2)+EPS
        ddi=np.sqrt(xi**2+yi**2+zi**2)+EPS
        xe=xe/dde
        ye=ye/dde
        ze=ze/dde
        xi=xi/ddi
        yi=yi/ddi
        zi=zi/ddi
        if ze>=0:
            x1e.append(float(xe))
            y1e.append(float(ye))
        else:
            x2e.append(float(xe))
            y2e.append(float(ye))
        if zi>=0:
            x1i.append(float(xi))
            y1i.append(float(yi))
        else:
            x2i.append(float(xi))
            y2i.append(float(yi))
            
    x3e=[]
    y3e=[]
    x4e=[]
    y4e=[]
    x3i=[]
    y3i=[]
    x4i=[]
    y4i=[]
    for idx in idx_coset:
        sop=symop[idx]
        vn=np.dot(sop,vn0)
        p=projection_numerical(vn)
        print(p)
        xe=p[0]
        ye=p[1]
        ze=p[2]
        xi=p[3]
        yi=p[4]
        zi=p[5]
        dde=np.sqrt(xe**2+ye**2+ze**2)
        ddi=np.sqrt(xi**2+yi**2+zi**2)
        xe=xe/dde
        ye=ye/dde
        ze=ze/dde
        xi=xi/ddi
        yi=yi/ddi
        zi=zi/ddi
        if ze>=0:
            x3e.append(float(xe))
            y3e.append(float(ye))
        else:
            x4e.append(float(xe))
            y4e.append(float(ye))
        if zi>=0:
            x3i.append(float(xi))
            y3i.append(float(yi))
        else:
            x4i.append(float(xi))
            y4i.append(float(yi))
            
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('site symmetry (par)')
    ax1.scatter([0], [0], s=30000, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax1.scatter(x2e, y2e, s=200, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax1.scatter(x1e, y1e, s=40,  marker='o', color='black', alpha=1.0, edgecolors='black')
    ax1.set_xlim(-1.3,1.3)
    ax1.set_ylim(-1.3,1.3)
    ax1.axis("off")
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('coset (par)')
    ax2.scatter([0], [0], s=30000, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax2.scatter(x4e, y4e, s=200, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax2.scatter(x3e, y3e, s=40,  marker='o', color='black', alpha=1.0, edgecolors='black')
    ax2.set_xlim(-1.3,1.3)
    ax2.set_ylim(-1.3,1.3)
    ax2.axis("off")
    
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('site symmetry (perp)')
    ax3.scatter([0], [0], s=30000, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax3.scatter(x2i, y2i, s=200, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax3.scatter(x1i, y1i, s=40,  marker='o', color='black', alpha=1.0, edgecolors='black')
    ax3.set_xlim(-1.3,1.3)
    ax3.set_ylim(-1.3,1.3)
    ax3.axis("off")
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('coset (perp)')
    ax4.scatter([0], [0], s=30000, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax4.scatter(x4i, y4i, s=200, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax4.scatter(x3i, y3i, s=40,  marker='o', color='black', alpha=1.0, edgecolors='black')
    ax4.set_xlim(-1.3,1.3)
    ax4.set_ylim(-1.3,1.3)
    ax4.axis("off")
    
    #plt.axis("off")
    plt.show()
    return 0

if __name__ == '__main__':
    
    # test
    
    import random
    from numericalc import (numerical_vectors,
                            numerical_vector,
                            numeric_value,)
                            
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
    
    def generate_random_triangle():
        return generate_random_vectors(3)
    """
    print("TEST: symop_vec()")
    symop=dodesymop()
    vt=generate_random_vector()
    counter=0
    cen0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    for sop in symop:
        #
        # calc using symop_vec
        svt=symop_vec(sop,vt,cen0)
        svn1=numerical_vector(svt)
        #print(svn1)
        #
        # calc using no.dot with float values
        vn=numerical_vector(vt)
        svn2=np.dot(sop,vn)
        #print(svn2)
        if np.allclose(svn1,svn2):
            pass
        else:
            counter+=1
    if counter==0:
        print('symop_vec: correct')
    else:
        print('symop_vec: worng')
        
    nset=4
    vts=generate_random_vectors(nset)
    print(vts)
    svts=symop_vecs(symop[1],vts,cen0)
    print(svts)
    """
        
    dim=5
    
    """
    vn0=np.array([1, 2, 3, 4, 5, 0]) 
    #vn0=np.array([0, 0, 0, 1, 1, 0])
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for sop in symop:
        vn=np.dot(sop,vn0)
        pvn=projection_numerical(vn)
        x=pvn[0]
        y=pvn[1]
        z=pvn[2]
        dd=np.sqrt(x**2+y**2+z**2)
        x=x/dd
        y=y/dd
        z=z/dd
        if z>0:
            x1.append(float(x))
            y1.append(float(y))
            print('A %5.4f %5.4f %5.4f'%(x,y,z))
        else:
            x2.append(float(x))
            y2.append(float(y))
            print('B %5.4f %5.4f %5.4f'%(x,y,z))
    ax.scatter([0], [0], s=31415,  marker='o', color='white', alpha=1.0, edgecolors='black')
    ax.scatter(x2, y2, s=200, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax.scatter(x1, y1, s=40,  marker='o', color='black', alpha=1.0, edgecolors='black')
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    plt.show()
    """
    
    #pg='12/mmm'
    #pg='12mm'
    #pg='-12m2'
    pg='-12'
    #pg='12'
    #symop=dodesymop()
    symop=dodesymop_array(pg)
    
    ########################################
    # 生成元から作った集合が群をなすかどうかを確認
    ########################################
    flag='PG'
    #flag='SG'
    dim=5
    if check_group(symop,flag,dim):
        print('checking group: pass')
    else:
        print('checking group: fail')
    ########################################
    ########################################
    
    # Symmetric positions, P\bar{12}m2(12^5mm)
    V_1a =np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]) # ( 0,  0,  0,  0,  0) # \bar{12}m2(12^5mm)
    V_2a =np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[1,0,4],[0,0,1]]) # ( 0,  0,  0,  0, u1) # 6mm(6^5mm)
    V_4a =np.array([[0,0,1],[2,0,3],[0,0,1],[1,0,3],[1,0,4],[0,0,1]]) # ( 0,2/3,  0,1/3, u2) # 3m(3^2m)
    V_6a =np.array([[0,0,1],[1,0,2],[0,0,1],[0,0,1],[1,0,4],[0,0,1]]) # ( 0,1/2,  0,  0, u3) # mm2(mm1)
    V_6b =np.array([[0,0,1],[1,0,2],[1,0,2],[0,0,1],[1,0,4],[0,0,1]]) # ( 0,1/2,1/2,  0, u4) # mm2(mm1)
    V_12a=np.array([[0,0,1],[1,0,2],[1,0,3],[0,0,1],[1,0,4],[0,0,1]]) # ( 0,1/2,  z,  0, u5) # m11(m11)
    #site=V_1a
    #site=V_2a
    site=V_4a
    #site=V_6a
    #site=V_6b
    #site=V_12a
    
    #vn0=np.array([0.0, 1.0, 0.1, 0.0, 0.1, 0])
    vn0=numerical_vector(site)
    
    #idx_site,idx_coset=site_symmetry_and_coset(site,dim,pg)
    #print(' idx_site:',idx_site)
    #print('idx_coset:',idx_coset)
    #print('\n')
    idx_site=site_symmetry(site,dim,pg)
    print(' idx_site:',idx_site)
    #idx_equiv=equivalent_positions(site,dim,pg)
    #print('idx_equiv:',idx_equiv)
    #vts=generator_obj_symmetric_vector_specific_symop(site,V0,idx_equiv,pg)
    #idx_site_unit=[]
    #for i,vt in enumerate(vts):
    #    #print('%d'%(i),vt)
    #    vn=numerical_vector(vt)
    #    #print(vn)
    #    if vn[0]>=0 and vn[1]>=0 and vn[2]>=0 and vn[3]>=0:
    #        print(vn)
    #        idx_site_unit.append(i)
    #print('idx_site_unit:',idx_site_unit)
    
    #print('idx_coset:',idx_coset)
    
    #idx_coset=coset(site,dim,pg)
    idx_coset=coset_a(site,dim,pg) # 全ての余剰類分解を計算
    print('idx_coset:')
    num=0
    for a in idx_coset:
        print('  %d'%(num),a)
        num+=1
        
    RTOL=1e-02
    ATOL=1e-03
    
    print('\n checking...')
    lst0=[]
    for b in idx_coset[0]:
        vn=symop[b]@vn0
        lst0.append(vn)
        print(vn)
    #print(lst0)
    print('\n')
    for i in range(1,len(idx_coset)):
        lst1=[]
        for b in idx_coset[i]:
            vn=symop[b]@vn0
            lst1.append(vn)
            #print(vn)
        counter=0
        for l1 in lst1:
            for l0 in lst0:
                #if np.all(l1==l0):
                if np.all(np.isclose(l0,l1,rtol=RTOL, atol=ATOL)):
                    counter+=1
                    break
                else:
                    pass
        if counter==len(idx_coset):
            print('%d: ok'%(i))
        else:
            print('%d: diff'%(i),counter)
            
            #lst1.append(vn)
        #tmp=set(lst1)
        #l2=list(tmp)
        #l2.sort()
        #print(l2)
        
    idx_coset=idx_coset[17]
    print(idx_coset)
    
    print('Stereographic projection')
    stereographic_projection(symop,idx_site,idx_coset,vn0)
    