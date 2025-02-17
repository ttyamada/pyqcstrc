#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import itertools
from octa2.math1 import (add, 
                        matrixpow, 
                        dot_product, 
                        dot_product_1, 
                        sub_vectors, 
                        add_vectors,
                        )
from octa2.utils import (remove_doubling_in_perp_space, 
                        remove_doubling,
                        )
from octa2.numericalc import (projection_numerical,
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

def generator_obj_symmetric_obj(obj,centre,pg=None):
    """
    """
    if obj.ndim==3 or obj.ndim==4:
        if np.all(centre==V0):
            mop=octasymop_array()
        else:
            lst_site_symmetry=site_symmetry(centre)
            mop=[]
            tmp=octasymop_array()
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

def generator_obj_symmetric_triangle(obj,centre,pg=None):
    """
    """
    return generator_obj_symmetric_obj(obj,centre,pg)

def generator_obj_symmetric_vector_specific_symop(obj,centre,index_of_symmetry_operation,pg=None):
    """
    vector: triangles
    (6,3)
    """
    # using specific symmetry operations
    if obj.ndim==2:
        mop=octasymop_array()
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

def generator_obj_symmetric_triangle_specific_symop(obj,centre,index_of_symmetry_operation,pg=None):
    """
    triangle: triangles
    (3,6,3)
    """
    # using specific symmetry operations
    if obj.ndim==3:
        mop=octasymop_array()
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

def generator_obj_symmetric_obj_specific_symop(obj,centre,index_of_symmetry_operation,pg=None):
    """
    obj: a set of triangles
    (n,3,6,3)
    """
    # using specific symmetry operations
    if obj.ndim==4:
        mop=octasymop_array()
        shape=tuple([len(index_of_symmetry_operation)])
        a=np.zeros(shape+obj.shape,dtype=np.int64)
        j=0
        for i1 in index_of_symmetry_operation:
            a[j]=symop_obj(mop[i1],obj,centre)
            j+=1
        n1,n2,_,_,_=a.shape
        return a.reshape(n1*n2,3,6,3)
    else:
        print('object has an incorrect shape!')
        return
    
def generator_obj_symmetric_triangle_0(obj,centre,symmetry_operation_index,pg=None):
    """
    """
    mop=octasymop_array()
    return symop_obj(mop[symmetry_operation_index],obj,centre)

def generator_obj_symmetric_vec(vectors,centre,pg=None):
    """
    """
    return generator_obj_symmetric_obj(vectors,centre,pg)

def generator_equivalent_vectors(vectors,centre,pg=None):
    """
    """
    a=generator_obj_symmetric_obj(vectors,centre,pg)
    return remove_doubling_in_perp_space(a)

def generator_equivalent_vec(vector,centre,pg=None):
    """
    """
    a=generator_obj_symmetric_obj(vector,centre,pg)
    return remove_doubling(a)

def octasymop_array():
    """
    Octagonal symmetry operations
    """
    
    def matrix_sym():
        # c8
        m1=np.array([[0, 1, 0, 0, 0, 0],\
                    [ 0, 0, 1, 0, 0, 0],\
                    [ 0, 0, 0, 1, 0, 0],\
                    [-1, 0, 0, 0, 0, 0],\
                    [ 0, 0, 0, 0, 1, 0],\
                    [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
        # mirror
        m2=np.array([[0, 0, 0, 1, 0, 0],\
                    [ 0, 0, 1, 0, 0, 0],\
                    [ 0, 1, 0, 0, 0, 0],\
                    [ 1, 0, 0, 0, 0, 0],\
                    [ 0, 0, 0, 0, 1, 0],\
                    [ 0, 0, 0, 0, 0, 1]],dtype=np.int64)
        symop=np.zeros((2,6,6),dtype=np.int64)
        symop[0]=m1
        symop[1]=m2
        return symop
    
    ops=matrix_sym()
    num=0
    m1=ops[0] # c8
    m2=ops[1] # mirror
    symop=np.zeros((16,6,6),dtype=np.int64)
    for i1 in range(2):
        s2=matrixpow(m2,i1)
        for i2 in range(8):
            s1=matrixpow(m1,i2)
            tmp=np.dot(s2,s1)
            symop[num]=tmp
            num+=1
    return symop
    
    
def generator_symmetric_vec_specific_symop(vector,centre,index_of_symmetry_operation,pg=None):
    mop=octasymop_array()
    return symop_vec(mop[symmetry_operation_index],vector,centre)
    

    
################ 
# numeric
################
def generator_equivalent_numeric_vector_specific_symop(vn,index_of_symmetry_operation,pg='-12m2'):
    mop=octasymop_array()
    out=np.zeros((len(index_of_symmetry_operation),6),dtype=np.float64)
    for i1 in index_of_symmetry_operation:
        out[i1]=mop[i1]@vn
    return out
    
def generator_equivalent_numeric_vectors_specific_symop(vns,index_of_symmetry_operation,pg='-12m2'):
    mop=octasymop_array()
    num1=len(index_of_symmetry_operation)
    out=np.zeros((num1,len(vns),6),dtype=np.float64)
    for i1 in range(num1):
        for vn in vns:
            out[i1]=mop[i1]@vn
    return out
    
    
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
    """Similarity transformation of Octagonal QC
    """
    m1=np.array([[ 1, 1, 0,-1, 0, 0],\
                 [ 1, 1, 1, 0, 0, 0],\
                 [ 0, 1, 1, 1, 0, 0],\
                 [-1, 0, 1, 1, 0, 0],\
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
