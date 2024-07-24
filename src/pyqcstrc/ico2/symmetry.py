#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
#sys.path.append('.')
from pyqcstrc.ico2.math1 import (add, 
                                matrixpow, 
                                dot_product, 
                                dot_product_1, 
                                sub_vectors, 
                                add_vectors,
                                )
from pyqcstrc.ico2.utils import (remove_doubling_in_perp_space, 
                                remove_doubling,
                                )
from pyqcstrc.ico2.numericalc import (length_numerical)
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

def symop_vecs(symop,tetrahedron,centre):
    """ Apply a symmetric operation on set of vectors around given centre. in TAU-style
    
    """
    tetrahedron1=np.zeros(tetrahedron.shape,dtype=np.int64)
    for i,vt in enumerate(tetrahedron):
        tetrahedron1[i]=symop_vec(symop,vt,centre)
    return tetrahedron1

def symop_vec(symop,vt,centre):
    """ Apply a symmetric operation on a vector around given centre. in TAU-style
    
    """
    vt=sub_vectors(vt,centre)
    vt=dot_product_1(symop,vt)
    return add_vectors(vt,centre)





def generator_obj_symmetric_obj(obj,centre):
    
    if obj.ndim==3 or obj.ndim==4:
        mop=icosasymop()
        num=len(mop)
        shape=tuple([num])
        a=np.zeros(shape+obj.shape,dtype=np.int64)
        i=0
        for op in mop:
            a[i]=symop_obj(op,obj,centre)
            i+=1
        if obj.ndim==4:
            n1,n2,_,_=obj.shape
            a=a.reshape(num*n1,n2,6,3)
        else:
            pass
        return a
    else:
        print('object has an incorrect shape!')
        return

def generator_obj_symmetric_surface(obj,centre):
    return generator_obj_symmetric_obj(obj,centre)
    
def generator_obj_symmetric_tetrahedron(obj,centre):
    return generator_obj_symmetric_obj(obj,centre)

def generator_obj_symmetric_vector_specific_symop(vt,centre,list_of_symmetry_operation_index):
    # using specific symmetry operations
    if vt.ndim==2:
        mop=icosasymop()
        shape=tuple([len(list_of_symmetry_operation_index)])
        a=np.zeros(shape+vt.shape,dtype=np.int64)
        for j0,i1 in enumerate(list_of_symmetry_operation_index):
            a[j0]=symop_obj(mop[i1],vt,centre)
        return a
    else:
        print('vt has an incorrect shape!')
        return
        
def generator_obj_symmetric_tetrahedron_specific_symop(obj,centre,list_of_symmetry_operation_index):
    # using specific symmetry operations
    if obj.ndim==3 or obj.ndim==4:
        mop=icosasymop()
        shape=tuple([len(list_of_symmetry_operation_index)])
        a=np.zeros(shape+obj.shape,dtype=np.int64)
        j0=0
        for i1 in list_of_symmetry_operation_index:
            a[j0]=symop_obj(mop[i1],obj,centre)
            j0+=1
        return a
    else:
        print('object has an incorrect shape!')
        return

def generator_obj_symmetric_obj_specific_symop(obj,centre,list_of_symmetry_operation_index):
    # using specific symmetry operations
    if obj.ndim==4:
        mop=icosasymop()
        shape=tuple([len(list_of_symmetry_operation_index)])
        a=np.zeros(shape+obj.shape,dtype=np.int64)
        j0=0
        for i1 in list_of_symmetry_operation_index:
            a[j0]=symop_obj(mop[i1],obj,centre)
            j0+=1
        return a
    else:
        print('object has an incorrect shape!')
        return


def generator_obj_symmetric_tetrahedron_0(obj,centre,symmetry_operation_index):
    mop=icosasymop()
    return symop_obj(mop[symmetry_operation_index],obj,centre)

def generator_obj_symmetric_vec(vectors, centre):
    return generator_obj_symmetric_obj(vectors,centre)

def generator_equivalent_vectors(vectors,centre):
    a=generator_obj_symmetric_obj
    return remove_doubling_in_perp_space(a)

def generator_equivalent_vec(vector,centre):
    a=generator_obj_symmetric_obj(vector,centre)
    return remove_doubling(a)

def icosasymop():
    # icosahedral symmetry operations
    m1=np.array([[ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 1, 0, 0, 0, 0]],dtype=np.int64)
    # mirror
    m2=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0,-1],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0,-1, 0, 0, 0]],dtype=np.int64)
    # c2
    m3=np.array([[ 0, 0, 0, 0, 0,-1],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [-1, 0, 0, 0, 0, 0]],dtype=np.int64)
    # c3
    m4=np.array([[ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0]],dtype=np.int64)
    # inversion
    m5=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0,-1]],dtype=np.int64)
    symop=[]
    for m in range(2):
        for l in range(3):
            for k in range(2):
                for j in range(2):
                    for i in range(5):
                        s1=matrixpow(m1,i) # c5
                        s2=matrixpow(m2,j) # mirror
                        s3=matrixpow(m3,k) # c2
                        s4=matrixpow(m4,l) # c3
                        s5=matrixpow(m5,m) # inversion
                        tmp=np.dot(s5,s4)
                        tmp=np.dot(tmp,s3)
                        tmp=np.dot(tmp,s2)
                        tmp=np.dot(tmp,s1)
                        symop.append(tmp)
    return symop

def icosasymop_array():
    # icosahedral symmetry operations
    m1=np.array([[ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 1, 0, 0, 0, 0]],dtype=np.int64)
    # mirror
    m2=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0,-1],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0,-1, 0, 0, 0]],dtype=np.int64)
    # c2
    m3=np.array([[ 0, 0, 0, 0, 0,-1],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [-1, 0, 0, 0, 0, 0]],dtype=np.int64)
    # c3
    m4=np.array([[ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0]],dtype=np.int64)
    # inversion
    m5=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0,-1]],dtype=np.int64)
    symop=np.zeros((120,6,6),dtype=np.int64)
    num=0
    for m in range(2):
        for l in range(3):
            for k in range(2):
                for j in range(2):
                    for i in range(5):
                        s1=matrixpow(m1,i) # c5
                        s2=matrixpow(m2,j) # mirror
                        s3=matrixpow(m3,k) # c2
                        s4=matrixpow(m4,l) # c3
                        s5=matrixpow(m5,m) # inversion
                        tmp=np.dot(s5,s4)
                        tmp=np.dot(tmp,s3)
                        tmp=np.dot(tmp,s2)
                        tmp=np.dot(tmp,s1)
                        symop[num]=tmp
                        num+=1
    return symop

def translation(brv):
    """translational symmetry
    
    brv : bravais lattce p, i, f, s, c
            s : superlattice for decagonal quasicrystal
    """
    symop=[]
    tmp=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    symop.append(tmp)
    
    if brv=='p':
        lst=[-1,0,1]
    else:
        print('not supported')
        return 
    for i1 in lst:
        for i2 in lst:
            for i3 in lst:
                for i4 in lst:
                    for i5 in lst:
                        for i6 in lst:
                            tmp=np.array([[i1,0,1],[i2,0,1],[i3,0,1],[i4,0,1],[i5,0,1],[i6,0,1]])
                            symop.append(tmp)
    return symop

####### WIP ########
################ 
# site symmetry
################
def site_symmetry(site,brv):
    """symmetry operators in the site symmetry group G.
    
    Args:
        site (numpy.ndarray):
            xyz coordinate of the site.
            The shape is (6,3).
            
    Returns:
        List of index of symmetry operators of the site symmetry group G (list):
            The symmetry operators leaves xyz identical.
    """
    
    symop=icosasymop()
    traop=translation(brv)
    
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

def coset(site,brv):
    """coset
    """
    symop=icosasymop()
    
    list1=site_symmetry(site,brv)
    
    # List of index of symmetry operators which are not in the G.
    tmp=range(len(symop))
    tmp=set(tmp)-set(list1)
    list2=list(tmp)
    
    list2_new=remove_overlaps(list2)
    
    if len(symop)==len(list1):
        list5=[0]
    else:
        # left coset decomposition:
        list4=[]
        for i2 in list2_new:
            list3=[]
            for i1 in list1:
                op1=np.dot(symop[i2],symop[i1])
                for i3,op in enumerate(symop):
                    if np.all(op1==op):
                        list3.append(i3)
                        break
                    else:
                        pass
            list4.append(list3)
        
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
            if len(symop)==len(list5)*len(list1):
                break
            else:
                pass
    return list5

def site_symmetry_and_coset(site,brv):
    #symmetry operators in the site symmetry group G and its left coset decomposition.
    #
    #Args:
    #    site (numpy.ndarray):
    #        xyz coordinate of the site.
    #        The shape is (6,3).
    #
    #Returns:
    #    List of index of symmetry operators of the site symmetry group G (list):
    #        The symmetry operators leaves xyz identical.
    #    
    #    List of index of symmetry operators in the left coset representatives of the poibt group G (list):
    #        The symmetry operators generates equivalent positions of the site xyz.
    
    symop=icosasymop()
    traop=translation(brv)
    
    # List of index of symmetry operators of the site symmetry group G.
    # The symmetry operators leaves xyz identical.
    list1=[]
    
    # List of index of symmetry operators which are not in the G.
    list2=[]
    
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
    
    if len(symop)==len(list1_new):
        list5=[0]
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
    #vec1=[]
    op=similarity(m)
    return dot_product_1(op,vt)
    
def similarity(m):
    """Similarity transformation of icosahedral QCs
    """
    m1=0.5*np.array([[1, 1, 1, 1, 1, 1],\
                    [ 1, 1, 1,-1,-1, 1],\
                    [ 1, 1, 1, 1,-1,-1],\
                    [ 1,-1, 1, 1, 1,-1],\
                    [ 1,-1,-1, 1, 1, 1],\
                    [ 1, 1,-1,-1, 1, 1]],dtype=np.int64)
    return matrixpow(m1.T,m)
    
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
    
    def generate_random_tetrahedron():
        return generate_random_vectors(4)
    
    cen0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    
    
    print("TEST: symop_vec()")
    symop=icosasymop()
    vt=generate_random_vector()
    counter=0
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
    