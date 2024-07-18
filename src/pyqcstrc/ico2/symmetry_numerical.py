#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
from pyqcstrc.ico2.symmetry import (icosasymop,
                                    remove_overlaps,
                                    find_overlaps,
                                    similarity,
                                    )

EPS=1e-6
V0=np.array([0.,0.,0.,0.,0.,0.],dtype=np.float64)

def symop_vec(symop,vn,centre):
    """ Apply a symmetric operation on a vector around given centre.
    
    Parameters
    ----------
    symop: list
        list of symmetry operations
    vn: array
        6D cector
    centre: array
        symmetrical centre
    
    Returns
    -------
    vec: array
        vector
    """
    vn=vn-centre
    vn=np.dot(symop,vn)
    return vn+centre

def symop_vecs(symop,vns,centre):
    """ Apply a symmetric operation on set of vectors around given centre.
    
    """
    shape=tuple([len(symop)])
    out=np.zeros(shape+vns.shape,dtype=np.float64)
    for i1,op in enumerate(symop):
        for i2,vn in enumerate(vns):
            out[i1][i2]=symop_vec(op,vn,centre)
    return out

def symop_obj(symop,obj,centre):
    """ Apply a symmetric operation on an object around given centre.
    
    """
    ndim=obj.ndim
    if ndim==4:
        shape=tuple([len(symop)])
        out=np.zeros(shape+obj.shape,dtype=np.float64)
        for i1,op in enumerate(symop):
            for i2,vns in enumerate(obj):
                out[i1][i2]=symop_vecs(op,vns,centre)
        return out
    elif ndim==3:
        return symop_vecs(symop,obj,centre)
    elif ndim==2:
        return symop_vec(symop,obj,centre)
    else:
        print('object has an incorrect shape!')
        return 


def generator_obj_symmetric_vector_specific_symop(vn,centre,list_of_symmetry_operation_index):
    # using specific symmetry operations
    if vn.ndim==1:
        mop=icosasymop()
        symop=[]
        for i1 in list_of_symmetry_operation_index:
            symop.append(mop[i1])
        return symop_vec(symop,vn,centre)
    else:
        print('vn has an incorrect shape!')
        return



def translation(brv):
    """translational symmetry
    
    brv : bravais lattce p, i, f, s, c
            s : superlattice for decagonal quasicrystal
    """
    symop=[]
    symop.append(np.array([0,0,0,0,0,0],dtype=np.int64))
    
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
                            tmp=np.array([i1,i2,i3,i4,i5,i6])
                            symop.append(tmp)
    return symop

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
            #if length_numerical(tmp)<EPS:
            if np.sqrt(tmp[0]**2+tmp[1]**2+tmp[2]**2+tmp[3]**2+tmp[4]**2+tmp[5]**2)<EPS:
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
            site2=site1+top
            tmp=site-site2
            #if length_numerical(tmp)<EPS:
            if np.sqrt(tmp[0]**2+tmp[1]**2+tmp[2]**2+tmp[3]**2+tmp[4]**2+tmp[5]**2)<EPS:
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

############################
# Similarity transformation
############################
def similarity_obj(obj,m):
    """similarity transformation of an object
    """
    out=np.zeros(obj.shape,dtype=np.float64)
    for i1,od in enumerate(obj):
        out[i1]=similarity_triangle(od,m)
    return out

def similarity_vectors(vns,m):
    """similarity transformation of a set of vectors
    """
    out=np.zeros(vns.shape,dtype=np.float64)
    for i1,vn in enumerate(vns):
        out[i1]=similarity_vec(vn,m)
    return out

def similarity_vector(vn,m):
    """similarity transformation of a vector
    """
    op=similarity(m)
    return np.dot(op,vn)
    




