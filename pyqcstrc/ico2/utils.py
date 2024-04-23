#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
sys.path.append('.')
from math1 import (projection3,
                    sub_vectors,
                    add_vectors,
                    outer_product,
                    inner_product,
                    centroid,
                    add,
                    sub,
                    mul,
                    div)
from numericalc import numeric_value
import numpy as np

def shift_object(obj,shift):
    """shift an object
    """
    #vol0=obj_volume_6d(obj)
    obj_new=np.zeros(obj.shape,dtype=np.int64)
    i1=0
    for tetrahedron in obj:
        i2=0
        for vertex in tetrahedron:
            obj_new[i1][i2]=add_vectors(vertex,shift)
            i2+=1
        i1+=1
    #vol1=obj_volume_6d(obj_new)
    #if np.all(vol0==vol1):
    #    return obj_new
    #else:
    #    return 
    return obj_new

def obj_volume_6d(obj):
    w=np.array([0,0,1])
    if obj.ndim==4:
        for tetrahedron in obj:
            v=tetrahedron_volume_6d(tetrahedron)
            w=add(w,v)
        return w
    elif obj.ndim==5:
        for tset in obj:
            for tetrahedron in tset:
                v=tetrahedron_volume_6d(tetrahedron)
                w=add(w,v)
        return w
    else:
        print('object has an incorrect shape!')
        return 
    
def tetrahedron_volume_6d(tetrahedron):
    vts=np.zeros((4,3,3),dtype=np.int64)
    for i in range(4):
        vts[i]=projection3(tetrahedron[i])
    return tetrahedron_volume(vts)

def tetrahedron_volume(vts):
    # This function returns volume of a tetrahedron
    # input: vertex coordinates of the tetrahedron (x0,y0,z0),(x1,y1,z1),(x2,y2,z2),(x3,y3,z3) in TAU-style.
    v1=sub_vectors(vts[1],vts[0])
    v2=sub_vectors(vts[2],vts[0])
    v3=sub_vectors(vts[3],vts[0])
    
    v=outer_product(v1,v2)
    v=inner_product(v,v3)
    
    # avoid a negative value
    val=numeric_value(v)
    if val<0.0: # to avoid negative volume
        return mul(v,np.array([-1,0,6]))
    else:
        return mul(v,np.array([1,0,6]))




def remove_doubling(vst):
    """remove doubling 6d coordinates
    
    Parameters
    ----------
    obj: array
        set of 6-dimensional vectors in TAU-style
    
    Returns
    -------
    obj: array
        set of 6-dimensional vectors in TAU-style
    """
    ndim=vst.ndim
    if ndim==4:
        n1,n2,_,_=vst.shape
        num=n1*n2
        vst=vst.reshape(num,6,3)
    elif ndim==3:
        num,_,_=vst.shape
    else:
        print('ndim should be larger than 3.')
    return np.unique(vst,axis=0)

def remove_doubling_in_perp_space(vst):
    """ remove 6d coordinates which is doubled in Eperp.
    
    Parameters
    ----------
    obj: array
        set of 6-dimensional vectors in TAU-style
    
    Returns
    -------
    obj: array
        set of 6-dimensional vectors in TAU-style
    """
    ndim=vst.ndim
    if ndim==4:
        n1,n2,_,_=vst.shape
        num=n1*n2
        vst=vst.reshape(num,6,3)
    elif ndim==3:
        num,_,_=vst.shape
    
    # first run remove_doubling()
    vst=remove_doubling(vst)
    num=len(vst)
    
    # then, remove doubling in perp space.
    a=np.zeros((num,3,3),dtype=np.int64)
    for i in range(num):
        a[i]=projection3(vst[i])
    b=np.unique(a,return_index=True,axis=0)[1]
    num=len(b)
    a=np.zeros((num,6,3),dtype=np.int64)
    for i in range(num):
        a[i]=vst[b[i]]
    return a







########## WIP ##########

def generator_surface_1(obj):
    """
    # remove doubling surface in a set of tetrahedra in the OD (dim4)
    #
    """
    
    def get_tetrahedron_surface(tetrahedron):
        """
        get four triangles of tetrahedron.
        """
        # four triangles of tetrahedron: 1-2-3, 1-2-4, 1-3-4, 2-3-4
        comb=[\
        [0,1,2],\
        [0,1,3],\
        [0,2,3],\
        [1,2,3]] 
        #
        a=np.zeros((4,3,6,3),dtype=np.int64)
        i1=0
        for k in comb:
            i2=0
            for l in k:
                a[i1][i2]=tetrahedron[l]
                i2+=1
            i1+=1
        return a
    
    def equivalent_triangle_1(triangle1,triangle2):
        """Check whether triangle1 and triangle2 are equivalent or not.
        """
        a=np.vstack([triangle1,triangle2])
        a=remove_doubling_in_perp_space(a)
        if len(a)==3:
            return True # equivalent traiangle
        else:
            return False # not equivalent traiangles
    
    # (1) preparing a list of triangle surfaces without doubling (tmp2)
    #print('get_tetrahedron_surface() starts')
    n1,_,_,_=obj.shape
    triangles=np.zeros((n1,4,3,6,3),dtype=np.int64)
    i1=0
    for tetrahedron in obj:
        triangles[i1]=get_tetrahedron_surface(tetrahedron)
        i1+=1
    triangles=triangles.reshape(n1*4,3,6,3)
    #print('get_tetrahedron_surface() ends')
    #print('      number of triangle',len(triangles))
    if n1==1:
        return triangles
    else:
        # (2) 重複している三角形を探し、重複のない三角形（すなはちobject表面の三角形）のみを得る。
        # 三角形が重複していれば重心も同じことを利用する。
        #print(' search dounbling.....')
        
        a=np.zeros((len(triangles),6,3),dtype=np.int64)
        for i1 in range(len(a)):
            a[i1]=centroid(triangle[i1])
        b=np.unique(a,return_index=True,axis=0)[1]
        num=len(b)
        a=np.zeros((num,3,6,3),dtype=np.int64)
        for i1 in range(num):
            a[i1]=triangles[b[i1]]
        """
        lst=[triangles[0]]
        for i1 in range(1,len(triangles)):
            tr1=triangles[i1]
            counter=0
            for tr2 in lst:
                if equivalent_triangle_1(tr1,tr2): # equivalent
                    counter+=1
                    break
                else:
                    pass
            if counter==0:
                lst.append(tr1)
        return np.array(lst,dtype=np.int64)
        """
        return a
        
def generator_edge(obj):
    """
    generates edges
    
    Input
    triangles, np.array with a shape=(number_of_triangles,3,6,3)
    """
    
    def get_triangle_edge(triangle):
        """
        get four triangles of tetrahedron.
        """
        # three edges of triange: 0-1, 0-2, 1-2
        comb=[\
        [0,1],\
        [0,2],\
        [1,2]] 
        #
        # Four triangles the tetrahedron.
        a=np.zeros((3,2,6,3),dtype=np.int64)
        i1=0
        for k in comb:
            i2=0
            for l in k:
                a[i1][i2]=triangle[l]
                i2+=1
            i1+=1
        return a
    
    def equivalent_edge_1(edge1,edge2):
        """Check whether edge1 and edge2 are equivalent or not.
        """
        a=np.vstack([edge1,edge2])
        a=remove_doubling_in_perp_space(a)
        if len(a)==2:
            return True # equivalent
        else:
            return False # not equivalent
    
    # (1) preparing a list of edges without doubling
    n1,n2,_,_=obj.shape
    edges=np.zeros((n1,n2,2,6,3),dtype=np.int64)
    i1=0
    for triangle in obj:
        edges[i1]=get_triangle_edge(triangle)
        i1+=1
    edges=edges.reshape(n1*n2,2,6,3)
    if n1==1:
        return edges
    else:
        # (2) 重複している辺を探し、重複なしの辺（すなはちobject表面の辺）を得る。
        #print(' search dounbling.....')
        a=np.zeros((len(edges),6,3),dtype=np.int64)
        for i1 in range(len(a)):
            a[i1]=centroid(edges[i1])
        b=np.unique(a,return_index=True,axis=0)[1]
        num=len(b)
        a=np.zeros((num,2,6,3),dtype=np.int64)
        for i1 in range(num):
            a[i1]=edges[b[i1]]
        """
        lst=[edges[0]]
        for i1 in range(1,len(edges)):
            ed1=edges[i1]
            counter=0
            for ed2 in lst:
                if equivalent_edge_1(ed1,ed2): # equivalent
                    counter+=1
                    break
                else:
                    pass
            if counter==0:
                lst.append(ed1)
        return np.array(lst,dtype=np.int64)
        """
        return a
if __name__ == '__main__':
    
    # test
    
    import random
    
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
    
    #================
    # 重複のテスト
    #================
    nset=10
    vst=generate_random_vectors(nset)
    vst_d3=np.concatenate([vst,vst]) # doubling dim3 vectors
    vst_d4=np.stack([vst_d3,vst_d3]) # doubling dim4 vectors
    
    a=remove_doubling(vst_d4)
    if len(a)==nset:
        print('remove_doubling: pass')
    else:
        print('remove_doubling: error')
        
    a=remove_doubling_in_perp_space(vst_d4)
    if len(a)==nset:
        print('remove_doubling_in_perp_space: pass')
    else:
        print('remove_doubling_in_perp_space: error')
    
    #================
    # 面と辺のテスト
    #================
    tetrahedron=generate_random_tetrahedron()
    
    # doubled tetrahedon
    obj=np.stack([tetrahedron,tetrahedron]) # doubled tetrahedon
    generator_surface_1(obj)
    
    # a tetrahedon
    obj=tetrahedron
    surface=generator_surface_1(obj.reshape(1,4,6,3))
    generator_edge(surface)
    