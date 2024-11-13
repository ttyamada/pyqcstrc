#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
import sys
import numpy as np
try:
    from pyqcstrc.ico2.strc import (strc,
                                    )
    import pyqcstrc.ico2.occupation_domain as od
    import pyqcstrc.ico2.symmetry as symmetry
    import pyqcstrc.ico2.math1 as math1
    import pyqcstrc.ico2.numericalc as numericalc
    import pyqcstrc.ico2.symmetry_numerical as symmetry_numerical
except ImportError:
    print('import error\n')

TAU=(1+np.sqrt(5))/2.0

# Predefined 6D coordinates in TAU-style
POS_V  = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
POS_C  = np.array([[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2]],dtype=np.int64)
POS_EC = np.array([[1,0,2],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)

V0 = np.array([ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])

if __name__ == "__main__":
    
    ############################
    # TEST symmetry operations
    ############################ 
    V0_6d=np.array([0.,0.,0.,0.,0.,0.],dtype=np.float64)
    V0_3d=np.array([0.,0.,0.],dtype=np.float64)
    
    vt=np.array([[1,0,1],[0,2,1],[3,0,1],[0,4,1],[5,0,1],[0,6,1]],dtype=np.int64) # general 6d vector
    #vt=np.array([[1,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64) # 5fold
    vn=numericalc.numerical_vector(vt)
    
    #lst=[0,1,5,10,20,60] # identity, c5, c2, c2, c3, inversion symmetry
    lst=[]
    for i in range(120):
        lst.append(i)
    
    print('Symmetry operations on 6D positional vector')
    vns1=symmetry_numerical.generator_obj_symmetric_vector_specific_symop_1(vn,V0_6d,lst,'axial')
    for i1,vn1 in enumerate(vns1):
        vne1=numericalc.projection_par_numerical(vn1)
        vne1=vne1/np.linalg.norm(vne1)
        print('%d %8.6f %8.6f %8.6f'%(i1,vne1[0],vne1[1],vne1[2]))
    
    print('Symmetry operations on 3D positional vector')
    vne=numericalc.projection_par_numerical(vn)
    vne=vne/np.linalg.norm(vne)
    vnes2=symmetry_numerical.generator_obj_symmetric_vector_specific_symop_1(vne,V0_3d,lst,'axial')
    for i1,vne2 in enumerate(vnes2):
        print('%d %8.6f %8.6f %8.6f'%(i1,vne2[0],vne2[1],vne2[2]))
    
    
    
    
    lst=[]
    for i in range(120):
        lst.append(i)
    
    lst=[0,108,22,92,119,16,14,111,98,28,102,5]
    #lst=[0,1,2,3,4,60,61,62,63,64]
    #lst=[0,5]
    #lst1=[0,1,2,3,4] # identity and c5 around [1,tau,0]
    #lst2=[5,6,7,8,9] # mirror and c5
    #lst=lst1+lst2
    print('Symmetry operation on 6D axial vector')
    print("TEST :generator_obj_symmetric_vector_specific_symop_1()")
    vn=np.array([1,0,0,0,0,0]) # 6d mu vector // 5-fold
    mus=symmetry_numerical.generator_obj_symmetric_vector_specific_symop_1(vn,V0_6d,lst,'axial')
    poss=symmetry_numerical.generator_obj_symmetric_vector_specific_symop_1(vn,V0_6d,lst,'normal')
    out=[]
    for i1 in range(len(lst)):
        mu=numericalc.projection_par_numerical(mus[i1])
        pos=numericalc.projection_par_numerical(poss[i1])
        print('%d %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f'%(i1,pos[0],pos[1],pos[2],mu[0],mu[1],mu[2]))
        out.append(['Yb',pos,i1,vn,mu,i1])
    od.write_vesta(out,path='.',basename='test_6d_axial_vec',color='k',select='mag',verbose=0)
    
    print('Symmetry operation on 3D axial vector')
    print("TEST :generator_obj_symmetric_vector_specific_symop_1()")
    vne=np.array([-1.,TAU,0.]) # 3d mu vector // 5-fold in Epar
    mus=symmetry_numerical.generator_obj_symmetric_vector_specific_symop_1(vne,V0_3d,lst,'axial')
    poss=symmetry_numerical.generator_obj_symmetric_vector_specific_symop_1(vne,V0_3d,lst,'normal')
    out=[]
    for i1 in range(len(lst)):
        mu=mus[i1]
        pos=poss[i1]
        print('%d %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f'%(i1,pos[0],pos[1],pos[2], mu[0],mu[1],mu[2]))
        out.append(['Yb',pos,i1,vn,mu,i1])
    od.write_vesta(out,path='.',basename='test_3d_axial_vec',color='k',select='mag',verbose=0)
    
    
    
    
    
    
    """
    op1=symmetry.icosasymop3_array('axial')
    op2=symmetry.icosasymop3_array('normal')
    
    # three vectors that defines asymmetric unit
    vt5=np.array([[1,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64) # 5f
    vt3=np.array([[1,0,1],[0,0,1],[-1,0,1],[0,0,1],[-1,0,1],[0,0,1]],dtype=np.int64) # 3f
    vt2=np.array([[1,0,1],[0,0,1],[0,0,1],[0,0,1],[-1,0,1],[0,0,1]],dtype=np.int64) # 2f
    tmp=math1.add_vectors(vt5,vt3)
    vt=math1.add_vectors(tmp,vt2)
    
    indx=symmetry.get_index_of_symmetry_operation_for_equivalent_vectors(vt)
    print(indx)
    
    vn=numericalc.numerical_vector(vt)
    vne=numericalc.projection_sets_par_numerical_normalized(vn)
    V2=np.array([0,0,0],dtype=np.float64)
    
    # symmetry operation on axial vectors
    vn1=symmetry_numerical.generator_obj_symmetric_vector_specific_symop_1(vne,V2,indx,'axial')
    
    # symmetry operation on vectors
    vn2=symmetry_numerical.generator_obj_symmetric_vector_specific_symop_1(vne,V2,indx,'normal')
    
    
    for i1 in range(len(vn1)):
        v=vn1[i1]
        w=vn2[i1]
        if np.allclose(v,w):
            print('%3d %8.5f %8.5f %8.5f | %8.5f %8.5f %8.5f (same)'%(i1,v[0],v[1],v[2],w[0],w[1],w[2]))
        else:
            print('%3d %8.5f %8.5f %8.5f | %8.5f %8.5f %8.5f (different)'%(i1,v[0],v[1],v[2],w[0],w[1],w[2]))
            print(op1[i1]) 
            print(op2[i1]) 
    """
        
    """
    xyzpath='../../../xyz/ico/kumazawa'
    # Kumazawa's OD for icosahedron shell
    od0=od.read_xyz(path=xyzpath,basename='strt_aysmmetric',  select='tetrahedron',verbose=0)
    od1=od.read_xyz(path=xyzpath,basename='R3R3R5',  select='tetrahedron',verbose=0)
    od2=od.read_xyz(path=xyzpath,basename='R3R3R5R5',select='tetrahedron',verbose=0)
    od3=od.read_xyz(path=xyzpath,basename='R3R5',    select='tetrahedron',verbose=0)
    od4=od.read_xyz(path=xyzpath,basename='R3R5R5',  select='tetrahedron',verbose=0)
    od5=od.read_xyz(path=xyzpath,basename='R3R5R5R5',select='tetrahedron',verbose=0)
    od6=od.read_xyz(path=xyzpath,basename='R5',      select='tetrahedron',verbose=0)
    od7=od.read_xyz(path=xyzpath,basename='R5R5',    select='tetrahedron',verbose=0)
    od8=od.read_xyz(path=xyzpath,basename='R5R5R5',  select='tetrahedron',verbose=0)
    
    objs = [od0,od1,od2,od3,od4,od5,od6,od7,od8]
    
    point = np.array([ -1.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    oshift = np.array([ 0.03, -0.03, -0.03, 0.00, -0.03, -0.02]) # inside asymmetric unit
    point = numericalc.projection3_numerical(point+oshift)
    
    idx=0
    symop = symmetry.icosasymop_array()
    for i1,op in enumerate(symop):
        counter=0
        for obj in objs:
            od = symmetry.symop_obj(op,obj,centre=POS_V)
            for tetrahedron in od:
                if numericalc.inside_outside_tetrahedron_tau_v2(point,tetrahedron):
                    counter+=1
                    break
        if counter!=0:
            idx=i1
            print(i1)
    
    x1=np.array([1, 0, 0, 0, 0, 0],dtype=np.float64) #5f
    x2=np.array([1, 0,-1, 0,-1, 0],dtype=np.float64) #3f
    x3=np.array([1, 0, 0, 0,-1, 0],dtype=np.float64) #2f
    
    y1=symmetry_numerical.symop_vec(symop[idx],x1,centre=V0)
    y2=symmetry_numerical.symop_vec(symop[idx],x2,centre=V0)
    y3=symmetry_numerical.symop_vec(symop[idx],x3,centre=V0)
    print(y1)
    print(y2)
    print(y3)
    """
    