#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import timeit
import os
import sys
import numpy as np
import random
sys.path.append('.')
import qcmath

TAU=(1+np.sqrt(5))/2.0

def simplification_special(obj,num,num_cycle,verbose_level):
    # 複雑に四面体分割されたObjectを単純化する
    # 一部分（ただし、凸多面体に限る）だけ単純化したい場合．
    # Parameters:
    #  obj: 単純化したいobject(dim4のnumpy.ndarray)
    #  num: 凸多面体を構成する四面体の配列インデックスをnumで指定(dim2のlist)
    #  num_cycle: qcmath.simplification_convex_polyhedronのサイクル
    #  verbose_level: qcmath.simplification_convex_polyhedronのverbose_level
    n1,n2,n3=qcmath.obj_volume_6d(obj)
    obj_new=np.array([0])
    a=[]
    len(obj)
    for i3 in range(len(obj)):
        counter=0
        for i1 in range(len(num)):
            for i2 in num[i1]:
                if i3==i2:
                    counter+=1
                    break
                else:
                    pass
        if counter==0:
            if len(obj_new)==1:
                obj_new=obj[i3].reshape(72)
            else:
                obj_new=np.append(obj_new,obj[i3])
        else:
            pass
    if obj_new.tolist!=[0]:
        obj_new=obj_new.reshape(len(obj_new)/72,4,6,3)
    else:
        pass
        
    for i1 in range(len(num)):
        tmp=np.array([0])
        for i2 in num[i1]:
            if len(tmp)==1:
                tmp=obj[i2].reshape(72)
            else:
                tmp=np.append(tmp,obj[i2])
        tmp=qcmath.simplification_convex_polyhedron(tmp.reshape(len(tmp)/72,4,6,3),num_cycle,verbose_level)
        if len(obj_new)==1:
            obj_new=tmp.reshape(len(tmp)*72)
        else:
            obj_new=np.append(obj_new,tmp)
    obj_new=obj_new.reshape(len(obj_new)/72,4,6,3)
    m1,m2,m3=qcmath.obj_volume_6d(obj_new)
    print(' volume of original obj  : %d %d %d (%10.8f)'%(n1,n2,n3,(n1+TAU*n2)/float(n3)))
    print(' volume of simplified obj: %d %d %d (%10.8f)'%(m1,m2,m3,(m1+TAU*m2)/float(m3)))
    if n1==m1 and n2==m2 and n3==m3:
        print(' succeed')
        return obj_new
    else:
        print(' fail, original obj returned.')
        return obj

def simplification_special_1(obj,num,num_cycle,verbose_level):
    # 複雑に四面体分割されたObjectを単純化する
    # 一部分（だけ単純化したい場合．
    # Parameters:
    #  obj: 単純化したいobject(dim4のnumpy.ndarray)
    #  num: 多面体を構成する四面体の配列インデックスをnumで指定(dim2のlist)
    #  num_cycle: qcmath.simplification_convex_polyhedronのサイクル
    #  verbose_level: qcmath.simplification_obj_edges()のverbose_level
    n1,n2,n3=qcmath.obj_volume_6d(obj)
    obj_new=np.array([0])
    a=[]
    len(obj)
    for i3 in range(len(obj)):
        counter=0
        for i1 in range(len(num)):
            for i2 in num[i1]:
                if i3==i2:
                    counter+=1
                    break
                else:
                    pass
        if counter==0:
            if len(obj_new)==1:
                obj_new=obj[i3].reshape(72)
            else:
                obj_new=np.append(obj_new,obj[i3])
        else:
            pass
    if obj_new.tolist!=[0]:
        obj_new=obj_new.reshape(len(obj_new)/72,4,6,3)
        #print len(obj_new)
    else:
        pass
        
    for i1 in range(len(num)):
        tmp=np.array([0])
        for i2 in num[i1]:
            if len(tmp)==1:
                tmp=obj[i2].reshape(72)
            else:
                tmp=np.append(tmp,obj[i2])
        #tmp=qcmath.simplification_obj_edges(tmp.reshape(len(tmp)/72,4,6,3),num_cycle,verbose_level-1)
        tmp=qcmath.simplification_obj_edges_1(tmp.reshape(len(tmp)/72,4,6,3),num_cycle,verbose_level-1)
        if len(obj_new)==1:
            obj_new=tmp.reshape(len(tmp)*72)
        else:
            obj_new=np.append(obj_new,tmp)
    obj_new=obj_new.reshape(len(obj_new)/72,4,6,3)
    m1,m2,m3=qcmath.obj_volume_6d(obj_new)
    if verbose_level>1:
        print(' volume of original obj  : %d %d %d (%10.8f)'%(n1,n2,n3,(n1+TAU*n2)/float(n3)))
        print(' volume of simplified obj: %d %d %d (%10.8f)'%(m1,m2,m3,(m1+TAU*m2)/float(m3)))
    else:
        pass
    if n1==m1 and n2==m2 and n3==m3:
        print(' succeed')
        return obj_new
    else:
        print(' fail, original obj returned.')
        return obj

def simplification_rondom(obj,num_cycle,combination_num,verbose_level):
    
    if verbose_level>0:
        print('  pod.simplification_rondom()')
    else:
        pass
        
    n1,n2,n3=qcmath.obj_volume_6d(obj)
    obj_new=np.array([0])
    obj_tmp=obj
    for i in range(num_cycle):
        print('--------------')
        print('     %d-cycle'%(i))
        print('--------------')
        num=random.sample(list(range(len(obj_tmp))),combination_num)
        print(num)
        num=[num]
        obj_new=simplification_special_1(obj_tmp,num,2,verbose_level-1)
        if len(obj_new)<len(obj_tmp):
            obj_tmp=obj_new
            if verbose_level>1:
                print('  reduced')
            else:
                pass
        else:
            pass
    
    m1,m2,m3=qcmath.obj_volume_6d(obj_new)
    print(' volume of original obj  : %d %d %d (%10.8f)'%(n1,n2,n3,(n1+TAU*n2)/float(n3)))
    print(' volume of simplified obj: %d %d %d (%10.8f)'%(m1,m2,m3,(m1+TAU*m2)/float(m3)))
    if n1==m1 and n2==m2 and n3==m3:
        print(' succeed')
        return obj_new
    else:
        print(' fail, original obj returned.')
        return obj

def simplification_special_manual(obj,num,coordinates,num_cycle,verbose_level):
    # 複雑に四面体分割されたObjectを単純化する
    # 一部分（ただし、凸多面体に限る）だけ単純化したい場合．
    # この一部分はnumで指定
    # その頂点座標をcoordinatesで与える
    # Parameters:
    #  obj: 単純化したいobject(dim4のnumpy.ndarray)
    #  num: 凸多面体を構成する四面体の配列インデックスをnumで指定(dim2のlist)
    #  coordinates: 単純化したい部分の頂点（四面体分割に使う. dim3のnumpy.ndarrayを含むリスト）
    #  num_cycle: qcmath.simplification_convex_polyhedronのサイクル
    #  verbose_level: qcmath.simplification_convex_polyhedronのverbose_level
    n1,n2,n3=qcmath.obj_volume_6d(obj)
    obj_new=np.array([0])
    a=[]
    len(obj)
    for i3 in range(len(obj)):
        counter=0
        for i1 in range(len(num)):
            for i2 in num[i1]:
                if i3==i2:
                    counter+=1
                    break
                else:
                    pass
        if counter==0:
            if len(obj_new)==1:
                obj_new=obj[i3].reshape(72)
            else:
                obj_new=np.append(obj_new,obj[i3])
        else:
            pass
    if obj_new.tolist!=[0]:
        obj_new=obj_new.reshape(len(obj_new)/72,4,6,3)
        #print len(obj_new)
    else:
        pass
        
    for i1 in range(len(coordinates)):
        tmp4=qcmath.tetrahedralization_points(coordinates[i1])
        if len(obj_new)==1:
            obj_new=tmp4.reshape(len(tmp4)*72)
        else:
            obj_new=np.append(obj_new,tmp4)
    obj_new=obj_new.reshape(len(obj_new)/72,4,6,3)
    m1,m2,m3=qcmath.obj_volume_6d(obj_new)
    print(' volume of original obj  : %d %d %d (%10.8f)'%(n1,n2,n3,(n1+TAU*n2)/float(n3)))
    print(' volume of simplified obj: %d %d %d (%10.8f)'%(m1,m2,m3,(m1+TAU*m2)/float(m3)))
    if n1==m1 and n2==m2 and n3==m3:
        print(' succeed')
        return obj_new
    else:
        print(' fail, original obj returned.')
        return obj

def simplification_special_manual_2(obj,num,obj_partial):
    # 複雑に四面体分割されたObjectを単純化する
    # 四面分割を単純化したい部分が凸多面体でない場合、手で新たに四面体分割したobjを与える．
    # 一部分はnumで指定
    # 新しく四面体分割したobjを与える．
    # Parameters:
    #  obj: 単純化したいobject(dim4のnumpy.ndarray)
    #  num: 凸多面体を構成する四面体の配列インデックスをnumで指定(dim2のlist)
    #  obj_partial: 手で四面体分割したobj（四面体分割に使う. dim4のnumpy.ndarray）
    n1,n2,n3=qcmath.obj_volume_6d(obj)
    obj_new=np.array([0])
    a=[]
    len(obj)
    for i3 in range(len(obj)):
        counter=0
        for i1 in range(len(num)):
            for i2 in num[i1]:
                if i3==i2:
                    counter+=1
                    break
                else:
                    pass
        if counter==0:
            if len(obj_new)==1:
                obj_new=obj[i3].reshape(72)
            else:
                obj_new=np.append(obj_new,obj[i3])
        else:
            pass
    if obj_new.tolist!=[0]:
        obj_new=obj_new.reshape(len(obj_new)/72,4,6,3)
        #print len(obj_new)
    else:
        pass
        
    if len(obj_new)==1:
        obj_new=obj_partial.reshape(len(obj_partial)*72)
    else:
        obj_new=np.append(obj_new,obj_partial)
    obj_new=obj_new.reshape(len(obj_new)/72,4,6,3)
    m1,m2,m3=qcmath.obj_volume_6d(obj_new)
    print(' volume of original obj  : %d %d %d (%10.8f)'%(n1,n2,n3,(n1+TAU*n2)/float(n3)))
    print(' volume of simplified obj: %d %d %d (%10.8f)'%(m1,m2,m3,(m1+TAU*m2)/float(m3)))
    if n1==m1 and n2==m2 and n3==m3:
        print(' succeed')
        return obj_new
    else:
        print(' fail, original obj returned.')
        return obj

def position(vec1,vec2):
    # parameters
    # vec1: 6d index of a 6d vector 
    # vec2: 6d index of a 6d vector which is projected onto Eperp
    a=[]
    vec3=qcmath.projection_perp(vec2[0],vec2[1],vec2[2],vec2[3],vec2[4],vec2[5])
    for i in range(6):
        a1,a2,a3=qcmath.add(vec1[i][0],vec1[i][1],vec1[i][2],vec3[i][0],vec3[i][1],vec3[i][2])
        a.append(a1)
        a.append(a2)
        a.append(a3)
    return np.array(a).reshape(6,3)

def subdivision(tetrahedron, points):
    # 4面体の内部に新たな点を加え4面体分割する.
    # parameters:
    #  tetrahedron (dim3)
    #  points (dim3)
    # return:
    #  subdivided tetrahedron (dim4)
    return qcmath.tetrahedralization(tetrahedron,points)

class Intersection_wip(object):
    #      POD1 and      POD2 : obj_common
    #      POD1 and Not POD2 : obj_a (point_a)
    # Not POD1 and      POD2 : obj_b (point_b)
    def __init__(self,pod1,pod2,path,option,name):
        self._pod1=pod1
        self._pod2=pod2
        self._path=path
        self._option=option # (option=0) simplification, (option!=0) no simplification in intersection_using_tetrahedron_3()
        self._name=name
        if os.path.exists(self._path) == False:
             os.makedirs(self._path)
             #print 'No such directory: %s'%(self._path)
        else:
             pass
    def using_surface(self):
        # This is very simple but work correctly only when each subdivided 
        # three ODs (i.e part part, ODA and ODB) are be able to define as 
        # a series of tetrahedra.
        pod1_surface=qcmath.generator_surface(self._pod1)
        pod1_edge=qcmath.generator_edge(pod1_surface)
        pod2_surface=qcmath.generator_surface(self._pod2)
        pod2_edge=qcmath.generator_edge(pod2_surface)
        # Generating two obj in input
        print('INPUTed two PODs')
        file_tmp='%s/%s_pod_1.xyz'%(self._path,self._name)
        qcmath.generator_xyz_dim4(pod1_surface,file_tmp)
        print(' 1st POD saved as %s/%s_pod_1.xyz'%(self._path,self._name))
        file_tmp='%s/pod_2.xyz'%(self._path)
        qcmath.generator_xyz_dim4(pod2_surface,file_tmp)
        print(' 2nd POD saved as %s/%s_pod_2.xyz'%(self._path,self._name))
        point_common,point_a,point_b=qcmath.intersection_using_surface(pod1_surface,\
                                                                                            pod2_surface,\
                                                                                            pod1_edge,\
                                                                                            pod2_edge,\
                                                                                            self._pod1,\
                                                                                            self._pod2,\
                                                                                            self._path)
        if point_common.tolist()!=[[[[0]]]]:
            print('Generating .XYZ files in %s'%(self._path))
            print('  %s/%s_point_a.xyz'%(self._path,self._name))
            print('  %s/%s_point_b.xyz'%(self._path,self._name))
            print('  %s/%s_point_common.xyz'%(self._path,self._name))
            file_tmp='%s/%s_point_a.xyz'%(self._path,self._name)
            qcmath.generator_xyz_dim3(point_a,file_tmp)
            file_tmp='%s/%s_point_b.xyz'%(self._path,self._name)
            qcmath.generator_xyz_dim3(point_b,file_tmp)
            file_tmp='%s/%s_point_common.xyz'%(self._path,self._name)
            qcmath.generator_xyz_dim3(point_common,file_tmp)
        else:
            print('No intersection')
            print('Generating .XYZ files in %s'%(self._path))
            print('  %s/%s_point_a.xyz'%(self._path,self._name))
            print('  %s/%s_point_b.xyz'%(self._path,self._name))
            file_tmp='%s/%s_point_a.xyz'%(self._path,self._name)
            qcmath.generator_xyz_dim3(point_a,file_tmp)
            file_tmp='%s/%s_point_b.xyz'%(self._path,self._name)
            qcmath.generator_xyz_dim3(point_b,file_tmp)
        #return point_common,point_a,point_b
        return 0

    def using_tetrahedron(self):
        #verbose=0
        verbose=1
        #verbose=2
        print('Boolean operations (1) POD1 and POD2 (2) POD1 not POD2.')
        #print 'This performs three boolean operations (1) pod1 AND pod2, (2) pod1 NOT pod2, (3) pod2 NOT pod1'
        print(' 1st POD (POD1):')
        v1b,v2b,v3b=qcmath.obj_volume_6d(self._pod1)
        print('  volume = %d %d %d (%8.6f)'%(v1b,v2b,v3b,(v1b+TAU*v2b)/float(v3b)))
        file_tmp='%s/%s_pod_1'%(self._path,self._name)
        qcmath.generator_xyz_dim4_tetrahedron(self._pod1,file_tmp,0)
        #export_vesta_file(self._pod1,self._path,self._name+'_pod_1',1)
        #qcmath.generator_xyz_dim4_tetrahedron(self._pod1,file_tmp,1)
        print('  saved as %s/%s_pod_1.xyz'%(self._path,self._name))
        print(' 2nd pod (POD2):')
        v1a,v2a,v3a=qcmath.obj_volume_6d(self._pod2)
        print('  volume = %d %d %d (%8.6f)'%(v1a,v2a,v3a,(v1a+TAU*v2a)/float(v3a)))
        file_tmp='%s/%s_pod_2'%(self._path,self._name)
        qcmath.generator_xyz_dim4_tetrahedron(self._pod2,file_tmp,0)
        #export_vesta_file(self._pod2,self._path,self._name+'_pod_2',1)
        print('  saved as %s/%s_pod_2.xyz'%(self._path,self._name))
        #-----------------------------------------
        # (1) POD1 and POD2 : obj_common
        #-----------------------------------------
        print(' (1) POD1 and POD2')
        #pod_common=qcmath.intersection_using_tetrahedron(self._pod1,self._pod2,self._path)
        #pod_common=qcmath.intersection_using_tetrahedron_2(self._pod1,self._pod2,self._option,verbose,self._path)
        #pod_common=qcmath.intersection_using_tetrahedron_3(self._pod1,self._pod2,self._option,verbose,self._path) ###  use this
        pod_common=qcmath.intersection_using_tetrahedron_4(self._pod1,self._pod2,self._option,verbose,self._path) ###  teset
        if pod_common.tolist()!=[[[[0]]]]:
            print('    numbre of tetrahedron %d'%(len(pod_common)))
            file_tmp='%s/%s_pod_common'%(self._path,self._name)
            qcmath.generator_xyz_dim4_tetrahedron(pod_common,file_tmp,0)
            #qcmath.generator_xyz_dim4_tetrahedron(pod_common,file_tmp,1)
            print('    saved as %s/%s_pod_common.xyz'%(self._path,self._name))
        """
        #
        pod_common_simple=qcmath.simplification(pod_common,verbose,self._path)
        #
        if pod_common_simple.tolist()!=[[[[0]]]]:
            print '  (pod_common_simple) POD1 and POD2 : %d x tetrahedron'%(len(pod_common_simple))
            file_tmp='%s/%s_pod_common_simple.xyz'%(self._path,self._name)
            qcmath.generator_xyz_dim4_tetrahedron(pod_common,file_tmp)
            print '  saved as %s/%s_pod_common_simple.xyz'%(self._path,self._name)
        """
        #else:
        #    print '  (pod_common) POD1 AND POD2 : empty'
        #-----------------------------------------
        # (2) POD1 NOT POD2 : obj_a
        #-----------------------------------------
        if pod_common.tolist()!=[[[[0]]]]:
            print(' (2) POD1 not POD2')
            v1c,v2c,v3c=qcmath.obj_volume_6d(pod_common)
            #pod_a=qcmath.object_subtraction_new(self._pod1,pod_common,self._pod2,self._path)
            #pod_a=qcmath.object_subtraction_new(self._pod1,pod_common_simple,self._pod2,self._path)
            #pod_a=qcmath.object_subtraction_3(self._pod1,pod_common)
            #pod_a=qcmath.object_not_object(self._pod1,self._pod2)
            #
            pod_a=qcmath.object_subtraction_2(self._pod1,pod_common,verbose) ####  use this (i=0,2,3,5,...9)
            #pod_a=qcmath.object_subtraction_dev(self._pod1,pod_common,self._pod2,verbose) ####  use this (i=10), slow
            #pod_a=qcmath.object_subtraction_dev1(self._pod1,pod_common,self._pod2,verbose) ####  use this (i=1), slow
            #pod_a=qcmath.object_subtraction_dev2(self._pod1,pod_common,self._pod2,verbose) ####  use this (i=1,4), slow
            #
            #
            if pod_a.tolist()!=[[[[0]]]]:
                 v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
            else:
                v1a,v2a,v3a=0,0,1
            v1d,v2d,v3d=qcmath.add(v1c,v2c,v3c,v1a,v2a,v3a)
            """
            if pod_a.tolist()!=[[[[0]]]]:
                print '    numbre of tetrahedron %d'%(len(pod_a))
                file_tmp='%s/%s_pod_a'%(self._path,self._name)
                qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,0)
                qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,1)
                print '    saved as %s/%s_pod_a.xyz'%(self._path,self._name)
                v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
            else:
                v1a,v2a,v3a=0,0,1
                print '    empty'
            """
            #"""
            #
            # Try object_subtraction_2()
            #
            if v1d==v1b and v2d==v2b and v3d==v3b:
                if pod_a.tolist()!=[[[[0]]]]:
                    print('    numbre of tetrahedron %d'%(len(pod_a)))
                    file_tmp='%s/%s_pod_a'%(self._path,self._name)
                    qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,0)
                    #qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,1)
                    print('    saved as %s/%s_pod_a.xyz'%(self._path,self._name))
                    v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
                else:
                    v1a,v2a,v3a=0,0,1
                    print('    empty')
            else:
                print('    fail')
                #
                # Try object_subtraction_dev2()
                #
                #pod_a=qcmath.object_subtraction_new(self._pod1,pod_common,self._pod2,self._path)
                #pod_a=qcmath.object_subtraction_1(self._pod1,pod_common)
                #pod_a=qcmath.object_subtraction_3(self._pod1,pod_common)
                #pod_a=qcmath.object_subtraction_dev1(self._pod1,pod_common,self._pod2,verbose) ####  use this (i=1), slow
                pod_a=qcmath.object_subtraction_dev2(self._pod1,pod_common,self._pod2,verbose) ####  use this (i=1,4), slow
                if pod_a.tolist()!=[[[[0]]]]:
                     v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
                else:
                    v1a,v2a,v3a=0,0,1
                v1d,v2d,v3d=qcmath.add(v1c,v2c,v3c,v1a,v2a,v3a)
                if v1d==v1b and v2d==v2b and v3d==v3b:
                    if pod_a.tolist()!=[[[[0]]]]:
                        print('    numbre of tetrahedron %d'%(len(pod_a)))
                        file_tmp='%s/%s_pod_a'%(self._path,self._name)
                        qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,0)
                        #qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,1)
                        print('    saved as %s/%s_pod_a.xyz'%(self._path,self._name))
                        v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
                    else:
                        v1a,v2a,v3a=0,0,1
                        print('    empty')
                else:
                    print('    fail')
                    #
                    # Try object_subtraction_dev()
                    #
                    pod_a=qcmath.object_subtraction_dev(self._pod1,pod_common,self._pod2,verbose) ####  use this (i=10), slow
                    if pod_a.tolist()!=[[[[0]]]]:
                         v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
                    else:
                        v1a,v2a,v3a=0,0,1
                    v1d,v2d,v3d=qcmath.add(v1c,v2c,v3c,v1a,v2a,v3a)
                    if v1d==v1b and v2d==v2b and v3d==v3b:
                        if pod_a.tolist()!=[[[[0]]]]:
                            print('    numbre of tetrahedron %d'%(len(pod_a)))
                            file_tmp='%s/%s_pod_a'%(self._path,self._name)
                            qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,0)
                            #qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,1)
                            print('    saved as %s/%s_pod_a.xyz'%(self._path,self._name))
                            v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
                        else:
                            v1a,v2a,v3a=0,0,1
                            print('    empty')
                    else:
                        print('    fail')
            #"""
            
            
            
            #
            #pod_a_simple=qcmath.simplification(pod_a,verbose,self._path)
            #
            #if pod_a.tolist()!=[[[[0]]]]:
            #    print '  (pod_a) POD1 NOT POD2 : %d x tetrahedron'%(len(pod_a))
            #    file_tmp='%s/%s_pod_a_simple.xyz'%(self._path,self._name)
            #    qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp)
            #    print '  saved as %s/%s_pod_a_simple.xyz'%(self._path,self._name)
            #else:
            #    print '    (pod_a) POD1 NOT POD2 : empty'
        else:
            v1c,v2c,v3c=0,0,1
            print('    empty')
            pod_a=self._pod1
            v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
        print(' (A) POD1 and POD2, volume = %d %d %d (%8.6f)'%(v1c,v2c,v3c,(v1c+TAU*v2c)/float(v3c)))
        print(' (B) POD1 not POD2, volume = %d %d %d (%8.6f)'%(v1a,v2a,v3a,(v1a+TAU*v2a)/float(v3a)))
        v1a,v2a,v3a=qcmath.add(v1a,v2a,v3a,v1c,v2c,v3c)
        print(' (C) SUM (A+B)     , volume = %d %d %d (%8.6f)'%(v1a,v2a,v3a,(v1a+TAU*v2a)/float(v3a)))
        """
        #-----------------------------------------
        # (3) POD2 NOT POD2 : obj_b
        #-----------------------------------------
        pod_b=qcmath.object_subtraction_new(self._pod2,pod_common,self._pod1,self._path)
        if pod_b.tolist()!=[[[[0]]]]:
            print '  (pod_b) POD2 NOT POD1 : %d x tetrahedron'%(len(pod_b))
            file_tmp='%s/%s_pod_b.xyz'%(self._path,self._name)
            qcmath.generator_xyz_dim4_tetrahedron(pod_b,file_tmp)
            print '  saved as %s/%s_pod_b.xyz'%(self._path,self._name)
        else:
            print '  (pod_b) POD2 NOT POD1 : empty'
        """
        #return pod_common,pod_a,pod_b
        return pod_common,pod_a
        #return pod_common_simple, pod_a_simple
    def using_tetrahedron_common(self):
        #verbose=0
        verbose=1
        #verbose=2
        print('Boolean operation (1) pod1 AND pod2.')
        #print 'This performs three boolean operations (1) pod1 AND pod2, (2) pod1 NOT pod2, (3) pod2 NOT pod1'
        print('INPUTed two PODs')
        file_tmp='%s/%s_pod_1'%(self._path,self._name)
        qcmath.generator_xyz_dim4_tetrahedron(self._pod1,file_tmp,0)
        print(' 1st POD saved as %s/%s_pod_1.xyz'%(self._path,self._name))
        file_tmp='%s/%s_pod_2'%(self._path,self._name)
        qcmath.generator_xyz_dim4_tetrahedron(self._pod2,file_tmp,0)
        print(' 2nd POD saved as %s/%s_pod_2.xyz'%(self._path,self._name))
        #-----------------------------------------
        # (1) POD1 and POD2 : obj_common
        #-----------------------------------------
        #pod_common=qcmath.intersection_using_tetrahedron_2(self._pod1,self._pod2,self._option,self._path)
        #pod_common=qcmath.intersection_using_tetrahedron_3(self._pod1,self._pod2,self._option,verbose,self._path) ###  use this
        pod_common=qcmath.intersection_using_tetrahedron_4(self._pod1,self._pod2,self._option,verbose,1) ###  teset
        #
        if pod_common.tolist()!=[[[[0]]]]:
            #v1a,v2a,v3a=qcmath.obj_volume_6d(pod_common)
            print('  (pod_common) POD1 and POD2 : %d x tetrahedron'%(len(pod_common)))
            file_tmp='%s/%s_pod_common'%(self._path,self._name)
            qcmath.generator_xyz_dim4_tetrahedron(pod_common,file_tmp,0)
            print('  saved as %s/%s_pod_common.xyz'%(self._path,self._name))
            """
            #
            pod_common_simple=qcmath.simplification(pod_common,self._path)
            #
            print '  (pod_common_simple) POD1 and POD2 : %d x tetrahedron'%(len(pod_common_simple))
            file_tmp='%s/%s_pod_common_simple.xyz'%(self._path,self._name)
            qcmath.generator_xyz_dim4_tetrahedron(pod_common,file_tmp)
            print '  saved as %s/%s_pod_common_simple.xyz'%(self._path,self._name)
            return pod_common_simple
            """
            return pod_common
        else:
            print('  (pod_common) POD1 AND POD2 : empty')
            v1c,v2c,v3c=0,0,1
            #print pod_common
            return np.array([0]).reshape(1,1,1,1)
    def using_tetrahedron_all(self):
        #verbose=0
        verbose=1
        #verbose=2
        print('Boolean operations (1) POD1 and POD2, (2) POD1 not POD2, (3) POD2 not POD1')
        #print 'This performs three boolean operations (1) pod1 AND pod2, (2) pod1 NOT pod2, (3) pod2 NOT pod1'
        print(' 1st POD (POD1):')
        v1b,v2b,v3b=qcmath.obj_volume_6d(self._pod1)
        print('  volume = %d %d %d (%8.6f)'%(v1b,v2b,v3b,(v1b+TAU*v2b)/float(v3b)))
        file_tmp='%s/%s_pod_1'%(self._path,self._name)
        qcmath.generator_xyz_dim4_tetrahedron(self._pod1,file_tmp,0)
        #qcmath.generator_xyz_dim4_tetrahedron(self._pod1,file_tmp,1)
        print('  saved as %s/%s_pod_1.xyz'%(self._path,self._name))
        print(' 2nd pod (POD2):')
        v1a,v2a,v3a=qcmath.obj_volume_6d(self._pod2)
        print('  volume = %d %d %d (%8.6f)'%(v1a,v2a,v3a,(v1a+TAU*v2a)/float(v3a)))
        file_tmp='%s/%s_pod_2'%(self._path,self._name)
        qcmath.generator_xyz_dim4_tetrahedron(self._pod2,file_tmp,0)
        print('  saved as %s/%s_pod_2.xyz'%(self._path,self._name))
        #-----------------------------------------
        # (1) POD1 and POD2 : obj_common
        #-----------------------------------------
        print(' (1) POD1 and POD2')
        #pod_common=qcmath.intersection_using_tetrahedron(self._pod1,self._pod2,self._path)
        #pod_common=qcmath.intersection_using_tetrahedron_2(self._pod1,self._pod2,self._option,verbose,self._path)
        #pod_common=qcmath.intersection_using_tetrahedron_3(self._pod1,self._pod2,self._option,verbose,self._path) ###  use this
        pod_common=qcmath.intersection_using_tetrahedron_4(self._pod1,self._pod2,self._option,verbose,self._path) ###  teset
        if pod_common.tolist()!=[[[[0]]]]:
            print('    numbre of tetrahedron %d'%(len(pod_common)))
            file_tmp='%s/%s_pod_common'%(self._path,self._name)
            qcmath.generator_xyz_dim4_tetrahedron(pod_common,file_tmp,0)
            #qcmath.generator_xyz_dim4_tetrahedron(pod_common,file_tmp,1)
            print('    saved as %s/%s_pod_common.xyz'%(self._path,self._name))
        """
        #
        pod_common_simple=qcmath.simplification(pod_common,verbose,self._path)
        #
        if pod_common_simple.tolist()!=[[[[0]]]]:
            print '  (pod_common_simple) POD1 and POD2 : %d x tetrahedron'%(len(pod_common_simple))
            file_tmp='%s/%s_pod_common_simple.xyz'%(self._path,self._name)
            qcmath.generator_xyz_dim4_tetrahedron(pod_common,file_tmp)
            print '  saved as %s/%s_pod_common_simple.xyz'%(self._path,self._name)
        """
        #else:
        #    print '  (pod_common) POD1 AND POD2 : empty'
        #-----------------------------------------
        # (2) POD1 NOT POD2 : obj_a
        #-----------------------------------------
        if pod_common.tolist()!=[[[[0]]]]:
            print(' (2) POD1 not POD2')
            v1c,v2c,v3c=qcmath.obj_volume_6d(pod_common)
            #pod_a=qcmath.object_subtraction_new(self._pod1,pod_common,self._pod2,self._path)
            #pod_a=qcmath.object_subtraction_new(self._pod1,pod_common_simple,self._pod2,self._path)
            #pod_a=qcmath.object_subtraction_3(self._pod1,pod_common)
            #pod_a=qcmath.object_not_object(self._pod1,self._pod2)
            #
            pod_a=qcmath.object_subtraction_2(self._pod1,pod_common,verbose) ####  use this (i=0,2,3,5,...9)
            #pod_a=qcmath.object_subtraction_dev(self._pod1,pod_common,self._pod2,verbose) ####  use this (i=10), slow
            #pod_a=qcmath.object_subtraction_dev1(self._pod1,pod_common,self._pod2,verbose) ####  use this (i=1), slow
            #pod_a=qcmath.object_subtraction_dev2(self._pod1,pod_common,self._pod2,verbose) ####  use this (i=1,4), slow
            #
            #
            if pod_a.tolist()!=[[[[0]]]]:
                 v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
            else:
                v1a,v2a,v3a=0,0,1
            v1d,v2d,v3d=qcmath.add(v1c,v2c,v3c,v1a,v2a,v3a)
            """
            if pod_a.tolist()!=[[[[0]]]]:
                print '    numbre of tetrahedron %d'%(len(pod_a))
                file_tmp='%s/%s_pod_a'%(self._path,self._name)
                qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,0)
                qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,1)
                print '    saved as %s/%s_pod_a.xyz'%(self._path,self._name)
                v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
            else:
                v1a,v2a,v3a=0,0,1
                print '    empty'
            """
            #"""
            #
            # Try object_subtraction_2()
            #
            if v1d==v1b and v2d==v2b and v3d==v3b:
                if pod_a.tolist()!=[[[[0]]]]:
                    print('    numbre of tetrahedron %d'%(len(pod_a)))
                    file_tmp='%s/%s_pod_a'%(self._path,self._name)
                    qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,0)
                    #qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,1)
                    print('    saved as %s/%s_pod_a.xyz'%(self._path,self._name))
                    v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
                else:
                    v1a,v2a,v3a=0,0,1
                    print('    empty')
            else:
                print('    fail')
                #
                # Try object_subtraction_dev2()
                #
                #pod_a=qcmath.object_subtraction_new(self._pod1,pod_common,self._pod2,self._path)
                #pod_a=qcmath.object_subtraction_1(self._pod1,pod_common)
                #pod_a=qcmath.object_subtraction_3(self._pod1,pod_common)
                #pod_a=qcmath.object_subtraction_dev1(self._pod1,pod_common,self._pod2,verbose) ####  use this (i=1), slow
                pod_a=qcmath.object_subtraction_dev2(self._pod1,pod_common,self._pod2,verbose) ####  use this (i=1,4), slow
                if pod_a.tolist()!=[[[[0]]]]:
                     v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
                else:
                    v1a,v2a,v3a=0,0,1
                v1d,v2d,v3d=qcmath.add(v1c,v2c,v3c,v1a,v2a,v3a)
                if v1d==v1b and v2d==v2b and v3d==v3b:
                    if pod_a.tolist()!=[[[[0]]]]:
                        print('    numbre of tetrahedron %d'%(len(pod_a)))
                        file_tmp='%s/%s_pod_a'%(self._path,self._name)
                        qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,0)
                        #qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,1)
                        print('    saved as %s/%s_pod_a.xyz'%(self._path,self._name))
                        v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
                    else:
                        v1a,v2a,v3a=0,0,1
                        print('    empty')
                else:
                    print('    fail')
                    #
                    # Try object_subtraction_dev()
                    #
                    pod_a=qcmath.object_subtraction_dev(self._pod1,pod_common,self._pod2,verbose) ####  use this (i=10), slow
                    if pod_a.tolist()!=[[[[0]]]]:
                         v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
                    else:
                        v1a,v2a,v3a=0,0,1
                    v1d,v2d,v3d=qcmath.add(v1c,v2c,v3c,v1a,v2a,v3a)
                    if v1d==v1b and v2d==v2b and v3d==v3b:
                        if pod_a.tolist()!=[[[[0]]]]:
                            print('    numbre of tetrahedron %d'%(len(pod_a)))
                            file_tmp='%s/%s_pod_a'%(self._path,self._name)
                            qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,0)
                            #qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,1)
                            print('    saved as %s/%s_pod_a.xyz'%(self._path,self._name))
                            v1a,v2a,v3a=qcmath.obj_volume_6d(pod_a)
                        else:
                            v1a,v2a,v3a=0,0,1
                            print('    empty')
                    else:
                        print('    fail')

        #-----------------------------------------
        # (3) POD2 NOT POD1 : obj_b
        #-----------------------------------------
        if pod_common.tolist()!=[[[[0]]]]:
            print(' (3) POD2 not POD1')
            v1c,v2c,v3c=qcmath.obj_volume_6d(pod_common)
            #
            pod_b=qcmath.object_subtraction_2(self._pod2,pod_common,verbose) ####  use this (i=0,2,3,5,...9)
            #
            if pod_b.tolist()!=[[[[0]]]]:
                 v1a,v2a,v3a=qcmath.obj_volume_6d(pod_b)
            else:
                v1a,v2a,v3a=0,0,1
            v1d,v2d,v3d=qcmath.add(v1c,v2c,v3c,v1a,v2a,v3a)
            #"""
            #
            # Try object_subtraction_2()
            #
            if v1d==v1b and v2d==v2b and v3d==v3b:
                if pod_b.tolist()!=[[[[0]]]]:
                    print('    numbre of tetrahedron %d'%(len(pod_b)))
                    file_tmp='%s/%s_pod_b'%(self._path,self._name)
                    qcmath.generator_xyz_dim4_tetrahedron(pod_b,file_tmp,0)
                    #qcmath.generator_xyz_dim4_tetrahedron(pod_b,file_tmp,1)
                    print('    saved as %s/%s_pod_b.xyz'%(self._path,self._name))
                    v1a,v2a,v3a=qcmath.obj_volume_6d(pod_b)
                else:
                    v1a,v2a,v3a=0,0,1
                    print('    empty')
            else:
                print('    fail')
                #
                # Try object_subtraction_dev2()
                #
                pod_b=qcmath.object_subtraction_dev2(self._pod2,pod_common,self._pod1,verbose) ####  use this (i=1,4), slow
                if pod_b.tolist()!=[[[[0]]]]:
                     v1a,v2a,v3a=qcmath.obj_volume_6d(pod_b)
                else:
                    v1a,v2a,v3a=0,0,1
                v1d,v2d,v3d=qcmath.add(v1c,v2c,v3c,v1a,v2a,v3a)
                if v1d==v1b and v2d==v2b and v3d==v3b:
                    if pod_b.tolist()!=[[[[0]]]]:
                        print('    numbre of tetrahedron %d'%(len(pod_b)))
                        file_tmp='%s/%s_pod_b'%(self._path,self._name)
                        qcmath.generator_xyz_dim4_tetrahedron(pod_b,file_tmp,0)
                        #qcmath.generator_xyz_dim4_tetrahedron(pod_b,file_tmp,1)
                        print('    saved as %s/%s_pod_b.xyz'%(self._path,self._name))
                        v1a,v2a,v3a=qcmath.obj_volume_6d(pod_b)
                    else:
                        v1a,v2a,v3a=0,0,1
                        print('    empty')
                else:
                    print('    fail')
                    #
                    # Try object_subtraction_dev()
                    #
                    pod_b=qcmath.object_subtraction_dev(self._pod2,pod_common,self._pod1,verbose) ####  use this (i=10), slow
                    if pod_b.tolist()!=[[[[0]]]]:
                         v1a,v2a,v3a=qcmath.obj_volume_6d(pod_b)
                    else:
                        v1a,v2a,v3a=0,0,1
                    v1d,v2d,v3d=qcmath.add(v1c,v2c,v3c,v1a,v2a,v3a)
                    if v1d==v1b and v2d==v2b and v3d==v3b:
                        if pod_b.tolist()!=[[[[0]]]]:
                            print('    numbre of tetrahedron %d'%(len(pod_b)))
                            file_tmp='%s/%s_pod_b'%(self._path,self._name)
                            qcmath.generator_xyz_dim4_tetrahedron(pod_b,file_tmp,0)
                            #qcmath.generator_xyz_dim4_tetrahedron(pod_b,file_tmp,1)
                            print('    saved as %s/%s_pod_b.xyz'%(self._path,self._name))
                            v1a,v2a,v3a=qcmath.obj_volume_6d(pod_b)
                        else:
                            v1a,v2a,v3a=0,0,1
                            print('    empty')
                    else:
                        print('    fail')
        else:
            v1c,v2c,v3c=0,0,1
            print('    empty')
            pod_b=self._pod2
            v1b,v2b,v3b=qcmath.obj_volume_6d(pod_b)
        print(' (A) POD1 and POD2, volume = %d %d %d (%8.6f)'%(v1c,v2c,v3c,(v1c+TAU*v2c)/float(v3c)))
        print(' (B) POD1 not POD2, volume = %d %d %d (%8.6f)'%(v1a,v2a,v3a,(v1a+TAU*v2a)/float(v3a)))
        print(' (C) POD2 not POD1, volume = %d %d %d (%8.6f)'%(v1b,v2b,v3b,(v1b+TAU*v2b)/float(v3b)))
        v1a,v2a,v3a=qcmath.add(v1a,v2a,v3a,v1c,v2c,v3c)
        print(' (D) SUM (A+B)     , volume = %d %d %d (%8.6f)'%(v1a,v2a,v3a,(v1a+TAU*v2a)/float(v3a)))

        return pod_common,pod_a,pod_b
        
    def using_tetrahedron_test(self):
        #verbose=0
        verbose=1
        #verbose=2
        print('Boolean operation (1) pod1 AND pod2, (2) pod1 NOT pod2')

        print('INPUTed two PODs')
        file_tmp='%s/%s_pod_1'%(self._path,self._name)
        qcmath.generator_xyz_dim4_tetrahedron(self._pod1,file_tmp,0)
        print(' 1st POD saved as %s/%s_pod_1.xyz'%(self._path,self._name))
        file_tmp='%s/%s_pod_2'%(self._path,self._name)
        qcmath.generator_xyz_dim4_tetrahedron(self._pod2,file_tmp,0)
        print(' 2nd POD saved as %s/%s_pod_2.xyz'%(self._path,self._name))
        #-----------------------------------------
        # POD1 and POD2, POD1 not POD2
        #-----------------------------------------
        pod_common,pod_a=qcmath.intersection_using_tetrahedron_5(self._pod1,self._pod2,self._option,verbose,self._path) ###  teset
        #
        if pod_common.tolist()!=[[[[0]]]] and pod_a.tolist()!=[[[[0]]]]:
            #
            print('  (pod_common) POD1 and POD2 : %d x tetrahedron'%(len(pod_common)))
            file_tmp='%s/%s_pod_common'%(self._path,self._name)
            qcmath.generator_xyz_dim4_tetrahedron(pod_common,file_tmp,0)
            print('  saved as %s/%s_pod_common.xyz'%(self._path,self._name))
            #
            print('  (pod_a) POD1 not POD2 : %d x tetrahedron'%(len(pod_a)))
            file_tmp='%s/%s_pod_a'%(self._path,self._name)
            qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,0)
            print('  saved as %s/%s_pod_a.xyz'%(self._path,self._name))
            #
        elif pod_a.tolist()!=[[[[0]]]] and pod_a.tolist()==[[[[0]]]]:
            #
            print('  (pod_common) POD1 and POD2 : %d x tetrahedron'%(len(pod_common)))
            file_tmp='%s/%s_pod_common'%(self._path,self._name)
            qcmath.generator_xyz_dim4_tetrahedron(pod_common,file_tmp,0)
            print('  saved as %s/%s_pod_common.xyz'%(self._path,self._name))
            #
            print('  (pod_a) POD1 not POD2 : empty')
            #
        elif pod_a.tolist()==[[[[0]]]] and pod_a.tolist()!=[[[[0]]]]:
            #
            print('  (pod_common) POD1 and POD2 : empty')
            #
            print('  (pod_a) POD1 not POD2 : %d x tetrahedron'%(len(pod_a)))
            file_tmp='%s/%s_pod_a'%(self._path,self._name)
            qcmath.generator_xyz_dim4_tetrahedron(pod_a,file_tmp,0)
            print('  saved as %s/%s_pod_a.xyz'%(self._path,self._name))
            #
        else:
            print('  (pod_common) POD1 and POD2 : empty')
            print('  (pod_a) POD1 not POD2 : empty')
            #
        return pod_common,pod_a

    
    
    
    
    
    
    