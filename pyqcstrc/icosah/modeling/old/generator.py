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
sys.path.append('.')
import qcmath

TAU=(1+np.sqrt(5))/2.0

def shift_obj(obj,shift,vorbose):
    return qcmath.shift_object(obj,shift,vorbose)

def gen_obj_edges(obj):
    obj_surface=qcmath.generator_surface(obj)
    obj_edge=qcmath.generator_edge(obj_surface)
    return obj_edge

def gen_obj_surface(obj):
    obj_surface=qcmath.generator_surface(obj)
    return obj_surface

class OD(object):
    def __init__(self,asymmetric_part,position):
        self._asymmetric_part=asymmetric_part
        self._position=position
    def symmetric(self):
        return qcmath.generator_obj_symmetric_tetrahedron(self._asymmetric_part,self._position)
    def asymmetric(self):
        return qcmath.generator_obj_symmetric_tetrahedron_0(self._asymmetric_part,self._position,0)
    def as_it_is(self):
        return np.array(self._asymmetric_part)

class Intersection(object):
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

    def using_tetrahedron(self):
        print('Intersection of POD1 and POD2.')
        print(' 1st POD (POD1):')
        v1b,v2b,v3b=qcmath.obj_volume_6d(self._pod1)
        print('  volume = (%d+%d*TAU)/%d ( = %8.6f)'%(v1b,v2b,v3b,(v1b+TAU*v2b)/float(v3b)))
        file_tmp='%s/%s_pod_1'%(self._path,self._name)
        qcmath.generator_xyz_dim4_tetrahedron(self._pod1,file_tmp,0)
        #export_vesta_file(self._pod1,self._path,self._name+'_pod_1',1)
        #qcmath.generator_xyz_dim4_tetrahedron(self._pod1,file_tmp,1)
        print('  saved in %s/%s_pod_1.xyz'%(self._path,self._name))
        print(' 2nd pod (POD2):')
        v1a,v2a,v3a=qcmath.obj_volume_6d(self._pod2)
        print('  volume = (%d+%d*TAU)/%d ( = %8.6f)'%(v1a,v2a,v3a,(v1a+TAU*v2a)/float(v3a)))
        file_tmp='%s/%s_pod_2'%(self._path,self._name)
        qcmath.generator_xyz_dim4_tetrahedron(self._pod2,file_tmp,0)
        #export_vesta_file(self._pod2,self._path,self._name+'_pod_2',1)
        print('  saved in %s/%s_pod_2.xyz'%(self._path,self._name))
        #-----------------------------------------
        # (1) POD1 and POD2 : obj_common
        #-----------------------------------------
        print(' Intersection, POD1 and POD2')
        pod_common=qcmath.intersection_two_obj(self._pod1,self._pod2,verbose=0)
        if pod_common.tolist()!=[[[[0]]]]:
            #print('    numbre of tetrahedron %d'%(len(pod_common)))
            file_tmp='%s/%s_pod_common'%(self._path,self._name)
            qcmath.generator_xyz_dim4_tetrahedron(pod_common,file_tmp,0)
            #qcmath.generator_xyz_dim4_tetrahedron(pod_common,file_tmp,1)
            v1a,v2a,v3a=qcmath.obj_volume_6d(pod_common)
            print('  volume = (%d+%d*TAU)/%d ( = %8.6f)'%(v1a,v2a,v3a,(v1a+TAU*v2a)/float(v3a)))
            print('  saved in %s/%s_pod_common.xyz'%(self._path,self._name))
        else:
            print('    no intersection')
