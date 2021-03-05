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
import pyqcstrc.icosah.math1 as math1
import pyqcstrc.icosah.intsct as intsct
import pyqcstrc.icosah.mics as mics
import pyqcstrc.icosah.numericalc as numericalc
import pyqcstrc.icosah.symmetry as symmetry
import pyqcstrc.icosah.utils as utils
import pyqcstrc.icosah.occupation_domain as od

TAU=(1+np.sqrt(5))/2.0

def intersection(obj1, obj2, flag = 0, fname = 'tmp1', verbose = 0):
    """
    flag = 0, rough check intersection of tetrahedron in obj1 and obj2; flag2 = 1, no
    """
    
    if obj1.ndim == 3:
        obj1=obj1.reshape(int(len(obj1)/4),4,6,3)
    else:
        pass
    if obj2.ndim == 3:
        obj2=obj2.reshape(int(len(obj2)/4),4,6,3)
    else:
        pass
    
    common=intsct.intersection_two_obj_1(obj1, obj2, flag, verbose)
    #if flag1==0:
    #    common=intsct.intersection_two_obj(obj1, obj2, flag1, flag2, flag3, verbose)
    #elif flag1!=0:
    #    common=intsct.intersection_using_tetrahedron_4(obj1, obj2, flag1, verbose, dummy=0)
    if common.tolist()!=[[[[0]]]]:
        [v1,v2,v3]=utils.obj_volume_6d(common)
        print('    common part found: volume = %d %d %d ( = %8.6f).'%(v1,v2,v3,(v1+TAU*v2)/v3))
        #od.write(common, path='.', basename=fname, format='xyz')
        return common
    else:
        print('    no intersection found.')
        return np.array([[[[0]]]])

def intersection_convex(obj1, obj2, fname = 'tmp2', verbose = 0):
    point_common,point_a,point_b = intsct.intersection_two_obj_convex(obj1, obj2, verbose)
    common=intsct.tetrahedralization_points(point_common,verbose-1)
    if common.tolist()!=[[[[0]]]]:
        [v1,v2,v3]=utils.obj_volume_6d(common)
        print('    common part found: volume = %d %d %d ( = %8.6f).'%(v1,v2,v3,(v1+TAU*v2)/v3))
        #od.write(common, path='.', basename=fname, format='xyz')
        return common
    else:
        print('    no intersection found.')
        return np.array([[[[0]]]])

def intersection_old(obj1, obj2, flag = [0,0,0], fname = 'tmp0', verbose = 0):
    """
    flag1 = 0, rough check intersection of obj1 and obj2; flag1 = 1, no
    flag2 = 0, rough check intersection of tetrahedron in obj1 and obj2; flag2 = 1, no
    flag3 = 0, rough check intersection of tetrahedron in obj1 and tetrahedron in obj2; flag3 = 1, no
    """
    if obj1.ndim == 3:
        obj1=obj1.reshape(int(len(obj1)/4),4,6,3)
    else:
        pass
    if obj2.ndim == 3:
        obj2=obj2.reshape(int(len(obj2)/4),4,6,3)
    else:
        pass
    
    common=intsct.intersection_two_obj(obj1, obj2, flag[0], flag[1], flag[2], verbose)
    #if flag1==0:
    #    common=intsct.intersection_two_obj(obj1, obj2, flag1, flag2, flag3, verbose)
    #elif flag1!=0:
    #    common=intsct.intersection_using_tetrahedron_4(obj1, obj2, flag1, verbose, dummy=0)
    if common.tolist()!=[[[[0]]]]:
        [v1,v2,v3]=utils.obj_volume_6d(common)
        print('    common part found: volume = %d %d %d ( = %8.6f).'%(v1,v2,v3,(v1+TAU*v2)/v3))
        #od.write(common, path='.', basename=fname, format='xyz')
        return common
    else:
        print('    no intersection found.')
        return np.array([[[[0]]]])

def subtraction(obj1, obj2, obj_common, verbose = 0):
    """ obj1 NOT obj2
    """
    a1,b1,c1=utils.obj_volume_6d(obj1)
    a2,b2,c2=utils.obj_volume_6d(obj_common)

    obj3=mics.object_subtraction_2(obj1, obj_common, verbose)

    if obj3.tolist()!=[[[[0]]]]:
        a3,b3,c3=utils.obj_volume_6d(obj3)
    else:
       a3,b3,c3=0,0,1
    a4,b4,c4=math1.add(a2,b2,c2,a3,b3,c3)
    if verbose>0:
        print('       Volume0 (obj1)          = %d %d %d ( = %8.6f)'%(a1,b1,c1,(a1+TAU*b1)/c1))
        print('       Volume1 (obj1 AND obj2) = %d %d %d ( = %8.6f)'%(a2,b2,c2,(a2+TAU*b2)/c2))
        print('       Volume2 (obj1 NOT obj2) = %d %d %d ( = %8.6f)'%(a3,b3,c3,(a3+TAU*b3)/c3))
        print('       Volume1+Volume2         = %d %d %d ( = %8.6f)'%(a4,b4,c4,(a4+TAU*b4)/c4))
    if a1==a4 and b1==b4 and c1==c4:
        print('    succeeded: object_subtraction_2()')
        print('      obj1 NOT obj2: volume = (%d+%d*TAU)/%d ( = %8.6f).'%(a3,b3,c3,(a3+TAU*b3)/c3))
        return obj3
    else:
        print('fail: object_subtraction_2()')
        obj3=mics.object_subtraction_dev2(obj1, obj_common, obj2, verbose)
        if obj3.tolist()!=[[[[0]]]]:
            a3,b3,c3=utils.obj_volume_6d(obj3)
        else:
           a3,b3,c3=0,0,1
        a4,b4,c4=math1.add(a2,b2,c2,a3,b3,c3)
        if verbose>0:
            print('       Volume0 (obj1)          = %d %d %d ( = %8.6f)'%(a1,b1,c1,(a1+TAU*b1)/c1))
            print('       Volume1 (obj1 AND obj2) = %d %d %d ( = %8.6f)'%(a2,b2,c2,(a2+TAU*b2)/c2))
            print('       Volume2 (obj1 NOT obj2) = %d %d %d ( = %8.6f)'%(a3,b3,c3,(a3+TAU*b3)/c3))
            print('       Volume1+Volume2         = %d %d %d ( = %8.6f)'%(a4,b4,c4,(a4+TAU*b4)/c4))
        if a1==a4 and b1==b4 and c1==c4:
            print('    succeeded: object_subtraction_dev2()')
            print('      obj1 NOT obj2: volume = %d %d %d ( = %8.6f).'%(a3,b3,c3,(a3+TAU*b3)/c3))
            return obj3
        else:
            print('fail: object_subtraction_dev2()')
            obj3=mics.object_subtraction_dev1(obj1, obj_common, obj2, verbose)
            if obj3.tolist()!=[[[[0]]]]:
                a3,b3,c3=utils.obj_volume_6d(obj3)
            else:
               a3,b3,c3=0,0,1
            a4,b4,c4=math1.add(a2,b2,c2,a3,b3,c3)
            if verbose>0:
                print('       Volume0 (obj1)          = %d %d %d ( = %8.6f)'%(a1,b1,c1,(a1+TAU*b1)/c1))
                print('       Volume1 (obj1 AND obj2) = %d %d %d ( = %8.6f)'%(a2,b2,c2,(a2+TAU*b2)/c2))
                print('       Volume2 (obj1 NOT obj2) = %d %d %d ( = %8.6f)'%(a3,b3,c3,(a3+TAU*b3)/c3))
                print('       Volume1+Volume2         = %d %d %d ( = %8.6f)'%(a4,b4,c4,(a4+TAU*b4)/c4))
            if a1==a4 and b1==b4 and c1==c4:
                print('    succeeded: object_subtraction_dev1()')
                print('      obj1 NOT obj2: volume = %d %d %d ( = %8.6f).'%(a3,b3,c3,(a3+TAU*b3)/c3))
                return obj3
            else:
                print('fail: object_subtraction_dev1()')
                obj3=mics.object_subtraction_dev(obj1, obj_common, obj2, verbose)
                if obj3.tolist()!=[[[[0]]]]:
                    a3,b3,c3=utils.obj_volume_6d(obj3)
                else:
                   a3,b3,c3=0,0,1
                a4,b4,c4=math1.add(a2,b2,c2,a3,b3,c3)
                if verbose>0:
                    print('       Volume0 (obj1)          = %d %d %d ( = %8.6f)'%(a1,b1,c1,(a1+TAU*b1)/c1))
                    print('       Volume1 (obj1 AND obj2) = %d %d %d ( = %8.6f)'%(a2,b2,c2,(a2+TAU*b2)/c2))
                    print('       Volume2 (obj1 NOT obj2) = %d %d %d ( = %8.6f)'%(a3,b3,c3,(a3+TAU*b3)/c3))
                    print('       Volume1+Volume2         = %d %d %d ( = %8.6f)'%(a4,b4,c4,(a4+TAU*b4)/c4))
                if a1==a4 and b1==b4 and c1==c4:
                    print('    succeeded: object_subtraction_dev()')
                    print('      obj1 NOT obj2: volume = %d %d %d ( = %8.6f).'%(a3,b3,c3,(a3+TAU*b3)/c3))
                    return obj3
                else:
                    print('subtraction fail')
                    return np.array([[[[0]]]])

 