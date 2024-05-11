#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import timeit
import os
import sys
import numpy as np

try:
    import pyqcstrc.ico.math1 as math1
    import pyqcstrc.ico.intsct as intsct
    import pyqcstrc.ico.mics as mics
    import pyqcstrc.ico.numericalc as numericalc
    import pyqcstrc.ico.symmetry as symmetry
    import pyqcstrc.ico.utils as utils
    import pyqcstrc.ico.occupation_domain as od

except ImportError:
    print('import error\n')

TAU=(1+np.sqrt(5))/2.0

def intersection(obj1, obj2, flag = 0, verbose = 0):
    """
    Intersection of two occupation domains projected onto perp space.
    
    Args:
        obj1 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
        obj2 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
        flag (int):
            flag = 0, Intersection with rough checking.
            flag = 1, Intersection without checking.
        verbose (int):
            verbose = 0 (silent, default)
            verbose = 1 (normal)
    
    Returns:
        Common part of two occupation domains projected onto perp space (numpy.ndarray):
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.

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
        if verbose>0:
            print('    common part found: volume = %d %d %d ( = %8.6f).'%(v1,v2,v3,(v1+TAU*v2)/v3))
        #od.write(common, path='.', basename=fname, format='xyz')
        return common
    else:
        if verbose>0:
            print('    no intersection found.')
        return np.array([[[[0]]]])

def intersection_2(obj1, obj2, verbose = 0):
    """
    Intersection of two occupation domains projected onto perp space.
    
    Args:
        obj1 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
        obj2 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
        verbose (int):
            verbose = 0 (silent, default)
            verbose = 1 (normal)
    
    Returns:
        Common part of two occupation domains projected onto perp space (numpy.ndarray):
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.

    """
    
    if obj1.ndim == 3:
        obj1=obj1.reshape(int(len(obj1)/4),4,6,3)
    else:
        pass
    if obj2.ndim == 3:
        obj2=obj2.reshape(int(len(obj2)/4),4,6,3)
    else:
        pass
    
    common=intsct.intersection_two_obj_2(obj1, obj2, verbose)

    if common.tolist()!=[[[[0]]]]:
        [v1,v2,v3]=utils.obj_volume_6d(common)
        print('    common part found: volume = %d %d %d ( = %8.6f).'%(v1,v2,v3,(v1+TAU*v2)/v3))
        return common
    else:
        print('    no intersection found.')
        return np.array([[[[0]]]])

def intersection_3(obj1, obj2, verbose = 0):
    """
    Intersection of two occupation domains projected onto perp space.
    
    Args:
        obj1 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
        obj2 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
        verbose (int):
            verbose = 0 (silent, default)
            verbose = 1 (normal)
    
    Returns:
        Common part of two occupation domains projected onto perp space (numpy.ndarray):
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.

    """
    
    if obj1.ndim == 3:
        obj1=obj1.reshape(int(len(obj1)/4),4,6,3)
    else:
        pass
    if obj2.ndim == 3:
        obj2=obj2.reshape(int(len(obj2)/4),4,6,3)
    else:
        pass
    
    common=intsct.intersection_two_obj_3(obj1, obj2, verbose)

    if common.tolist()!=[[[[0]]]]:
        [v1,v2,v3]=utils.obj_volume_6d(common)
        print('    common part found: volume = %d %d %d ( = %8.6f).'%(v1,v2,v3,(v1+TAU*v2)/v3))
        return common
    else:
        print('    no intersection found.')
        return np.array([[[[0]]]])

def intersection_convex(obj1, obj2, verbose = 0):
    """
    Intersection of two occupation domains projected onto perp space.
    The common part forms convex hull.
    
    Args:
        obj1 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
        obj2 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
        verbose (int):
            verbose = 0 (silent, default)
            verbose = 1 (normal)
    
    Returns:
        Common part of two occupation domains projected onto perp space (numpy.ndarray)
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.

    """
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
    legacy
    Intersection of two occupation domains projected onto perp space.
    
    Args:
        obj1 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
        obj2 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
        flag (list): list = [a1,a2,a3]
            a1 = 0, rough check intersection of obj1 and obj2; a1 = 1, no
            a2 = 0, rough check intersection of tetrahedron in obj1 and obj2; a2 = 1, no
            a3 = 0, rough check intersection of tetrahedron in obj1 and tetrahedron in obj2; a3 = 1, no
        fname (str): filename of output
        verbose (int):
            verbose = 0 (silent, default)
            verbose = 1 (normal)
    
    Returns:
        Common part of two occupation domains projected onto perp space (numpy.ndarray):
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
    
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

    legacy
    Substruction of two occupation domains (obj1, obj2) projected onto perp space, obj1 NOT obj2
    
    Args:
        obj1 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
        obj2 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
        obj_common (numpy.ndarray):
            Common part of two occupation domains projected onto perp space.
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        verbose (int):
            verbose = 0 (silent, default)
            verbose = 1 (normal)
    
    Returns:
        Substructed part of two occupation domains projected onto perp space (numpy.ndarray)
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
    
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

def mergev(obj1, obj2, path = '.', basename = 'tmp', verbose = 0):
    """
    merge vertices of two obj
    """
    v=np.append(obj1,obj2)
    v=utils.remove_doubling_dim3_in_perp_space(v.reshape(int(len(v)/18),6,3))
    od.write_xyz(obj=v, path=path, basename=basename, select='vertex', verbose = verbose)
    if verbose>0:
        print('vertices merged, %s/%s'%(path, basename))
    else:
        pass
    
    return 0

def mergev3(obj1, obj2, obj3, path = '.', basename = 'tmp', verbose = 0):
    """
    merge vertices of three obj
    """
    v=np.append(obj1,obj2)
    v=np.append(v,obj3)
    v=utils.remove_doubling_dim3_in_perp_space(v.reshape(int(len(v)/18),6,3))
    od.write_xyz(obj=v, path=path, basename=basename, select='vertex', verbose = verbose)
    if verbose>0:
        print('vertices merged, %s/%s'%(path, basename))
    else:
        pass
    
    return 0

def mergev_objs(objs_in_list, path = '.', basename = 'tmp', verbose = 0):
    """
    merge vertices of objs
    """
    v=objs_in_list[0]
    for i in range(1,len(objs_in_list)):
        v=np.append(v,objs_in_list[i])
    v=utils.remove_doubling_dim3_in_perp_space(v.reshape(int(len(v)/18),6,3))
    od.write_xyz(obj=v, path=path, basename=basename, select='vertex', verbose = verbose)
    if verbose>0:
        print('vertices merged, %s/%s'%(path, basename))
    else:
        pass
    
    return 0

def genobjv(obj, vlist = [0], path = '.', basename = 'tmp', verbose = 0):
    """
    Generate objs from vlist.
    
    Args:
        obj (numpy.ndarray): ndarray contains vertices
            The shape is (num,4,6,3), where num=numbre_of_vertices.
        vlist (list): 
        path (str): Path of the output files
        basename (str): Basename of the output files
        verbose (int):
            verbose = 0 (silent, default)
            verbose = 1 (normal)
    Returns:
        int: 0 (succeed), 1 (fail)
    
    """
    if os.path.exists(path) == False:
        os.makedirs(path)
    else:
        pass
    for i1 in range(len(vlist)):
        a=0
        for i2 in range(len(vlist[i1])):
            for i3 in range(4):
                if i2==0 and i3==0:
                    a=obj[vlist[i1][i2][0]-1]
                else:
                    a=np.append(a,obj[vlist[i1][i2][i3]-1])
        a=a.reshape(int(len(a)/72),4,6,3)
        od.write_xyz(obj=a, path=path, basename='%s_part%d'%(basename,i1), verbose=verbose+1)
        od.write_vesta(obj=a, path=path, basename='%s_part%d'%(basename,i1), color='p', verbose=verbose+1)
    return 0
