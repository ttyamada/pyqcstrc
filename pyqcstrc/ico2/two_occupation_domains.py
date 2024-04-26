#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import timeit
import os
import sys
import numpy as np

try:
    import math1 as math1
    import intsct as intsct
    import numericalc as numericalc
    import symmetry as symmetry
    import utils as utils
    import occupation_domain as od
except ImportError:
    print('import error\n')

TAU=(1+np.sqrt(5))/2.0

def intersection(obj1, obj2):
    """
    Intersection of two occupation domains projected onto perp space.
    
    Args:
        obj1 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
        obj2 (numpy.ndarray):
            The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
    
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
        
    common=intsct.intersection_two_obj_1(obj1, obj2)
    if np.all(common==None):
        print('no common part')
        return 
    else:
        return common

def intersection_convex(obj1,obj2):
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
    #point_common,point_a,point_b=intsct.intersection_two_obj_convex(obj1, obj2)
    #common=intsct.tetrahedralization_points(point_common)
    common=intsct.intersection_two_obj_convex(obj1, obj2)
    if common.tolist()!=[[[[0]]]]:
        return common
    else:
        print('no common part')
        return 
        
