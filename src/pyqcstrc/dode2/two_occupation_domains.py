#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import timeit
import os
import sys
import numpy as np
sys.path.append('.')
try:
    import math1 as math1
    import utils as utils
    import umericalc as numericalc
    import symmetry as symmetry
    import intsct as intsct
    import occupation_domain as od
"""
except ImportError:
    print('import error\n')
try:
    import pyqcstrc.dode2.math1 as math1
    import pyqcstrc.dode2.intsct as intsct
    import pyqcstrc.dode2.numericalc as numericalc
    import pyqcstrc.dode2.symmetry as symmetry
    import pyqcstrc.dode2.utils as utils
    import pyqcstrc.dode2.occupation_domain as od
except ImportError:
    print('import error\n')
"""
TAU=np.sqrt(3)/2.0

def intersection(obj1,obj2,kind=None,verbose=0):
    """
    Return an intersection between two objects: obj1 AND obj2.
    
    Parameters
    ----------
    obj1 : ndarray
        a set of triangles to be intersected with obj2.
    obj2 : ndarray
        a set of triangles to be intersected with obj1.
    kind : {'standard', 'simple'}, optional
        The default is 'standard'. 
    
    Returns
    -------
    intersection between obj1 and obj2 : ndarray
        Array of the same type and shape as `obj1` and `obj2`.
    
    Notes
    -----
    
    'standard' intersection ...
    
    
    
    'simple' intersection ...
    
    """
    if obj1.ndim==4 and obj2.ndim==4:
        common=intsct.intersection_two_obj_1(obj1,obj2,kind)
        if np.all(common==None):
            if verbose>0:
                print('no common part')
            return 
        else:
            return common
    else:
        if verbose>0:
            print('incorrect ndim')
        return 

def intersection_convex(obj1,obj2,verbose=0):
    """
    Intersection of two occupation domains projected onto perp space: obj1 AND obj2.
    The common part forms convex hull.
    
    Parameters
    ----------
    obj1 (numpy.ndarray):
        The shape is (num,3,6,3), where num=numbre_of_triangles.
    obj2 (numpy.ndarray):
        The shape is (num,3,6,3), where num=numbre_of_triangles.
    
    Returns
    -------
    Common part of two occupation domains projected onto perp space (numpy.ndarray)
        The shape is (num,3,6,3), where num=numbre_of_triangles.

    """
    if obj1.ndim==4 and obj2.ndim==4:
        common=intsct.intersection_two_obj_convex(obj1,obj2)
        if np.all(common==None):
            if verbose>0:
                print('no common part')
            return 
        else:
            return common
    else:
        if verbose>0:
            print('incorrect ndim')
        return
        
def subtraction(obj1,obj2,verbose=0):
    """
    Subtraction of two occupation domains projected onto perp space: obj1 NOT obj2 = obj1 NOT (obj1 AND obj2).
    
    Parameters
    ----------
    obj1 (numpy.ndarray):
        The shape is (num,3,6,3), where num=numbre_of_triangles.
    obj2 (numpy.ndarray):
        The shape is (num,3,6,3), where num=numbre_of_triangles.
    
    Returns
    -------
    obj1 NOT obj2
        The shape is (num,3,6,3), where num=numbre_of_triangles.

    """
    
    if obj1.ndim==4 and obj2.ndim==4:
        return intsct.subtraction_two_obj(obj1,obj2,verbose)
    else:
        if verbose>0:
            print('incorrect ndim')
        return
