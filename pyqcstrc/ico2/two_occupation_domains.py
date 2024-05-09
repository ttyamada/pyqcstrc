#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import timeit
import os
import sys
import numpy as np

try:
    import pyqcstrc.ico2.math1 as math1
    import pyqcstrc.ico2.intsct as intsct
    import pyqcstrc.ico2.numericalc as numericalc
    import pyqcstrc.ico2.symmetry as symmetry
    import pyqcstrc.ico2.utils as utils
    import pyqcstrc.ico2.occupation_domain as od
except ImportError:
    print('import error\n')

TAU=(1+np.sqrt(5))/2.0

def intersection(obj1,obj2,kind=None,verbose=0):
    """
    Return an intersection between two objects: obj1 AND obj2.
    
    Parameters
    ----------
    obj1 : ndarray
        a set of tetrahedra to be intersected with obj2.
    obj2 : ndarray
        a set of tetrahedra to be intersected with obj1.
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
        The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
    obj2 (numpy.ndarray):
        The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
    
    Returns
    -------
    Common part of two occupation domains projected onto perp space (numpy.ndarray)
        The shape is (num,4,6,3), where num=numbre_of_tetrahedron.

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
        The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
    obj2 (numpy.ndarray):
        The shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
    
    Returns
    -------
    obj1 NOT obj2
        The shape is (num,4,6,3), where num=numbre_of_tetrahedron.

    """
    
    if obj1.ndim==4 and obj2.ndim==4:
        return intsct.subtraction_two_obj(obj1,obj2,verbose)
    else:
        if verbose>0:
            print('incorrect ndim')
        return
