#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import timeit
import os
import sys
import numpy as np
try:
    import pyqcstrc.dode2.math1 as math1
    import pyqcstrc.dode2.utils as utils
    import pyqcstrc.dode2.numericalc as numericalc
    import pyqcstrc.dode2.symmetry as symmetry
    import pyqcstrc.dode2.intsct as intsct
except ImportError:
    print('import error\n')

TAU=np.sqrt(3)/2.0

def volume(obj):
    return utils.obj_area_6d(obj)

def symmetric(obj,centre):
    """
    Generate symmterical occupation domain by symmetric elements on the asymmetric unit.
    
    Args:
        obj (numpy.ndarray):
            Asymmetric unit of the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_tetrahedron.
        centre (numpy.ndarray):
            6d coordinate of the symmetric centre.
            The shape is (6,3)
    
    Returns:
        Symmetric occupation domains (numpy.ndarray):
            The shape is (num,3,6,3), where num=numbre_of_tetrahedron.
    
    """
    if obj.ndim==3 or obj.ndim==4:
        #return symmetry.generator_obj_symmetric_tetrahedron(obj,centre)
        return symmetry.generator_obj_symmetric_triangle(obj,centre)
    else:
        print('object has an incorrect shape!')
        return 

def symmetric_0(obj,centre,indx_symop):
    """
    Generate symmtericic occupation domain by applying symmetric elements on the asymmetric unit.
    
    Args:
        obj (numpy.ndarray):
            Asymmetric unit of the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        centre (numpy.ndarray):
            6d coordinate of the symmetric centre.
            The shape is (6,3)
    
    Returns:
        Symmetric occupation domains (numpy.ndarray):
            The shape is (num,3,6,3), where num=numbre_of_tetrahedron.
    
    """
    if obj.ndim==3 or obj.ndim==4:
        return symmetry.generator_obj_symmetric_triangle_0(obj,centre,indx_symop)
    else:
        print('object has an incorrect shape!')
        return 

def shift(obj,shift):
    """
    Shift the occupation domain.
    
    Args:
        obj (numpy.ndarray):
            The occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        shift (numpy.ndarray):
            6d coordinate to which the occupation domain is shifted.
            The shape is (6,3)
        verbose (int):
            verbose = 0 (silent, default)
            verbose = 1 (normal)
    
    Returns:
        Shifted occupation domains (numpy.ndarray):
            The shape is (num,3,6,3), where num=numbre_of_tetrahedron.
    
    """
    return utils.shift_object(obj, shift)

def write(obj=None,path=None,basename=None,format=None,color='k',select=None,verbose=0):
    """
    Export occupation domains.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        path (str): Path of the output XYZ file
        basename (str): Basename of the output XYZ file
        format (str): format of output file
            format = 'xyz' (default)
            format = 'vesta'
        color (str)
            one of the characters {'k','r','b','p'}, which are short-hand notations 
            for shades of black, red, blue, and pink, in case where 'vesta' format is
            selected (default, color = 'k').
        select (str):'simple', 'normal', or 'egdes'
            'simple': Merging triangles into one single objecte
            'normal': Each triangle is set as single objecte (large file)
            'egdes':  Select this option when the obj is a set of edges.
    
    Returns:
        int: 0 (succeed), 1 (fail)
    
    """
    
    if os.path.exists(path)==False:
        os.makedirs(path)
    else:
        pass
    
    if np.all(obj==None):
        print('    Empty OD')
        return 0
    else:
        if format=='vesta':
            if select==None:
                select='normal'
            write_vesta(obj,path,basename,color,select,verbose)
            return 0
        elif format == 'xyz':
            if select==None:
                select='triangle'
            write_xyz(obj,path,basename,select,verbose)
            return 0
        else:
            return 1

def write_vesta(obj,path='.',basename='tmp',color='k',select='normal',verbose=0):
    """
    Export occupation domains in VESTA format.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        path (str): Path of the output XYZ file
        basename (str): Basename of the output XYZ file
        color (str)
            one of the characters {'k','r','b','p','l','y','c','s'}, which are short-hand notations 
            for shades of black, red, blue, pink, lime, yellow, cyan, and silver in case where 'vesta' format is
            selected (default, color = 'k').
        select (str):'simple', 'normal', 'egdes', or 'podatm'
            'simple': Merging triangles into one single objecte
            'normal': Each triangle is set as single objecte (large file)
            'egdes':  Select this option when the obj is a set of edges.
            'podatm': same as 'simple' but return "vertices" necessary to input 
            (default, select = 'normal')
    Returns:
        int: 0 (succeed), 1 (fail) when select = 'simple' or 'normal'.
        ndarray: vertices, when select = 'podatm'.
    """
    #print('write_vesta()')
    
    if os.path.exists(path)==False:
        os.makedirs(path)
    else:
        pass
    
    def colors(code):
        if code=='red' or code=='r':
            a = [255,0,0]
        elif code=='blue' or code=='b':
            a = [0,0,255]
        elif code=='black' or code=='k':
            a = [127,127,127]
        elif code=='pink' or code=='p':
            a = [255,0,255]
        elif code=='lime' or code=='l':
            a = [0,255,0]
        elif code=='yellow' or code=='y':
            a = [255,255,0]
        elif code=='cyan' or code=='c':
            a = [0,255,255]
        elif code=='silver' or code=='s':
            a = [192,192,192]
        elif len(code)==3:
            a = code
        else:
            a = [127,127,127]
        return a
    
    file_name='%s/%s.vesta'%(path,basename)
    f=open('%s'%(file_name),'w')
    
    #dmax=5.0
    dmax=10.0
    
    if select=='simple' or select=='egdes':
        if np.all(obj==None):
            print('no volume obj')
            return 0
        else:
            # get independent edges
            if select=='simple':
                edges = utils.generator_obj_edge(obj,verbose)
            else:
                edges = obj
            # get independent vertices of the edges
            vertices = utils.remove_doubling_in_perp_space(edges)
                
            # get bond pairs, [[distance, XXX, YYY],...]
            pairs = []
            for edge in edges:
                dist=intsct.distance_in_perp_space(edge[0],edge[1])
                a=[dist]
                for i2 in range(2):
                    for i3,vt in enumerate(vertices):
                        tmp=np.vstack([edge[i2],vt])
                        tmp=utils.remove_doubling_in_perp_space(tmp.reshape(2,6,3))
                        if len(tmp)==1:
                            a.append(i3)
                            break
                        else:
                            pass
                pairs.append(a)
                
            print('#VESTA_FORMAT_VERSION 3.5.0\n', file=f)
            print('MOLECULE\
            \nTITLE',file=f)
            print('%s/%s\n'%(path,basename), file=f)
            print('GROUP\
            \n1 1 Custom\
            \nSYMOP\
            \n 0.000000  0.000000  0.000000  1  0  0    0  1  0    0  0  1    1\
            \n -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0\
            \nTRANM 0\
            \n 0.000000  0.000000  0.000000  1  0  0    0  1  0    0  0  1\
            \nLTRANSL\
            \n -1\
            \n 0.000000  0.000000  0.000000  0.000000  0.000000  0.000000\
            \nLORIENT\
            \n -1    0    0    0    0\
            \n 1.000000  0.000000  0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000  0.000000  1.000000\
            \nLMATRIX\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000  0.000000\
            \nCELLP\
            \n  1.000000    1.000000    1.000000  90.000000  90.000000  90.000000\
            \n  0.000000    0.000000    0.000000    0.000000    0.000000    0.000000\
            \nSTRUC', file=f)
            for i2,vrtx in enumerate(vertices):
                xyz = math1.projection3(vrtx)
                xyz=numericalc.numerical_vector(xyz)
                print('%4d A        A%d  1.0000    %8.6f %8.6f %8.6f        1'%\
                (i2+1,i2+1,xyz[0],xyz[1],xyz[2]), file=f)
                print('                             0.000000    0.000000    0.000000  0.00', file=f)
            print('  0 0 0 0 0 0 0\
            \nTHERI 0', file = f)
            i2=0
            for __ in vertices:
                print('  %d        A%d  1.000000'%(i2+1,i2+1), file=f)
                i2+=1
            print('  0 0 0\
            \nSHAPE\
            \n  0         0         0         0    0.000000  0    192    192    192    192\
            \nBOUND\
            \n         0          1        0          1        0          1\
            \n  0    0    0    0  0\
            \nSBOND', file = f)
            clr=colors(color)
            for i2,pair in enumerate(pairs):
                print('  %d   A%d   A%d   %8.6f   %8.6f  0  1  1  1  2  0.250  2.000 %3d %3d %3d'%(\
                i2+1, pair[1]+1, pair[2]+1, pair[0]-0.01, pair[0]+0.01, clr[0], clr[1], clr[2]), file=f)
            print('  0 0 0 0\
            \nSITET', file = f)
            for i2 in range(len(vertices)):
                print('    %d        A%d  0.050  76  76  76  76  76  76 204  0'%(i2+1,i2+1), file=f)
            print('  0 0 0 0 0 0\
            \nVECTR\
            \n 0 0 0 0 0\
            \nVECTT\
            \n 0 0 0 0 0\
            \nSPLAN\
            \n  0    0    0    0\
            \nLBLAT\
            \n -1\
            \nLBLSP\
            \n -1\
            \nDLATM\
            \n -1\
            \nDLBND\
            \n -1\
            \nDLPLY\
            \n -1\
            \nPLN2D\
            \n  0    0    0    0', file = f)
        
            print('ATOMT\
            \n  1        A  0.0100  76  76  76  76  76  76 204\
            \n  0 0 0 0 0 0\
            \nSCENE\
            \n 1.000000 -0.000000 -0.000000  0.000000\
            \n 0.000000  1.000000 -0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n  0.000    0.000\
            \n  0.000\
            \n  1.320\
            \nHBOND 0 2\
            \n\
            \nSTYLE\
            \nDISPF 37753794\
            \nMODEL    0  1  0\
            \nSURFS    0  1  1\
            \nSECTS  32  1\
            \nFORMS    0  1\
            \nATOMS    0  0  1\
            \nBONDS    2\
            \nPOLYS    1\
            \nVECTS 1.000000\
            \nFORMP\
            \n  1  1.0    0    0    0\
            \nATOMP\
            \n 24  24    0  50  2.0    0\
            \nBONDP\
            \n  1  16  0.250  2.000 127 127 127\
            \nPOLYP\
            \n 204 1  1.000 180 180 180\
            \nISURF\
            \n  0    0    0    0\
            \nTEX3P\
            \n  1  0.00000E+00  1.00000E+00\
            \nSECTP\
            \n  1  5.00000E-01  5.00000E-01  0.00000E+00  0.00000E+00  0.00000E+00  0.00000E+00\
            \nCONTR\
            \n 0.1 -1 1 1 10 -1 2 5\
            \n 2 1 2 1\
            \n    0    0    0\
            \n    0    0    0\
            \n    0    0    0\
            \n    0    0    0\
            \nHKLPP\
            \n 192 1  1.000 255    0 255\
            \nUCOLP\
            \n    0    1  1.000    0    0    0\
            \nCOMPS 0\
            \nLABEL 1     12  1.000 0\
            \nPROJT 0  0.962\
            \nBKGRC\
            \n 255 255 255\
            \nDPTHQ 1 -0.5000  3.5000\
            \nLIGHT0 1\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n  26  26  26 255\
            \n 179 179 179 255\
            \n 255 255 255 255\
            \nLIGHT1\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nLIGHT2\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nLIGHT3\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nATOMM\
            \n 204 204 204 255\
            \n  25.600\
            \nBONDM\
            \n 255 255 255 255\
            \n 128.000\
            \nPOLYM\
            \n 255 255 255 255\
            \n 128.000\
            \nSURFM\
            \n    0    0    0 255\
            \n 128.000\
            \nFORMM\
            \n 255 255 255 255\
            \n 128.000\
            \nHKLPM\
            \n 255 255 255 255\
            \n 128.000',file = f)
        
            f.close()
            if verbose>0:
                print('    written in %s'%(file_name))
            return 0
        
    elif select=='normal':
        if np.all(obj==None):
            print('no volume obj')
            return 0
        else:
            print('#VESTA_FORMAT_VERSION 3.5.0\n', file=f)
            for i1,obj1 in enumerate(obj):
                print('MOLECULE\
                \nTITLE',file=f)
                print('%s/%s_%d\n'%(path,basename,i1), file=f)
                print('GROUP\
                \n1 1 Custom\
                \nSYMOP\
                \n 0.000000  0.000000  0.000000  1  0  0    0  1  0    0  0  1    1\
                \n -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0\
                \nTRANM 0\
                \n 0.000000  0.000000  0.000000  1  0  0    0  1  0    0  0  1\
                \nLTRANSL\
                \n -1\
                \n 0.000000  0.000000  0.000000  0.000000  0.000000  0.000000\
                \nLORIENT\
                \n -1    0    0    0    0\
                \n 1.000000  0.000000  0.000000  1.000000  0.000000  0.000000\
                \n 0.000000  0.000000  1.000000  0.000000  0.000000  1.000000\
                \nLMATRIX\
                \n 1.000000  0.000000  0.000000  0.000000\
                \n 0.000000  1.000000  0.000000  0.000000\
                \n 0.000000  0.000000  1.000000  0.000000\
                \n 0.000000  0.000000  0.000000  1.000000\
                \n 0.000000  0.000000  0.000000\
                \nCELLP\
                \n  1.000000    1.000000    1.000000  90.000000  90.000000  90.000000\
                \n  0.000000    0.000000    0.000000    0.000000    0.000000    0.000000\
                \nSTRUC', file=f)
                for i2,vertx in enumerate(obj1):
                    xyz=math1.projection3(vertx)
                    xyz=numericalc.numerical_vector(xyz)
                    print('%4d Xx        Xx%d  1.0000    %8.6f %8.6f %8.6f        1'%\
                    (i2+1,i2+1,xyz[0],xyz[1],xyz[2]), file=f)
                    print('                             0.000000    0.000000    0.000000  0.00', file=f)
                print('  0 0 0 0 0 0 0\
                \nTHERI 0', file=f)
                for i2,_ in enumerate(obj1):
                    print('  %d        Xx%d  1.000000'%(i2+1,i2+1), file=f)
                print('  0 0 0\
                \nSHAPE\
                \n  0         0         0         0    0.000000  0    192    192    192    192\
                \nBOUND\
                \n         0          1        0          1        0          1\
                \n  0    0    0    0  0\
                \nSBOND', file=f)
                clr=colors(color)
                print('  1     Xx     Xx     0.00000     %3.2f  0  1  1  0  2  0.250  2.000 %3d %3d %3d'%(dmax,clr[0],clr[1],clr[2]), file=f)
                print('  0 0 0 0\
                \nSITET', file=f)
                for i2,_ in enumerate(obj1):
                    print('    %d        Xx%d  0.0100  76  76  76  76  76  76 204  0'%(i2+1,i2+1), file=f)
                print('  0 0 0 0 0 0\
                \nVECTR\
                \n 0 0 0 0 0\
                \nVECTT\
                \n 0 0 0 0 0\
                \nSPLAN\
                \n  0    0    0    0\
                \nLBLAT\
                \n -1\
                \nLBLSP\
                \n -1\
                \nDLATM\
                \n -1\
                \nDLBND\
                \n -1\
                \nDLPLY\
                \n -1\
                \nPLN2D\
                \n  0    0    0    0', file=f)
            print('ATOMT\
            \n  1        Xx  0.0100  76  76  76  76  76  76 204\
            \n  0 0 0 0 0 0\
            \nSCENE\
            \n 1.000000 -0.000000 -0.000000  0.000000\
            \n 0.000000  1.000000 -0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n  0.000    0.000\
            \n  0.000\
            \n  1.320\
            \nHBOND 0 2\
            \n\
            \nSTYLE\
            \nDISPF 37753794\
            \nMODEL    0  1  0\
            \nSURFS    0  1  1\
            \nSECTS  32  1\
            \nFORMS    0  1\
            \nATOMS    0  0  1\
            \nBONDS    2\
            \nPOLYS    1\
            \nVECTS 1.000000\
            \nFORMP\
            \n  1  1.0    0    0    0\
            \nATOMP\
            \n 24  24    0  50  2.0    0\
            \nBONDP\
            \n  1  16  0.250  2.000 127 127 127\
            \nPOLYP\
            \n 204 1  1.000 180 180 180\
            \nISURF\
            \n  0    0    0    0\
            \nTEX3P\
            \n  1  0.00000E+00  1.00000E+00\
            \nSECTP\
            \n  1  5.00000E-01  5.00000E-01  0.00000E+00  0.00000E+00  0.00000E+00  0.00000E+00\
            \nCONTR\
            \n 0.1 -1 1 1 10 -1 2 5\
            \n 2 1 2 1\
            \n    0    0    0\
            \n    0    0    0\
            \n    0    0    0\
            \n    0    0    0\
            \nHKLPP\
            \n 192 1  1.000 255    0 255\
            \nUCOLP\
            \n    0    1  1.000    0    0    0\
            \nCOMPS 0\
            \nLABEL 1     12  1.000 0\
            \nPROJT 0  0.962\
            \nBKGRC\
            \n 255 255 255\
            \nDPTHQ 1 -0.5000  3.5000\
            \nLIGHT0 1\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n  26  26  26 255\
            \n 179 179 179 255\
            \n 255 255 255 255\
            \nLIGHT1\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nLIGHT2\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nLIGHT3\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nATOMM\
            \n 204 204 204 255\
            \n  25.600\
            \nBONDM\
            \n 255 255 255 255\
            \n 128.000\
            \nPOLYM\
            \n 255 255 255 255\
            \n 128.000\
            \nSURFM\
            \n    0    0    0 255\
            \n 128.000\
            \nFORMM\
            \n 255 255 255 255\
            \n 128.000\
            \nHKLPM\
            \n 255 255 255 255\
            \n 128.000',file=f)
            f.close()
            if verbose>0:
                print('    written in %s'%(file_name))
            return 0
    
    elif select == 'podatm':
        if np.all(obj==None):
            print('no volume obj')
            return 0
        else:
            # get independent edges
            edges = utils.generator_obj_edge(obj, verbose)
            # get independent vertices of the edges
            vertices = utils.remove_doubling_in_perp_space(edges)
            # get bond pairs, [[distance, XXX, YYY],...]
            pairs = []
            for edge in edges:
                dist=intsct.distance_in_perp_space(edge[0],edge[1])
                a=[dist]
                for i2 in range(2):
                    i3=0
                    for vrtx in vertices:
                        tmp=np.vstack([edge[i2],vrtx])
                        tmp=utils.remove_doubling_in_perp_space(tmp.reshape(2,6,3))
                        i3+=1
                        if len(tmp)==1:
                            a.append(i3)
                            break
                        else:
                            pass
                pairs.append(a)
                
            print('#VESTA_FORMAT_VERSION 3.5.0\n', file=f)
            print('MOLECULE\
            \nTITLE',file=f)
            print('%s/%s\n'%(path,basename), file=f)
            print('GROUP\
            \n1 1 Custom\
            \nSYMOP\
            \n 0.000000  0.000000  0.000000  1  0  0    0  1  0    0  0  1    1\
            \n -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0\
            \nTRANM 0\
            \n 0.000000  0.000000  0.000000  1  0  0    0  1  0    0  0  1\
            \nLTRANSL\
            \n -1\
            \n 0.000000  0.000000  0.000000  0.000000  0.000000  0.000000\
            \nLORIENT\
            \n -1    0    0    0    0\
            \n 1.000000  0.000000  0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000  0.000000  1.000000\
            \nLMATRIX\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000  0.000000\
            \nCELLP\
            \n  1.000000    1.000000    1.000000  90.000000  90.000000  90.000000\
            \n  0.000000    0.000000    0.000000    0.000000    0.000000    0.000000\
            \nSTRUC', file=f)
            i2=0
            for vrtx in vertices:
                xyz = math1.projection3(vrtx)
                print('%4d A        A%d  1.0000    %8.6f %8.6f %8.6f        1'%\
                (i2+1,i2+1,numericalc.numeric_value(xyz[0]),numericalc.numeric_value(xyz[1]),numericalc.numeric_value(xyz[2])), file=f)
                i2+=1
                print('                             0.000000    0.000000    0.000000  0.00', file=f)
            print('  0 0 0 0 0 0 0\
            \nTHERI 0', file = f)
            for i2 in range(len(vertices)):
                print('  %d        A%d  1.000000'%(i2+1,i2+1), file=f)
            print('  0 0 0\
            \nSHAPE\
            \n  0         0         0         0    0.000000  0    192    192    192    192\
            \nBOUND\
            \n         0          1        0          1        0          1\
            \n  0    0    0    0  0\
            \nSBOND', file = f)
            clr=colors(color)
            for pair in range(len(pairs)):
                print('  %d   A%d   A%d   %6.3f   %6.3f  0  1  1  1  2  0.250  2.000 %3d %3d %3d'%(\
                i2+1, pair[1]+1, pair[2]+1, pair[0]-0.01, pair[0]+0.01, clr[0], clr[1], clr[2]), file=f)
                i2+=1
            print('  0 0 0 0\
            \nSITET', file = f)
            for i2 in range(len(vertices)):
                print('    %d        A%d  0.100  76  76  76  76  76  76 204  0'%(i2+1,i2+1), file=f)
            print('  0 0 0 0 0 0\
            \nVECTR\
            \n 0 0 0 0 0\
            \nVECTT\
            \n 0 0 0 0 0\
            \nSPLAN\
            \n  0    0    0    0\
            \nLBLAT\
            \n -1\
            \nLBLSP\
            \n -1\
            \nDLATM\
            \n -1\
            \nDLBND\
            \n -1\
            \nDLPLY\
            \n -1\
            \nPLN2D\
            \n  0    0    0    0', file = f)
        
            print('ATOMT\
            \n  1        A  0.0100  76  76  76  76  76  76 204\
            \n  0 0 0 0 0 0\
            \nSCENE\
            \n-0.538344 -0.838391  0.085359  0.000000\
            \n-0.362057  0.138632 -0.921789  0.000000\
            \n 0.760986 -0.527145 -0.378177  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n  0.000    0.000\
            \n  0.000\
            \n  1.320\
            \nHBOND 0 2\
            \n\
            \nSTYLE\
            \nDISPF 37753794\
            \nMODEL    0  1  0\
            \nSURFS    0  1  1\
            \nSECTS  32  1\
            \nFORMS    0  1\
            \nATOMS    0  0  1\
            \nBONDS    2\
            \nPOLYS    1\
            \nVECTS 1.000000\
            \nFORMP\
            \n  1  1.0    0    0    0\
            \nATOMP\
            \n 24  24    0  50  2.0    0\
            \nBONDP\
            \n  1  16  0.250  2.000 127 127 127\
            \nPOLYP\
            \n 204 1  1.000 180 180 180\
            \nISURF\
            \n  0    0    0    0\
            \nTEX3P\
            \n  1  0.00000E+00  1.00000E+00\
            \nSECTP\
            \n  1  5.00000E-01  5.00000E-01  0.00000E+00  0.00000E+00  0.00000E+00  0.00000E+00\
            \nCONTR\
            \n 0.1 -1 1 1 10 -1 2 5\
            \n 2 1 2 1\
            \n    0    0    0\
            \n    0    0    0\
            \n    0    0    0\
            \n    0    0    0\
            \nHKLPP\
            \n 192 1  1.000 255    0 255\
            \nUCOLP\
            \n    0    1  1.000    0    0    0\
            \nCOMPS 0\
            \nLABEL 1     12  1.000 0\
            \nPROJT 0  0.962\
            \nBKGRC\
            \n 255 255 255\
            \nDPTHQ 1 -0.5000  3.5000\
            \nLIGHT0 1\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n  26  26  26 255\
            \n 179 179 179 255\
            \n 255 255 255 255\
            \nLIGHT1\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nLIGHT2\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nLIGHT3\
            \n 1.000000  0.000000  0.000000  0.000000\
            \n 0.000000  1.000000  0.000000  0.000000\
            \n 0.000000  0.000000  1.000000  0.000000\
            \n 0.000000  0.000000  0.000000  1.000000\
            \n 0.000000  0.000000 20.000000  0.000000\
            \n 0.000000  0.000000 -1.000000\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \n    0    0    0    0\
            \nATOMM\
            \n 204 204 204 255\
            \n  25.600\
            \nBONDM\
            \n 255 255 255 255\
            \n 128.000\
            \nPOLYM\
            \n 255 255 255 255\
            \n 128.000\
            \nSURFM\
            \n    0    0    0 255\
            \n 128.000\
            \nFORMM\
            \n 255 255 255 255\
            \n 128.000\
            \nHKLPM\
            \n 255 255 255 255\
            \n 128.000',file = f)
        
            f.close()
            #write_vesta_separate(obj, path, basename, color, dmax)
            if verbose>0:
                print('    written in %s'%(file_name))
            return vertices
    
    else:
        return 1

def write_xyz(obj,path='.',basename='tmp',select='triangle',verbose=0):
    """
    Export occupation domains in XYZ format.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        path (str): Path of the output XYZ file
        basename (str): Basename of the output XYZ file
        select (str)
            'triangle'   : set of triangles (default)
            'edge'       : set of edges
            'vertex'      : set of vertices
            (default, select = 'triangle')
    
    Returns:
        int: 0 (succeed), 1 (fail)
    """
    
    def generator_xyz_dim4_triangle(obj,filename):
        """
        Generate object (set of triangles) object in XYZ format.
    
        Args:
            obj (numpy.ndarray): the occupation domain
                The shape is (num,3,6,3), where num=numbre_of_triangle.
            filename (str): filename of the output XYZ file
        
        Returns:
            int: 0 (succeed), 1 (fail)
        
        """
        f=open('%s'%(filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)*3))
        f.write('%s\n'%(filename))
        i1=0
        for i1,triangle in enumerate(obj):
            for i2,vt in enumerate(triangle):
                v=math1.projection3(vt)
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the triangle %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                (numericalc.numeric_value(v[0]),\
                numericalc.numeric_value(v[1]),\
                numericalc.numeric_value(v[2]),\
                i1,i2,\
                vt[0][0],vt[0][1],vt[0][2],\
                vt[1][0],vt[1][1],vt[1][2],\
                vt[2][0],vt[2][1],vt[2][2],\
                vt[3][0],vt[3][1],vt[3][2],\
                vt[4][0],vt[4][1],vt[4][2],\
                vt[5][0],vt[5][1],vt[5][2]))
        v=utils.obj_area_6d(obj)
        f.write('volume = %d %d %d (%8.6f)\n'%(v[0],v[1],v[2],numericalc.numeric_value(v)))
        for i1,triangle in enumerate(obj):
            v=utils.triangle_area_6d(triangle)
            f.write('%3d-the triangle, %d %d %d (%8.6f)\n'\
                    %(i1,v[0],v[1],v[2],numericalc.numeric_value(v)))
        f.closed
        return 0
    
    def generator_xyz_dim4_edge(obj,filename):
        """
        Generate object (set of edges) object in XYZ format.
    
        Args:
            obj (numpy.ndarray): the occupation domain
                The shape is (num,2,6,3), where num=numbre_of_e.
            filename (str): filename of the output XYZ file
        
        Returns:
            int: 0 (succeed), 1 (fail)
        
        """
        f=open('%s'%(filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)*2))
        f.write('%s\n'%(filename))
        for i1,edge in enumerate(obj):
            for i2,vt in enumerate(edge):
                v=math1.projection3(vt)
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the edge %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                (numericalc.numeric_value(v[0]),\
                numericalc.numeric_value(v[1]),\
                numericalc.numeric_value(v[2]),\
                i1,i2,\
                vt[0][0],vt[0][1],vt[0][2],\
                vt[1][0],vt[1][1],vt[1][2],\
                vt[2][0],vt[2][1],vt[2][2],\
                vt[3][0],vt[3][1],vt[3][2],\
                vt[4][0],vt[4][1],vt[4][2],\
                vt[5][0],vt[5][1],vt[5][2]))
        f.closed
        return 0
    
    def generator_xyz_dim4_vertex(obj, filename):
        """
        Generate object (set of vertexs) object in XYZ format.
    
        Args:
            obj (numpy.ndarray): the occupation domain
                The shape is (num,3,6,3), where num=numbre_of_tetrahedron.
            filename (str): filename of the output XYZ file
        
        Returns:
            int: 0 (succeed), 1 (fail)
        
        """
        f=open('%s'%(filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)))
        f.write('%s\n'%(filename))
        for i1,point in enumerate(obj):
            v=math1.projection3(point)
            f.write('Xx %8.6f %8.6f %8.6f # %d-th vertex # # # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
            (numericalc.numeric_value(v[0]),\
            numericalc.numeric_value(v[1]),\
            numericalc.numeric_value(v[2]),\
            i1,\
            point[0][0],point[0][1],point[0][2],\
            point[1][0],point[1][1],point[1][2],\
            point[2][0],point[2][1],point[2][2],\
            point[3][0],point[3][1],point[3][2],\
            point[4][0],point[4][1],point[4][2],\
            point[5][0],point[5][1],point[5][2]))
        f.closed
        return 0
    
    if np.all(obj==None):
        print('empty obj')
        return 
    elif obj.ndim!=4:
        print('object has an incorrect shape!')
        return 
    else:
        file_name='%s/%s.xyz'%(path,basename)
        if select=='triangle':
            generator_xyz_dim4_triangle(obj, file_name)
            if verbose>0:
                print('    written in %s/%s.xyz'%(path,basename))
            return 0
        elif select=='edge':
            generator_xyz_dim4_edge(obj, file_name)
            if verbose>0:
                print('    written in %s/%s.xyz'%(path,basename))
            return 0
        elif select=='vertex':
            generator_xyz_dim4_vertex(obj, file_name)
            if verbose>0:
                print('    written in %s/%s.xyz'%(path,basename))
            return 0
        else:
            if verbose>0:
                print('    error')
            return 

def read_xyz(path,basename,select='triangle',verbose=0):
    """
    Load new occupation domain on input XYZ file.
    
    Args:
        path (str): Path of the input XYZ file
        basename (str): Basename of the input XYZ file
        select (str)
            'triangle'    : read as a set of triangles (default)
            'vertex'      : read as a set of vertices
            (default, select = 'triangle')
        verbose (int): verbose option
    Returns:
        Occupation domains (numpy.ndarray):
            Loaded occupation domains.
            The shape is (num,3,6,3), where num=numbre_of_triangles (select = 'triangle').
            The shape is (num,6,3), where num=numbre_of_vertices (select = 'vertex').
    """
    
    def read_file(file):
        try:
            f=open(file,'r')
        except IOError as e:
            print(e)
            sys.exit(0)
        line=[]
        while 1:
            a=f.readline()
            if not a:
                break
            line.append(a[:-1])
        return line
    
    filename='%s/%s.xyz'%(path,basename)
    
    f1=read_file(filename)
    f0=f1[0].split()
    num=int(f0[0])
    
    for i in range(2,num+2):
        fi=f1[i]
        fi=fi.split()
        a1=int(fi[10])
        b1=int(fi[11])
        c1=int(fi[12])
        a2=int(fi[13])
        b2=int(fi[14])
        c2=int(fi[15])
        a3=int(fi[16])
        b3=int(fi[17])
        c3=int(fi[18])
        a4=int(fi[19])
        b4=int(fi[20])
        c4=int(fi[21])
        a5=int(fi[22])
        b5=int(fi[23])
        c5=int(fi[24])
        a6=int(fi[25])
        b6=int(fi[26])
        c6=int(fi[27])
        if i==2:
            tmp=np.array([a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6])
        else:
            tmp=np.append(tmp,[a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6])
    if verbose>0:
        print('    read %s/%s.xyz'%(path,basename))
    
    if select == 'triangle':
        return tmp.reshape(int(num/3),3,6,3)
    elif select == 'vertex':
        return tmp.reshape(int(num),6,3)

def simplification(obj,verbose=0):
    """
    Simplification of occupation domains.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_trianges.
        num_cycle (int): numbre of cycles
        verbose (int)
            verbose = 0 (silent, default)
            verbose = 1 (normal)
            verbose > 2 (detail)
    
    Returns:
    
        Simplified occupation domains (numpy.ndarray)
            The shape is (num,3,6,3), where num=numbre_of_tetrahedron.
    
    """
    if np.all(obj==None):
        if verbose>0:
            print('    zero volume')
        return 
    else:
        vol0=utils.obj_area_6d(obj)
        obj_convex_hull=utils.generate_convex_hull(obj)
        obj_tmp=intsct.intersection_two_obj_1(obj_convex_hull,obj)
        vol1=utils.obj_area_6d(obj_tmp)
        if np.all(vol0==vol1):
            if verbose>0:
                print('      simplification succeed:')
                print('      num of tetrahedra: %d --> %d'%(len(obj),len(obj_convex_hull)))
            return obj_convex_hull
        else:
            if verbose>0:
                print('      simplification: fail')
            return obj

def generate_border_edges(obj):
    """
    Generate border edges of the occupation domain.
    
    Args:
        obj (numpy.ndarray):
            The occupation domain
            The shape is (num,3,6,3), where num=numbre_of_tetrahedron.
    
    Returns:
        Border edges of the occupation domains (numpy.ndarray):
            The shape is (num,2,6,3), where num=numbre_of_edge.
    
    """
    triangle_surface=utils.generator_surface_1(obj)
    return utils.surface_cleaner(triangle_surface)

def outline(obj):
    """
    Generate outline of the occupation domain.
    
    Args:
        obj (numpy.ndarray): the shape is (num,3,6,3), where num=numbre_of_triangle.
    
    Returns:
        Outline of the occupation domain (numpy.ndarray):
            The shape is (num,2,6,3), where num=number of the outlines.
    
    """
    return utils.surface_cleaner(obj)
    
# new in version 0.0.2a2
def obj2podatm(obj,serial_number=1,path='.',basename='tmp',shift=[0,0,0,0,0,0]):
    
    def find_common_vertex(obj):
        #Find common vertex of tetrahedra in obj).
        counter1=0
        for i1 in [0,1,2]:
            vtx1=obj[0][i1]
            xyz1=math1.projection3(vtx1)
            counter2=0
            for i2 in range(1,len(obj)):
                counter3=0
                for i3 in [0,1,2]:
                    xyz2=math1.projection3(obj[i2][i3])
                    if np.all(xyz1==xyz2):
                        counter3=1
                        break
                if counter3==1:
                    counter2+=1
                else:
                    break
            if counter2==len(obj)-1:
                counter1=1
                break
            else:
                pass
        if counter1!=0:
            return vtx1
        else:
            return 
    
    # common vertex
    vrtx0=find_common_vertex(obj)
    
    if np.all(vrtx0!=None):
        
        if os.path.exists(path) == False:
            os.makedirs(path)
        else:
            pass
        fatm=open('%s/%s.atm'%(path,basename),'w', encoding="utf-8", errors="ignore")
        fpod=open('%s/%s.pod'%(path,basename),'w', encoding="utf-8", errors="ignore")
        
        #--------
        #  atm
        #--------
        fatm.write('%d \'Em\' 1 %d 1 2.0 0. 0. 1.0 0. 0. 0.\n'%(serial_number,serial_number))
        
        vn=numericalc.numerical_vector(vrtx0)
        fatm.write('x=  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f\n'%(\
        vn[0],vn[1],vn[2],vn[3],vn[4],vn[5]))
        
        # generate a list of verices and remove the common vertex from it.
        vtxs=utils.remove_doubling_in_perp_space(obj)
        vtxs=utils.remove_vector(vtxs,vrtx0)
        
        #--------
        #  pod
        #--------
        fpod.write('%d %d %d \'comment\'\n'%(serial_number,len(vtxs),2))
        for vtx in vtxs:
            vn=numericalc.numerical_vector(vtx)
            fpod.write('ej=  %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f\n'%(\
            vn[0],vn[1],vn[2],vn[3],vn[4],vn[5]))
        #
        lst_indx=[]
        for triangle in obj:
            for vrtx1 in triangle:
                for i1 in range(len(vtxs)):
                    if np.all(vrtx1==vtxs[i1]):
                        lst_indx.append(i1+1) # add 1 to avoide index 0.
                        break
                    else:
                        pass
        fpod.write('nth= %d'%(len(vtxs)))
        for indx in lst_indx:
            fpod.write(' %d'%(indx))
        fpod.write('\n')
        
        return 0
    
    else:
        print('No common vertex found in the object. pod and atm cannot be created.')
        return 1

#########################
#          WIP          #
#########################
def simple_hand_step1(obj, path, basename_tmp):
    """
    Simplification of occupation domains by hand (step1).
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        path (str): path of the tmporal file
        basename_tmp (str): name for tmporal file.
    
    Returns:
    
        Tmporal occupation domains (numpy.ndarray):
            The shape is (num,3,6,3), where num=numbre_of_triangles.
    
    """
    def write_xyz_smpl(a, path, basename):
        f=open('%s'%(path)+'/%s.xyz'%(basename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(a)))
        f.write('%s\n'%(basename))
        for i1 in range(len(a)):
            xyz=math1.projection3(a[i1])
            f.write('Xx %8.6f %8.6f %8.6f # %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
            (numericalc.numeric_value(xyz),\
            numericalc.numeric_value(xyz),\
            numericalc.numeric_value(xyz),\
            i1,\
            a[i1][0][0],a[i1][0][1],a[i1][0][2],\
            a[i1][1][0],a[i1][1][1],a[i1][1][2],\
            a[i1][2][0],a[i1][2][1],a[i1][2][2],\
            a[i1][3][0],a[i1][3][1],a[i1][3][2],\
            a[i1][4][0],a[i1][4][1],a[i1][4][2],\
            a[i1][5][0],a[i1][5][1],a[i1][5][2]))
        f.closed
        return 0
        
    od1a=utils.remove_doubling_in_perp_space(obj)
    write_xyz_smpl(od1a, path, basename_tmp)
    print('written in %s'%(path)+'/%s.xyz'%(basename_tmp))
    print('open above XYZ file in vesta and make merge_list, and run simple_hand_step2()')
    return od1a

def simple_hand_step2(obj, merge_list):
    """
    Simplification of occupation domains by hand (step2).
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        merge_list (list[[int,int,int,int,],[],...,[]])
            A list containing lists of indices of vertices of triangle.
            The indices of vertices of triangle in temporal file obtaind
            by 'simple_hand_step1()'.
    
    Returns:
        Simplified occupation domains (numpy.ndarray):
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
    
    """
    
    def merge(obj,mylist):
        tmp1=np.array([obj[mylist[0]-1]])
        for i in range(1,len(mylist)):
            tmp2=obj[mylist[i]-1]
            tmp1=np.append(tmp1,tmp2)
        return tmp1.reshape(len(mylist),6,3)
    
    for i in range(len(merge_list)):
        tmp1=merge(obj,merge_list[i])
        od2=intsct.tetrahedralization_points(tmp1)
        if i==0:
            od1=np.array(od2)
        else:
            od1=np.vstack([od1,od2])
    return od1

def similarity(obj,m):
    """
    obj:
    m: order of similarity transformation
    """
    return symmetry.similarity_obj(obj,m)

def qcstrc(mystrc,path,basename,phason_matrix,nmax,origin_shift,option=0,verbose=0):
    """
    mystrc
    
    """
    
    objs=[]
    pos=[]
    atm=[]
    eshift=[]
    
    for strc in mystrc:
        obj1,wsite,atom,shift=strc
        
        ndim=5
        V0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # 1a, (0,0,0,0)
        
        #
        #print('\nVertex OD')
        #print('obj1.shape:',obj1.shape)
        num_coset=symmetry.coset(wsite,ndim)
        #num_wsym=symmetry.site_symmetry(wsite,ndim)
        #print('num_wsym:',num_wsym)
        #print('num_coset:',num_coset)
        tmp=symmetry.generator_obj_symmetric_obj(obj1,wsite)
        num=len(tmp)
        tmp=symmetry.generator_obj_symmetric_obj_specific_symop(tmp,V0,num_coset)
        objs1=tmp.reshape(len(num_coset),num,3,6,3)
        pos1=symmetry.generator_obj_symmetric_vector_specific_symop(wsite,V0,num_coset)
        #print('objs1.shape:',objs1.shape)
        #print('pos1.shape:',pos1.shape)
        #
        objs.append(objs1)
        pos.append(pos1)
        atm.append(atom)
        eshift.append(shift)
        
    
    
    
    if np.all(phason_matrix)==0:
        phason_matrix=None
    else:
        pass
    
    lst=[]
    a=numericalc.strc(objs,pos,phason_matrix,nmax,eshift,origin_shift,verbose)
    f=open('%s/%s.xyz'%(path,basename),'w', encoding="utf-8", errors="ignore")
    f.write('%d\n'%(len(a)))
    f.write('%s.xyz\n'%(basename))
    for b in a:
        if option==0:
            #print('b:',b)
            f.write('%s %8.6f %8.6f %8.6f\n'%(atm[int(b[1])],b[0][0],b[0][1],b[0][2]))
        elif option==1: # Eperp, x, y
            f.write('%s %8.6f %8.6f %8.6f # %3d %3d %3d %3d\n'%(atm[int(b[3])],b[0],b[1],b[2],b[4],b[5],b[6],b[7]))
        else:
            pass
    f.closed
    print('    written in %s/%s.xyz'%(path,basename))
    return 0

if __name__ == "__main__":
    
    test_dir='../../tests/dode/test'
    xyz_dir='../../../xyz/dode'
    # import asymmetric part of OD(occupation domain) located at origin,0,0,0,0,0,0.
    od_asym = read_xyz(path=xyz_dir,basename='od_vertex_asymmetric')
    print(od_asym)
    
    pos0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
    od_sym = symmetric(obj = od_asym, centre = pos0)
    write(obj=od_sym, path=test_dir, basename = 'od_sym', format='vesta', color = 'k')
    write(obj=od_sym, path=test_dir, basename = 'od_sym', format='vesta', color = 'k')
    
    # move STRT OD to a position 1 1 1 0 -1 0.
    #pos_b1=np.array([[ 1, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 1],[ 0, 0, 1]]) # b_1
    #strt_pos1=shift(obj = strt_sym, shift = pos_b1)
    #write(pod=strt_pos1, path='.', basename='obj_strt', format='xyz')
    #write(obj=strt_pos1, path='.', basename='obj_strt', format='vesta', color='b')
    
    # intersection of "asymmetric part of strt" and "strt at position pos_b1"
    #    flag = 0,    with rough intersection chacking (faster)
    #    flag = 1, without rough intersection chacking
    #twoODs=TWO_ODs(pod1=strt_asym, pod2=strt_pos1, path='.',filename='common.xyz',flag=0,verbose=0)
    #intersection=Intersection(pod1=tmp1.reshape(1,4,6,3), pod2=tmp2.reshape(1,4,6,3), path='.',filename='common.xyz',flag=0,verbose=0)
    #common_part=twoODs.intersection()
    
    # export common_part in VESTA formated file.
    #write(obj=common_part, path='.', basename='common', format='vesta', color='r')
    
    