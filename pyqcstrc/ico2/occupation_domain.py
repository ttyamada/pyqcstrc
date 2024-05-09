#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import timeit
import os
import sys
#sys.path.append('.')
import numpy as np

try:
    import pyqcstrc.ico2.math1 as math1
    import pyqcstrc.ico2.utils as utils
    import pyqcstrc.ico2.numericalc as numericalc
    import pyqcstrc.ico2.symmetry as symmetry
    import pyqcstrc.ico2.intsct as intsct
except ImportError:
    print('import error\n')

TAU=(1+np.sqrt(5))/2.0

def volume(obj):
    return utils.obj_volume_6d(obj)
    
def symmetric(obj,centre):
    """
    Generate symmterical occupation domain by symmetric elements of m-3-5 on the asymmetric unit.
    
    Args:
        obj (numpy.ndarray):
            Asymmetric unit of the occupation domain
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        centre (numpy.ndarray):
            6d coordinate of the symmetric centre.
            The shape is (6,3)
    
    Returns:
        Symmetric occupation domains (numpy.ndarray):
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
    
    """
    if obj.ndim==3 or obj.ndim==4:
        return symmetry.generator_obj_symmetric_tetrahedron(obj,centre)
    else:
        print('object has an incorrect shape!')
        return 
    
def symmetric_0(obj,centre,indx_symop):
    """
    Generate symmterical occupation domain by symmetric elements of m-3-5 on the asymmetric unit.
    
    Args:
        obj (numpy.ndarray):
            Asymmetric unit of the occupation domain
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        centre (numpy.ndarray):
            6d coordinate of the symmetric centre.
            The shape is (6,3)
    
    Returns:
        Symmetric occupation domains (numpy.ndarray):
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
    
    """
    if obj.ndim==3 or obj.ndim==4:
        return symmetry.generator_obj_symmetric_tetrahedron_0(obj,centre,indx_symop)
    else:
        print('object has an incorrect shape!')
        return 

def shift(obj,shift):
    """
    Shift the occupation domain.
    
    Args:
        obj (numpy.ndarray):
            The occupation domain
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        shift (numpy.ndarray):
            6d coordinate to which the occupation domain is shifted.
            The shape is (6,3)
        verbose (int):
            verbose = 0 (silent, default)
            verbose = 1 (normal)
    
    Returns:
        Shifted occupation domains (numpy.ndarray):
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
    
    """
    return utils.shift_object(obj, shift)

def write(obj, path='.',basename='tmp',format='xyz',color='k',verbose=0,select='tetrahedron'):
    """
    Export occupation domains.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        path (str): Path of the output XYZ file
        basename (str): Basename of the output XYZ file
        format (str): format of output file
            format = 'xyz' (default)
            format = 'vesta'
        color (str)
            one of the characters {'k','r','b','p'}, which are short-hand notations 
            for shades of black, red, blue, and pink, in case where 'vesta' format is
            selected (default, color = 'k').
    Returns:
        int: 0 (succeed), 1 (fail)
    
    """
    
    if os.path.exists(path)==False:
        os.makedirs(path)
    else:
        pass
        
    if np.all(obj==None):
        print('    Empty OD')
        return 1
    else:
        if format=='vesta' or format=='v' or format=='VESTA':
            write_vesta(obj, path, basename, color, select='normal', verbose=0)
        elif format == 'xyz':
            write_xyz(obj, path, basename, select, verbose)
        else:
            pass
        return 0
    
def write_vesta(obj,path='.',basename='tmp',color='k',select='normal',verbose=0):
    """
    Export occupation domains in VESTA format.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        path (str): Path of the output XYZ file
        basename (str): Basename of the output XYZ file
        color (str)
            one of the characters {'k','r','b','p'}, which are short-hand notations 
            for shades of black, red, blue, and pink, in case where 'vesta' format is
            selected (default, color = 'k').
        select (str):'simple' or 'normal'
            'simple': Merging tetrahedra into one single objecte
            'normal': Each tetrahedron is set as single objecte (large file)
            'podatm': same as 'simple' but return "vertices" necessary to input 
            (default, select = 'normal')
    Returns:
        int: 0 (succeed), 1 (fail) when select = 'simple' or 'normal'.
        ndarray: vertices, when select = 'podatm'.
    """
    
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
    
    if select=='simple':
        
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
            for i1 in range(len(edges)):
                dist=intsct.distance_in_perp_space(edges[i1][0],edges[i1][1])
                a=[dist]
                for i2 in range(2):
                    for i3 in range(len(vertices)):
                        tmp=np.vstack([edges[i1][i2],vertices[i3]])
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
            i2=0
            for vrtx in vertices:
                xyz = math1.projection3(vrtx)
                xyz=numericalc.numerical_vector(xyz)
                print('%4d A        A%d  1.0000    %8.6f %8.6f %8.6f        1'%\
                (i2+1,i2+1,xyz[0],xyz[1],xyz[2]), file=f)
                i2+=1
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
            i2=0
            for pair in pairs:
                print('  %d   A%d   A%d   %8.6f   %8.6f  0  1  1  1  2  0.250  2.000 %3d %3d %3d'%(\
                i2+1, pair[1]+1, pair[2]+1, pair[0]-0.01, pair[0]+0.01, clr[0], clr[1], clr[2]), file=f)
                i2+=1
            print('  0 0 0 0\
            \nSITET', file = f)
            i2=0
            for __ in vertices:
                print('    %d        A%d  0.100  76  76  76  76  76  76 204  0'%(i2+1,i2+1), file=f)
                i2+=1
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
            if verbose>0:
                print('    written in %s'%(file_name))
            return 0
        
    elif select=='normal':
        
        if np.all(obj==None):
            print('no volume obj')
            return 1
        else:
            print('#VESTA_FORMAT_VERSION 3.5.0\n', file=f)
            i1=0
            for obj1 in obj:
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
                i2=0
                for vertx in obj1:
                    xyz=math1.projection3(vertx)
                    xyz=numericalc.numerical_vector(xyz)
                    print('%4d Xx        Xx%d  1.0000    %8.6f %8.6f %8.6f        1'%\
                    (i2+1,i2+1,xyz[0],xyz[1],xyz[2]), file=f)
                    i2+=1
                    print('                             0.000000    0.000000    0.000000  0.00', file=f)
                print('  0 0 0 0 0 0 0\
                \nTHERI 0', file = f)
                i2=0
                for _ in obj1:
                    print('  %d        Xx%d  1.000000'%(i2+1,i2+1), file=f)
                    i2+=1
                print('  0 0 0\
                \nSHAPE\
                \n  0         0         0         0    0.000000  0    192    192    192    192\
                \nBOUND\
                \n         0          1        0          1        0          1\
                \n  0    0    0    0  0\
                \nSBOND', file = f)
                clr=colors(color)
                print('  1     Xx     Xx     0.00000     %3.2f  0  1  1  0  2  0.250  2.000 %3d %3d %3d'%(dmax,clr[0],clr[1],clr[2]), file=f)
                print('  0 0 0 0\
                \nSITET', file = f)
                i2=0
                for _ in obj1:
                    print('    %d        Xx%d  0.0100  76  76  76  76  76  76 204  0'%(i2+1,i2+1), file=f)
                    i2+=1
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
                i1+=1
            print('ATOMT\
            \n  1        Xx  0.0100  76  76  76  76  76  76 204\
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
            if verbose>0:
                print('    written in %s'%(file_name))
            return 0
    
    if select == 'podatm':
        
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
    
def write_xyz(obj,path='.',basename='tmp',select='tetrahedron',verbose=0):
    """
    Export occupation domains in XYZ format.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        path (str): Path of the output XYZ file
        basename (str): Basename of the output XYZ file
        select (str)
            'tetrahedron': set of tetrahedra (default)
            'triangle'   : set of triangles
            'edge'       : set of edges
            'vertex'      : set of vertices
            (default, select = 'tetrahedron')
    
    Returns:
        int: 0 (succeed), 1 (fail)
    """
    
    def generator_xyz_dim4_tetrahedron(obj,filename):
        """
        Generate object (set of tetrahedra) object in XYZ format.
    
         Args:
            obj (numpy.ndarray): the occupation domain
                The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
            filename (str): filename of the output XYZ file
        
        Returns:
            int: 0 (succeed), 1 (fail)
    
        """
            
        f=open('%s'%(filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)*4))
        f.write('%s\n'%(filename))
        i1=0
        for tetrahedron in obj:
            for i2 in range(4):
                v=math1.projection3(tetrahedron[i2])
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                (numericalc.numeric_value(v[0]),\
                numericalc.numeric_value(v[1]),\
                numericalc.numeric_value(v[2]),\
                i1,i2,\
                tetrahedron[i2][0][0],tetrahedron[i2][0][1],tetrahedron[i2][0][2],\
                tetrahedron[i2][1][0],tetrahedron[i2][1][1],tetrahedron[i2][1][2],\
                tetrahedron[i2][2][0],tetrahedron[i2][2][1],tetrahedron[i2][2][2],\
                tetrahedron[i2][3][0],tetrahedron[i2][3][1],tetrahedron[i2][3][2],\
                tetrahedron[i2][4][0],tetrahedron[i2][4][1],tetrahedron[i2][4][2],\
                tetrahedron[i2][5][0],tetrahedron[i2][5][1],tetrahedron[i2][5][2]))
            i1+=1
        vol=utils.obj_volume_6d(obj)
        f.write('volume = %d %d %d (%8.6f)\n'%(vol[0],vol[1],vol[2],numericalc.numeric_value(vol)))
        for i1 in range(len(obj)):
            v=utils.tetrahedron_volume_6d(tetrahedron)
            f.write('%3d-the tetrahedron, %d %d %d (%8.6f)\n'\
                    %(i1,v[0],v[1],v[2],numericalc.numeric_value(v)))
        f.close()
        return 0
    
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
        for triangle in obj:
            for i2 in range(3):
                v=math1.projection3(triangle[i2])
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the triangle %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                (numericalc.numeric_value(v[0]),\
                numericalc.numeric_value(v[1]),\
                numericalc.numeric_value(v[2]),\
                i1,i2,\
                triangle[i2][0][0],triangle[i2][0][1],triangle[i2][0][2],\
                triangle[i2][1][0],triangle[i2][1][1],triangle[i2][1][2],\
                triangle[i2][2][0],triangle[i2][2][1],triangle[i2][2][2],\
                triangle[i2][3][0],triangle[i2][3][1],triangle[i2][3][2],\
                triangle[i2][4][0],triangle[i2][4][1],triangle[i2][4][2],\
                triangle[i2][5][0],triangle[i2][5][1],triangle[i2][5][2]))
            i1+=1
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
        i1=0
        for edge in obj:
            for i2 in range(2):
                v=math1.projection3(edge[i2])
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the edge %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                (numericalc.numeric_value(v[0]),\
                numericalc.numeric_value(v[1]),\
                numericalc.numeric_value(v[2]),\
                i1,i2,\
                edge[i2][0][0],edge[i2][0][1],edge[i2][0][2],\
                edge[i2][1][0],edge[i2][1][1],edge[i2][1][2],\
                edge[i2][2][0],edge[i2][2][1],edge[i2][2][2],\
                edge[i2][3][0],edge[i2][3][1],edge[i2][3][2],\
                edge[i2][4][0],edge[i2][4][1],edge[i2][4][2],\
                edge[i2][5][0],edge[i2][5][1],edge[i2][5][2]))
            i1+=1
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
        i1=0
        for point in obj:
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
            i1=0
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
        if select == 'tetrahedron':
            generator_xyz_dim4_tetrahedron(obj, file_name)
            if verbose>0:
                print('    written in %s/%s.xyz'%(path,basename))
            return 0
        elif select == 'triangle':
            generator_xyz_dim4_triangle(obj, file_name)
            if verbose>0:
                print('    written in %s/%s.xyz'%(path,basename))
            return 0
        elif select == 'edge':
            generator_xyz_dim4_edge(obj, file_name)
            if verbose>0:
                print('    written in %s/%s.xyz'%(path,basename))
            return 0
        elif select == 'vertex':
            generator_xyz_dim4_vertex(obj, file_name)
            if verbose>0:
                print('    written in %s/%s.xyz'%(path,basename))
            return 0
        else:
            if verbose>0:
                print('    error')
            return 

def read_xyz(path,basename,select='tetrahedron',verbose=0):
    """
    Load new occupation domain on input XYZ file.
    
    Args:
        path (str): Path of the input XYZ file
        basename (str): Basename of the input XYZ file
        select (str)
            'tetrahedron': set of tetrahedra (default)
            'vertex'      : set of vertices
            (default, select = 'tetrahedron')
        verbose (int): verbose option
    Returns:
        Occupation domains (numpy.ndarray):
            Loaded occupation domains.
            The shape is (num,4,6,3), where num=numbre_of_tetrahedra (select = 'tetrahedron').
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
    
    if select == 'tetrahedron':
        return tmp.reshape(int(num/4),4,6,3)
    elif select == 'vertex':
        return tmp.reshape(int(num),6,3)

def simplification(obj,verbose=0):
    """
    Simplification of occupation domains.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        num_cycle (int): numbre of cycles
        verbose (int)
            verbose = 0 (silent, default)
            verbose = 1 (normal)
            verbose > 2 (detail)
    
    Returns:
    
        Simplified occupation domains (numpy.ndarray)
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
    
    """
    if np.all(obj==None):
        if verbose>0:
            print('    zero volume')
        return 
    else:
        vol0=utils.obj_volume_6d(obj)
        obj_convex_hull=utils.generate_convex_hull(obj)
        obj_tmp=intsct.intersection_two_obj_1(obj_convex_hull,obj)
        vol1=utils.obj_volume_6d(obj_tmp)
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
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
    
    Returns:
        Border edges of the occupation domains (numpy.ndarray):
            The shape is (num,2,6,3), where num=numbre_of_edge.
    
    """
    triangle_surface=utils.generator_surface_1(obj)
    return utils.surface_cleaner(triangle_surface)

#########################
#          WIP          #
#########################
def simple_hand_step1(obj, path, basename_tmp):
    """
    Simplification of occupation domains by hand (step1).
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        path (str): path of the tmporal file
        basename_tmp (str): name for tmporal file.
    
    Returns:
    
        Tmporal occupation domains (numpy.ndarray):
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
    
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
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        merge_list (list[[int,int,int,int,],[],...,[]])
            A list containing lists of indices of vertices of tetrahedron.
            The indices of vertices of tetrahedron in temporal file obtaind
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

def site_symmetry(wyckoff_position, centering, verbose=0):
    """
    Symmetry operators in the site symmetry group G and its left coset decomposition.
    
    Args:
        Wyckoff position (numpy.ndarray):
            6D coordinate.
            The shape is (6,3).
        centering:
            primitive lattice ('p')
            face-centered lattice ('f') and 
            body-centered lattice ('i')
        verbose (int)
    
    Returns:
        List of index of symmetry operators of the site symmetry group G (list):
            The symmetry operators leaves xyz identical.
        
        List of index of symmetry operators in the left coset representatives of the poibt group G (list):
            The symmetry operators generates equivalent positions of the site xyz.
    """
    def translation():
        """
        translational symmetry
        primitive type lattice is assumed
        """
        
        # under development
        if centering=='i':
            cop = symmetry.generator_equivalent_vec(np.array([[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2]]))
        elif centering=='f':
            cop = symmetry.generator_equivalent_vec(np.array([[1,0,2],[1,0,2],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]))
        else:
            pass
        symop=[]
        tmp=np.array([0,0,0,0,0,0])
        symop.append(tmp)
        for i1 in [-1,0,1]:
            for i2 in [-1,0,1]:
                for i3 in [-1,0,1]:
                    for i4 in [-1,0,1]:
                        for i5 in [-1,0,1]:
                            for i6 in [-1,0,1]:
                                tmp=np.array([i1,i2,i3,i4,i5,i6])
                                symop.append(tmp)
        return symop
    
    def remove_overlaps_in_a_list(l1):
        """
        Remove overlap elements in list with set method.
        
        Args:
            l1 (list):
        
        Returns:
            l2 (list)
        """
        tmp=set(l1)
        l2=list(tmp)
        l2.sort()
        return l2
    
    def find_overlaps(l1,l2):
        """
        find overlap or not btween list1 and list2.
        
        Args:
            l1 (list):
            l2 (list):
        
        Returns:
            0 (int): no intersection
            1 (int): intersection
        """
        l3=remove_overlaps_in_a_list(l1+l2)
        if len(l1)+len(l2)==len(l3): # no overlap
            return 0
        else:
            return 1
    
    symop=symmetry.icosasymop()
    traop=translation()
    
     # List of index of symmetry operators of the site symmetry group G.
     # The symmetry operators leaves xyz identical.
    list1=[]
    
    # List of index of symmetry operators which are not in the G.
    list2=[]
    
    pos=wyckoff_position
    a1=(pos[0][0]+TAU*pos[0][1])/pos[0][2]
    a2=(pos[1][0]+TAU*pos[1][1])/pos[1][2]
    a3=(pos[2][0]+TAU*pos[2][1])/pos[2][2]
    a4=(pos[3][0]+TAU*pos[3][1])/pos[3][2]
    a5=(pos[4][0]+TAU*pos[4][1])/pos[4][2]
    a6=(pos[5][0]+TAU*pos[5][1])/pos[5][2]
    xyz=np.array([a1,a2,a3,a4,a5,a6])
    
    xyzi=numericalc.projection_numerical(xyz[0],xyz[1],xyz[2],xyz[3],xyz[4],xyz[5])
    xi=xyzi[3]
    yi=xyzi[4]
    zi=xyzi[5]
    if verbose>0:
        print(' site coordinates: %3.2f %3.2f %3.2f %3.2f'%(xyz[0],xyz[1],xyz[2],xyz[3],xyz[4],xyz[5]))
        print('         in Epar : %5.3f %5.3f %5.3f'%(xyi[0],xyi[1],xyi[2]))
        print('         in Eperp: %5.3f %5.3f %5.3f'%(xyi[3],xyi[4],xyi[5]))
    else:
        pass
    
    for i2 in range(len(symop)):
        flag=0
        for i1 in range(len(traop)):
            xyz1=np.dot(symop[i2],xyz)
            xyz2=xyz1+traop[i1]
            a=numericalc12.projection_numerical(xyz2[0],xyz2[1],xyz2[2],xyz2[3],xyz2[4],xyz2[5])
            if abs(a[3]-xi)<EPS and abs(a[4]-yi)<EPS and abs(a[5]-zi)<EPS:
                list1.append(i2)
                flag+=1
                break
            else:
                pass
        if flag==0:
            list2.append(i2)
    
    list1_new=remove_overlaps_in_a_list(list1)
    list2_new=remove_overlaps_in_a_list(list2)
    
    if verbose>0:
        print('     multiplicity:',len(list1_new))
        print('    site symmetry:',list1_new)
    else:
        pass
    
    if int(len(symop)/len(list1_new))==1:
        list5=[0]
        if verbose>0:
            print('       left coset:',list5)
        else:
            pass
    
    else:
        # left coset decomposition:
        list4=[]
        for i2 in list2_new:
            list3=[]
            for i1 in list1_new:
                op1=np.dot(symop[i2],symop[i1])
                for i3 in range(len(symop)):
                    if np.all(op1==symop[i3]):
                        list3.append(i3)
                        break
                    else:
                        pass
            list4.append(list3)
        
        #print('----------------')
        #for i2 in range(len(list4)):
        #    print(list4[i2])
        #print('----------------')
        
        for i2 in range(len(list4)-1):
            a=list4[i2]
            b=[]
            d=[]
            list5=[0] # symmetry element of identity, symop[0]
            list5.append(list2_new[i2])
            i3=i2+1
            while i3<len(list4):
                b=list4[i3]
                if len(d)==0:
                    if find_overlaps(a,b)==0:
                        d=a+b
                        list5.append(list2_new[i3])
                    else:
                        pass
                else:
                    if find_overlaps(d,b)==0:
                        d=d+b
                        list5.append(list2_new[i3])
                    else:
                        pass
                i3+=1
            b=remove_overlaps_in_a_list(d)
            if int(len(symop)/len(list1_new))==len(list5):
                if verbose>0:
                    print('       left coset:',list5)
                else:
                    pass
                break
            else:
                pass
    
    return list1_new, list5

def write_podatm(obj, position, vlist, path='.', basename='tmp', shift=[0.0,0.0,0.0,0.0,0.0,0.0], verbose=0):
    """
    Generate pod and atom files.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        position (numpy.ndarray): 6D coordinates of the position of the occupation domain.
        vertices (list): 
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
    
    if obj==None:
        print('no volume obj')
        return 0
    else:
        fatm=open('%s/%s.atm'%(path,basename),'w', encoding="utf-8", errors="ignore")
        fpod=open('%s/%s.pod'%(path,basename),'w', encoding="utf-8", errors="ignore")
        
        v=obj
        
        shft=shift
        # .atm file
        for i in range(len(vlist)):
            a=v[vlist[i][0]-1]
            fatm.write('%d \'Em\' 1 %d 1 2.0 0. 0. 1.0 0. 0. 0.\n'%(i+1,i+1))
            fatm.write('x=  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f\n'%(\
            numericalc.numeric_value(position[0]),\
            numericalc.numeric_value(position[1]),\
            numericalc.numeric_value(position[2]),\
            numericalc.numeric_value(position[3]),\
            numericalc.numeric_value(position[4]),\
            numericalc.numeric_value(position[5])))
            fatm.write('xe1= 1. 0.  0. 0.  0. 0. u1=0.0            5f\n')
            fatm.write('xe2= 1. 0. -1. 0. -1. 0. u2=0.0            3f\n')
            fatm.write('xe3= 1. 0.  0. 0. -1. 0. u3=0.0            2f\n')
            fatm.write('xi=  %8.6f  %8.6f  %8.6f  %8.6f  %8.6f  %8.6f  v=1.0\n'%(\
            numericalc.numeric_value(a[0])+shft[0],\
            numericalc.numeric_value(a[1])+shft[1],\
            numericalc.numeric_value(a[2])+shft[2],\
            numericalc.numeric_value(a[3])+shft[3],\
            numericalc.numeric_value(a[4])+shft[4],\
            numericalc.numeric_value(a[5])+shft[5]))
            fatm.write('isyd=1\n')
        fatm.close()
    
        # .pod file
        fpod.write('nsymo=4 icent=1 brv=\'p\' io=%d\n'%(len(vlist)))
        fpod.write('symmetry operator\n')
        fpod.write('x,z,t,u,v,y            5f\n')
        fpod.write('-x,-y,-v,-u,-t,-z      2f\n')
        fpod.write('-v,-y,t,z,-u,-x        2f\n')
        fpod.write('y,z,x,v,-t,-u          3f\n')
        for i1 in range(len(vlist)):
            a=v[vlist[i1][0]-1]
            tmp2=[vlist[i1][1]-1]
            for i2 in range(2,len(vlist[i1])):
                counter=0
                for i3 in range(len(tmp2)):
                    if vlist[i1][i2]-1==tmp2[i3]:
                        counter+=1
                        break
                    else:
                        pass
                if counter==0:
                    tmp2.append(vlist[i1][i2]-1)
                else:
                    pass
            tmp1=[]
            for i2 in range(1,len(vlist[i1])):
                for i3 in range(len(tmp2)):
                    if vlist[i1][i2]-1==tmp2[i3]:
                        tmp1.append(i3)
                        break
                    else:
                        pass
            fpod.write('%d %d %d \'comment\'\n'%(i1+1,len(tmp2),2))
            for i2 in range(len(tmp2)):
                b=v[tmp2[i2]]
                fpod.write('ej=  %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f\n'%(\
                numericalc.numeric_value(b[0])-numericalc.numeric_value(a[0]),\
                numericalc.numeric_value(b[1])-numericalc.numeric_value(a[1]),\
                numericalc.numeric_value(b[2])-numericalc.numeric_value(a[2]),\
                numericalc.numeric_value(b[3])-numericalc.numeric_value(a[3]),\
                numericalc.numeric_value(b[4])-numericalc.numeric_value(a[4]),\
                numericalc.numeric_value(b[5])-numericalc.numeric_value(a[5])))
            fpod.write('nth= %d'%(int((len(vlist[i1])-1)/3)))
            for i2 in range(len(tmp1)):
                fpod.write(' %d'%(tmp1[i2]+1))
            fpod.write('\n100000000000000000000000000000000000000000000000000000000000\n')
            fpod.write('000000000000000000000000000000000000000000000000000000000000\n')
        fpod.close()
        print('    written in %s/%s.atm'%(path,basename))
        print('    written in %s/%s.pod'%(path,basename))
    return 0



def asymmetric(symmetric_obj, position, vecs):
    """
    Asymmetric part of occupation domain.
    
    Args:
        symmetric_obj (numpy.ndarray):
            Occupation domain of which the asymmetric part is calculated.
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        position (numpy.ndarray):
            6d coordinate of the site of which the occupation domain centres.
            The shape is (6,3)
        vecs (numpy.ndarray):
            Three vectors that defines the asymmetric part.
            The shape is (3,6,3)
    
    Returns:
        Asymmetric part of the occupation domains (numpy.ndarray):
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
    """
    v0 = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    vecs = math1.mul_vectors(vecs,np.array([5,0,1]))
    # vecs multiplied by [a,b,c], where [a,b,c]=(a+TAU*b)/c. 
    # [a,b,c] has to be defined so that the tetrahedron whose vertices are defined 
    # by v0, and vecs covers the asymmetric unit of the ocuppation domains.
    # (default) [a,b,c]=[5,0,1].
    aum = np.append(v0,vecs).reshape(1,4,6,3)
    aum = shift(aum,position)
    od_asym = intsct.intersection_two_obj_1(symmetric_obj,aum)
    
    return od_asym
    
if __name__ == "__main__":
    
    # import asymmetric part of STRT OD(occupation domain) located at origin,0,0,0,0,0,0.
    strt_asym = read_xyz(path='../xyz',basename='strt_aysmmetric')
    write(obj=strt_asym, path='.', basename = 'obj_seed', format='vesta', color = 'k')
    
    # generat STRT OD located at 0,0,0,0,0,0 by symmetric operations (m-3-5).
    pos0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
    strt = symmetric(asymmetric_part_obj = strt_asym, position = pos0)
    
    # move STRT OD to a position 1 1 1 0 -1 0.
    pos_b1=np.array([[ 1, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 1],[ 0, 0, 1]]) # b_1
    strt_pos1=shift(obj = strt_sym, shift = pos_b1)
    write(pod=strt_pos1, path='.', basename='obj_strt', format='xyz')
    write(obj=strt_pos1, path='.', basename='obj_strt', format='vesta', color='b')
    
    # intersection of "asymmetric part of strt" and "strt at position pos_b1"
    #    flag = 0,    with rough intersection chacking (faster)
    #    flag = 1, without rough intersection chacking
    #twoODs=TWO_ODs(pod1=strt_asym, pod2=strt_pos1, path='.',filename='common.xyz',flag=0,verbose=0)
    #intersection=Intersection(pod1=tmp1.reshape(1,4,6,3), pod2=tmp2.reshape(1,4,6,3), path='.',filename='common.xyz',flag=0,verbose=0)
    #common_part=twoODs.intersection()
    
    # export common_part in VESTA formated file.
    #write(obj=common_part, path='.', basename='common', format='vesta', color='r')
    
    