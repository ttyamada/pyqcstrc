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
    
except ImportError:
    print('import error\n')

TAU=(1+np.sqrt(5))/2.0

def volume(obj):
    w1,w2,w3=utils.obj_volume_6d(obj)
    return [w1,w2,w3]
    
def as_it_is(obj):
    """
    Returns an object as it is,
    
    Args:
        obj (numpy.ndarray): the shape is (num,4,6,3) or (num*4,6,3), where num=numbre_of_tetrahedron.
    
    Returns:
        Occupation domains (numpy.ndarray): the shape is (num,4,6,3), where num=numbre_of_tetrahedron.
    """
    
    if obj.ndim == 3:
        return obj.reshape(int(len(obj)/4),4,6,3)
    elif obj.ndim == 4:
        return obj
    else:
        return 1
    
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
    vecs = math1.mul_vectors(vecs,[5,0,1])
    # vecs multiplied by [a,b,c], where [a,b,c]=(a+TAU*b)/c. 
    # [a,b,c] has to be defined so that the tetrahedron whose vertices are defined 
    # by v0, and vecs covers the asymmetric unit of the ocuppation domains.
    # (default) [a,b,c]=[5,0,1].
    tmp = np.append(v0,vecs).reshape(4,6,3)
    aum = as_it_is(tmp)
    aum = shift(aum,position)
    #od_asym = ods.intersection(symmetric_obj, aum)
    od_asym = intsct.intersection_two_obj_1(symmetric_obj, aum, 0, 0)
    
    return od_asym
    
def symmetric(asymmetric_part_obj, position):
    """
    Generate symmterical occupation domain by symmetric elements of m-3-5 on the asymmetric unit.
    
    Args:
        asymmetric_part_obj (numpy.ndarray):
            Asymmetric unit of the occupation domain
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        position (numpy.ndarray):
            6d coordinate of the site of which the occupation domain centres.
            The shape is (6,3)
    
    Returns:
        Symmetric occupation domains (numpy.ndarray):
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
    
    """
    if asymmetric_part_obj.ndim == 3:
        return symmetry.generator_obj_symmetric_tetrahedron(asymmetric_part_obj, position)
    elif asymmetric_part_obj.ndim == 4:
        asymmetric_part_obj=asymmetric_part_obj.reshape(len(asymmetric_part_obj)*4,6,3)
        return symmetry.generator_obj_symmetric_tetrahedron(asymmetric_part_obj, position)
    elif asymmetric_part_obj.ndim == 2:
        asymmetric_part_obj=asymmetric_part_obj.reshape(1,6,3)
        a=symmetry.generator_obj_symmetric_tetrahedron(asymmetric_part_obj, position)
        return a.reshape(4*len(a),6,3)
    else:
        return 1
    
def shift(obj, shift, verbose = 0):
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
    return utils.shift_object(obj, shift, verbose)

def generate_edges(obj, verbose = 0):
    """
    Generate edges of the occupation domain.
    
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
        Edges of the occupation domains (numpy.ndarray):
            The shape is (num,2,6,3), where num=numbre_of_edge.
    
    """
    obj_surface = utils.generator_surface_1(obj,verbose)
    #obj_surface = mics.generator_surface(obj)
    #return utils.generator_edge(obj_surface,verbose)
    return utils.generator_edge_1(obj_surface,verbose)
    
def gen_surface(obj,verbose = 0):
    """
    Generate triangles on the surface of the occupation domain.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        verbose (int):
            verbose = 0 (silent, default)
            verbose = 1 (normal)
    Returns:
        Triangles of the occupation domains (numpy.ndarray):
            The shape is (num,3,6,3), where num=numbre_of_triangles.
    
    """
    return utils.generator_surface_1(obj,verbose)
    
def write(obj, path = '.', basename = 'tmp', format = 'xyz', color = 'k', verbose = 0, select = 'tetrahedron'):
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
    
    if os.path.exists(path) == False:
        os.makedirs(path)
    else:
        pass
        
    if obj.tolist()==[[[[0]]]]:
        print('    Empty OD')
        return 1
    else:
        if format == 'vesta' or format == 'v' or format == 'VESTA':
            write_vesta(obj, path, basename, color, select = 'normal', verbose = 0)
        elif format == 'xyz':
            write_xyz(obj, path, basename, select, verbose)
        else:
            pass
        return 0
    
def write_vesta(obj, path = '.', basename = 'tmp', color = 'k', select = 'normal', verbose = 0):
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
    
    if os.path.exists(path) == False:
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
    
    file_name = '%s/%s.vesta'%(path,basename)
    f = open('%s'%(file_name),'w')
    
    dmax = 5.0
    
    if select == 'simple':
        
        if obj.tolist()==[[[[0]]]] or obj.tolist()==[[[0]]]:
            print('no volume obj')
            return 0
        
        else:
            # get independent edges
            edges = utils.generator_obj_edge(obj, verbose)
            # get independent vertices of the edges
            vertices = utils.remove_doubling_dim4_in_perp_space(edges)
        
            # get bond pairs, [[distance, XXX, YYY],...]
            pairs = []
            #print(len(edges))
            for i1 in range(len(edges)):
                dist=intsct.distance_in_perp_space(edges[i1][0],edges[i1][1])
                a=[dist]
                for i2 in range(2):
                    for i3 in range(len(vertices)):
                        tmp=np.vstack([edges[i1][i2],vertices[i3]])
                        tmp=utils.remove_doubling_dim3_in_perp_space(tmp.reshape(2,6,3))
                        #tmp=utils.remove_doubling_dim3_in_perp_space(tmp)
                        if len(tmp)==1:
                            a.append(i3)
                            break
                        else:
                            pass
                pairs.append(a)
                #print(i1)
            #print(pairs)
        
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
        
            for i2 in range(len(vertices)):
                a4,a5,a6 = math1.projection3(vertices[i2][0],vertices[i2][1],vertices[i2][2],vertices[i2][3],vertices[i2][4],vertices[i2][5])
                print('%4d A        A%d  1.0000    %8.6f %8.6f %8.6f        1'%\
                (i2+1,i2+1,(a4[0]+a4[1]*TAU)/a4[2],(a5[0]+a5[1]*TAU)/a5[2],(a6[0]+a6[1]*TAU)/a6[2]), file=f)
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
            for i2 in range(len(pairs)):
                print('  %d   A%d   A%d   %8.6f   %8.6f  0  1  1  1  2  0.250  2.000 %3d %3d %3d'%(\
                i2+1, pairs[i2][1]+1, pairs[i2][2]+1, pairs[i2][0]-0.1, pairs[i2][0]+0.1, clr[0], clr[1], clr[2]), file=f)
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
            return 0
        
    elif select == 'normal':
        
        if obj.tolist()==[[[[0]]]] or obj.tolist()==[[[0]]]:
            print('no volume obj')
            return 1
        else:
            print('#VESTA_FORMAT_VERSION 3.5.0\n', file=f)
            for i1 in range(len(obj)):
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
                for i2 in range(len(obj[i1])):
                    a4,a5,a6 = math1.projection3(obj[i1][i2][0],obj[i1][i2][1],obj[i1][i2][2],obj[i1][i2][3],obj[i1][i2][4],obj[i1][i2][5])
                    print('%4d Xx        Xx%d  1.0000    %8.6f %8.6f %8.6f        1'%\
                    (i2+1,i2+1,(a4[0]+a4[1]*TAU)/a4[2],(a5[0]+a5[1]*TAU)/a5[2],(a6[0]+a6[1]*TAU)/a6[2]), file=f)
                    print('                             0.000000    0.000000    0.000000  0.00', file=f)
                print('  0 0 0 0 0 0 0\
                \nTHERI 0', file = f)
                for i2 in range(len(obj[i1])):
                    print('  %d        Xx%d  1.000000'%(i2+1,i2+1), file=f)
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
                for i2 in range(len(obj[i1])):
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
                \n  0    0    0    0', file = f)
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
            #write_vesta_separate(obj, path, basename, color, dmax)
            if verbose>0:
                print('    written in %s'%(file_name))
            return 0
    
    if select == 'podatm':
        
        if obj.tolist()==[[[[0]]]] or obj.tolist()==[[[0]]]:
            print('no volume obj')
            return 0
        else:
            #"""
            # get independent edges
            edges = utils.generator_obj_edge(obj, verbose)
            # get independent vertices of the edges
            vertices = utils.remove_doubling_dim4_in_perp_space(edges)
            #"""
            # get bond pairs, [[distance, XXX, YYY],...]
            pairs = []
            #print(len(edges))
            for i1 in range(len(edges)):
                dist=intsct.distance_in_perp_space(edges[i1][0],edges[i1][1])
                a=[dist]
                for i2 in range(2):
                    for i3 in range(len(vertices)):
                        tmp=np.vstack([edges[i1][i2],vertices[i3]])
                        tmp=utils.remove_doubling_dim3_in_perp_space(tmp.reshape(2,6,3))
                        #tmp=utils.remove_doubling_dim3_in_perp_space(tmp)
                        if len(tmp)==1:
                            a.append(i3)
                            break
                        else:
                            pass
                pairs.append(a)
                #print(i1)
            #print(pairs)
        
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
        
            for i2 in range(len(vertices)):
                a4,a5,a6 = math1.projection3(vertices[i2][0],vertices[i2][1],vertices[i2][2],vertices[i2][3],vertices[i2][4],vertices[i2][5])
                print('%4d A        A%d  1.0000    %8.6f %8.6f %8.6f        1'%\
                (i2+1,i2+1,(a4[0]+a4[1]*TAU)/a4[2],(a5[0]+a5[1]*TAU)/a5[2],(a6[0]+a6[1]*TAU)/a6[2]), file=f)
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
            for i2 in range(len(pairs)):
                print('  %d   A%d   A%d   %8.6f   %8.6f  0  1  1  1  2  0.250  2.000 %3d %3d %3d'%(\
                i2+1, pairs[i2][1]+1, pairs[i2][2]+1, pairs[i2][0]-0.1, pairs[i2][0]+0.1, clr[0], clr[1], clr[2]), file=f)
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
    
def write_xyz(obj, path = '.', basename = 'tmp', select = 'tetrahedron', verbose = 0):
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
    
    def generator_xyz_dim4_tetrahedron(obj, filename):
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
        for i1 in range(len(obj)):
            for i2 in range(4):
                a4,a5,a6=math1.projection3(obj[i1][i2][0],\
                                            obj[i1][i2][1],\
                                            obj[i1][i2][2],\
                                            obj[i1][i2][3],\
                                            obj[i1][i2][4],\
                                            obj[i1][i2][5])
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                ((a4[0]+a4[1]*TAU)/(a4[2]),\
                (a5[0]+a5[1]*TAU)/(a5[2]),\
                (a6[0]+a6[1]*TAU)/(a6[2]),\
                i1,i2,\
                obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
                obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
                obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
                obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
                obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
                obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2]))
        w1,w2,w3=utils.obj_volume_6d(obj)
        f.write('volume = %d %d %d (%8.6f)\n'%(w1,w2,w3,(w1+TAU*w2)/(w3)))
        for i1 in range(len(obj)):
            [v1,v2,v3]=utils.tetrahedron_volume_6d(obj[i1])
            f.write('%3d-the tetrahedron, %d %d %d (%8.6f)\n'\
                    %(i1,v1,v2,v3,(v1+TAU*v2)/(v3)))
        f.close()
        return 0

    def generator_xyz_dim4_triangle(obj, filename):
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
        for i1 in range(len(obj)):
            for i2 in range(3):
                a4,a5,a6=math1.projection3(obj[i1][i2][0],\
                                            obj[i1][i2][1],\
                                            obj[i1][i2][2],\
                                            obj[i1][i2][3],\
                                            obj[i1][i2][4],\
                                            obj[i1][i2][5])
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the triangle %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                ((a4[0]+a4[1]*TAU)/(a4[2]),\
                (a5[0]+a5[1]*TAU)/(a5[2]),\
                (a6[0]+a6[1]*TAU)/(a6[2]),\
                i1,i2,\
                obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
                obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
                obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
                obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
                obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
                obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2]))
        f.closed
        return 0
    
    def generator_xyz_dim4_edge(obj, filename):
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
        for i1 in range(len(obj)):
            for i2 in range(2):
                a4,a5,a6=math1.projection3(obj[i1][i2][0],\
                                            obj[i1][i2][1],\
                                            obj[i1][i2][2],\
                                            obj[i1][i2][3],\
                                            obj[i1][i2][4],\
                                            obj[i1][i2][5])
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the edge %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                ((a4[0]+a4[1]*TAU)/(a4[2]),\
                (a5[0]+a5[1]*TAU)/(a5[2]),\
                (a6[0]+a6[1]*TAU)/(a6[2]),\
                i1,i2,\
                obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
                obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
                obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
                obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
                obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
                obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2]))
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
        for i1 in range(len(obj)):
            a4,a5,a6=math1.projection3(obj[i1][0],\
                                        obj[i1][1],\
                                        obj[i1][2],\
                                        obj[i1][3],\
                                        obj[i1][4],\
                                        obj[i1][5])
            f.write('Xx %8.6f %8.6f %8.6f # %d-th vertex # # # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
            ((a4[0]+a4[1]*TAU)/(a4[2]),\
            (a5[0]+a5[1]*TAU)/(a5[2]),\
            (a6[0]+a6[1]*TAU)/(a6[2]),\
            i1,\
            obj[i1][0][0],obj[i1][0][1],obj[i1][0][2],\
            obj[i1][1][0],obj[i1][1][1],obj[i1][1][2],\
            obj[i1][2][0],obj[i1][2][1],obj[i1][2][2],\
            obj[i1][3][0],obj[i1][3][1],obj[i1][3][2],\
            obj[i1][4][0],obj[i1][4][1],obj[i1][4][2],\
            obj[i1][5][0],obj[i1][5][1],obj[i1][5][2]))
        f.closed
        return 0
    
    if obj.tolist()==[[[[0]]]] or obj.tolist()==[[[0]]]:
        print('no volume obj')
        return 0
    
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
            return 1

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

def write_podatm(obj, position, vlist = [0], path = '.', basename = 'tmp', shift=[0.0,0.0,0.0,0.0,0.0,0.0], verbose = 0):
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
    
    if obj.tolist()==[[[[0]]]] or obj.tolist()==[[[0]]]:
        print('no volume obj')
        return 0
    else:
        fatm=open('%s/%s.atm'%(path,basename),'w', encoding="utf-8", errors="ignore")
        fpod=open('%s/%s.pod'%(path,basename),'w', encoding="utf-8", errors="ignore")
    
        """
        # get independent edges
        edges = utils.generator_obj_edge(obj, verbose-1)
        # get independent vertices of the edges
        v = utils.remove_doubling_dim4_in_perp_space(edges)
        """
        #v=vertices
        v=obj
    
        # shift
        #shft=[0.00001,0.00002,0.00000,-0.00001,0.00001,-0.00002]
        #shft=[0.0,0.0,0.0,0.0,0.0,0.0]
        shft=shift
        # .atm file
        for i in range(len(vlist)):
            a=v[vlist[i][0]-1]
            fatm.write('%d \'Em\' 1 %d 1 2.0 0. 0. 1.0 0. 0. 0.\n'%(i+1,i+1))
            fatm.write('x=  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f\n'%(\
            (position[0][0]+position[0][1]*TAU)/(position[0][2]),\
            (position[1][0]+position[1][1]*TAU)/(position[1][2]),\
            (position[2][0]+position[2][1]*TAU)/(position[2][2]),\
            (position[3][0]+position[3][1]*TAU)/(position[3][2]),\
            (position[4][0]+position[4][1]*TAU)/(position[4][2]),\
            (position[5][0]+position[5][1]*TAU)/(position[5][2])))
            fatm.write('xe1= 1. 0.  0. 0.  0. 0. u1=0.0            5f\n')
            fatm.write('xe2= 1. 0. -1. 0. -1. 0. u2=0.0            3f\n')
            fatm.write('xe3= 1. 0.  0. 0. -1. 0. u3=0.0            2f\n')
            fatm.write('xi=  %8.6f  %8.6f  %8.6f  %8.6f  %8.6f  %8.6f  v=1.0\n'%(\
            (a[0][0]+a[0][1]*TAU)/(a[0][2])+shft[0],\
            (a[1][0]+a[1][1]*TAU)/(a[1][2])+shft[1],\
            (a[2][0]+a[2][1]*TAU)/(a[2][2])+shft[2],\
            (a[3][0]+a[3][1]*TAU)/(a[3][2])+shft[3],\
            (a[4][0]+a[4][1]*TAU)/(a[4][2])+shft[4],\
            (a[5][0]+a[5][1]*TAU)/(a[5][2])+shft[5]))
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
            #print('tmp2=',tmp2)
        
            tmp1=[]
            for i2 in range(1,len(vlist[i1])):
                for i3 in range(len(tmp2)):
                    if vlist[i1][i2]-1==tmp2[i3]:
                        tmp1.append(i3)
                        break
                    else:
                        pass
            #print('tmp1=',tmp1)
        
            fpod.write('%d %d %d \'comment\'\n'%(i1+1,len(tmp2),2))
            for i2 in range(len(tmp2)):
                b=v[tmp2[i2]]
                fpod.write('ej=  %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f\n'%(\
                (b[0][0]+b[0][1]*TAU)/(b[0][2])-(a[0][0]+a[0][1]*TAU)/(a[0][2]),\
                (b[1][0]+b[1][1]*TAU)/(b[1][2])-(a[1][0]+a[1][1]*TAU)/(a[1][2]),\
                (b[2][0]+b[2][1]*TAU)/(b[2][2])-(a[2][0]+a[2][1]*TAU)/(a[2][2]),\
                (b[3][0]+b[3][1]*TAU)/(b[3][2])-(a[3][0]+a[3][1]*TAU)/(a[3][2]),\
                (b[4][0]+b[4][1]*TAU)/(b[4][2])-(a[4][0]+a[4][1]*TAU)/(a[4][2]),\
                (b[5][0]+b[5][1]*TAU)/(b[5][2])-(a[5][0]+a[5][1]*TAU)/(a[5][2])))
                """
                for i3 in range(6):
                    if i3==0:
                        fpod.write('ej=  %8.6f'%(\
                        (b[i3][0]+b[i3][1]*TAU)/(b[i3][2]) - (a[i3][0]+a[i3][1]*TAU)/(a[i3][2])\
                        )
                    elif i3==5:
                        fpod.write(' %8.6f\n'%(\
                        (b[i3][0]+b[i3][1]*TAU)/(b[i3][2]) - (a[i3][0]+a[i3][1]*TAU)/(a[i3][2])\
                        )
                    else:
                        fpod.write(' %8.6f'%(\
                        (b[i3][0]+b[i3][1]*TAU)/(b[i3][2]) - (a[i3][0]+a[i3][1]*TAU)/(a[i3][2])\
                        )
                """
            fpod.write('nth= %d'%(int((len(vlist[i1])-1)/3)))
            for i2 in range(len(tmp1)):
                fpod.write(' %d'%(tmp1[i2]+1))
            fpod.write('\n100000000000000000000000000000000000000000000000000000000000\n')
            fpod.write('000000000000000000000000000000000000000000000000000000000000\n')
        fpod.close()
    
        if verbose>0:
            print('    written in %s/%s.atm'%(path,basename))
            print('    written in %s/%s.pod'%(path,basename))
    
    return 0

def read_xyz(path, basename, select = 'tetrahedron'):
    """
    Load new occupation domain on input XYZ file.
    
    Args:
        path (str): Path of the input XYZ file
        basename (str): Basename of the input XYZ file
        select (str)
            'tetrahedron': set of tetrahedra (default)
            'vertex'      : set of vertices
            (default, select = 'tetrahedron')
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
    print('    read %s/%s.xyz'%(path,basename))
    
    if select == 'tetrahedron':
        return tmp.reshape(int(num/4),4,6,3)
    elif select == 'vertex':
        return tmp.reshape(int(num),6,3)
    
def simplification(obj, num_cycle = 10, verbose = 0):
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
    if obj.tolist()!=[[[[0]]]]:
        n1,n2,n3=utils.obj_volume_6d(obj)
        obj1=mics.generate_convex_hull(obj, np.array([[0]]), num_cycle, verbose-1)
        obj2=intsct.intersection_two_obj_2(obj1, obj, verbose-1)
        m1,m2,m3=utils.obj_volume_6d(obj2)
        if verbose>0:
            print(' volume of original obj  : %d %d %d (%10.8f)'%(n1,n2,n3,(n1+TAU*n2)/n3))
            print(' volume of simplified obj: %d %d %d (%10.8f)'%(m1,m2,m3,(m1+TAU*m2)/m3))
        else:
            pass
        if n1==m1 and n2==m2 and n3==m3:
            print(' simplification: succeed')
            return obj2
        else:
            print(' simplification: fail')
            print(' return initial obj')
            #return np.array([[[[0]]]])
            return obj
    else:
        print(' zero volume')
        return np.array([[[[0]]]])

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
            a4,a5,a6=math1.projection3(a[i1][0],\
                                        a[i1][1],\
                                        a[i1][2],\
                                        a[i1][3],\
                                        a[i1][4],\
                                        a[i1][5])
            f.write('Xx %8.6f %8.6f %8.6f # %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
            ((a4[0]+a4[1]*TAU)/(a4[2]),\
            (a5[0]+a5[1]*TAU)/(a5[2]),\
            (a6[0]+a6[1]*TAU)/(a6[2]),\
            i1,\
            a[i1][0][0],a[i1][0][1],a[i1][0][2],\
            a[i1][1][0],a[i1][1][1],a[i1][1][2],\
            a[i1][2][0],a[i1][2][1],a[i1][2][2],\
            a[i1][3][0],a[i1][3][1],a[i1][3][2],\
            a[i1][4][0],a[i1][4][1],a[i1][4][2],\
            a[i1][5][0],a[i1][5][1],a[i1][5][2]))
        f.closed
        return 0
    od1a = utils.remove_doubling_dim4_in_perp_space(obj)
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
        od2=intsct.tetrahedralization_points(tmp1,0)
        if i==0:
            od1=np.array(od2)
        else:
            od1=np.vstack([od1,od2])
    return od1

#  legacy simple
def simple(obj, select, num_cycle = 3, verbose = 0, num_cycle_234 = [3,0,0], num_shuffle_234 = [1,0,0]):
    """
    Legacy: simple
    """
    if select == 0:
        obj_new = mics.simplification_obj_edges(obj, num_cycle, verbose)
    elif select == 1:
        obj_new = mics.simplification_obj_edges_1(obj, num_cycle, verbose)
    elif select == 2:
        obj_new=mics.simplification_obj_smart(obj, num_cycle, verbose)
    elif select == 3:
        obj_new = mics.simplification_convex_polyhedron(obj, num_cycle, verbose, 0)
    elif select == 4:
        [num2_of_cycle, num3_of_cycle, num4_of_cycle] = num_cycle_234
        [num2_of_shuffle, num3_of_shuffle, num4_of_shuffle] = num_shuffle_234
        obj_new = mics.simplification(obj,\
                                    num2_of_cycle,\
                                    num3_of_cycle,\
                                    num4_of_cycle,\
                                    num2_of_shuffle,\
                                    num3_of_shuffle,\
                                    num4_of_shuffle,\
                                    verbose)
    else:
        obj_new=mics.simplification_obj_smart(obj, num_cycle, verbose)
    return obj_new

def simple_special(obj, num, num_cycle, verbose_level = 0):
    """
    Legacy: simple_special
    """
    # 複雑に四面体分割されたObjectを単純化する
    # 一部分（ただし、凸多面体に限る）だけ単純化したい場合．
    # Parameters:
    #  obj: 単純化したいobject(dim4のnumpy.ndarray)
    #  num: 凸多面体を構成する四面体の配列インデックスをnumで指定(dim2のlist)
    #  num_cycle: qcmodel.simplification_convex_polyhedronのサイクル
    #  verbose_level: qcmodel.simplification_convex_polyhedronのverbose_level
    n1,n2,n3=utils.obj_volume_6d(obj)
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
        obj_new=obj_new.reshape(int(len(obj_new)/72),4,6,3)
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
        tmp=mics.simplification_convex_polyhedron(tmp.reshape(int(len(tmp)/72),4,6,3),num_cycle,verbose_level,0)
        if len(obj_new)==1:
            obj_new=tmp.reshape(len(tmp)*72)
        else:
            obj_new=np.append(obj_new,tmp)
    obj_new=obj_new.reshape(int(len(obj_new)/72),4,6,3)
    m1,m2,m3=utils.obj_volume_6d(obj_new)
    if verbose_level>0:
         print(' volume of original obj  : %d %d %d (%10.8f)'%(n1,n2,n3,(n1+TAU*n2)/n3))
         print(' volume of simplified obj: %d %d %d (%10.8f)'%(m1,m2,m3,(m1+TAU*m2)/m3))
    if n1==m1 and n2==m2 and n3==m3:
        print(' succeed')
        return obj_new
    else:
        print(' fail, original obj returned.')
        return obj

def simple_special_1(obj, num, num_cycle, verbose_level = 0):
    """
    Legacy: simple
    """
    # 複雑に四面体分割されたObjectを単純化する
    # 一部分（だけ単純化したい場合．
    # Parameters:
    #  obj: 単純化したいobject(dim4のnumpy.ndarray)
    #  num: 多面体を構成する四面体の配列インデックスをnumで指定(dim2のlist)
    #  num_cycle: qcmodel.simplification_convex_polyhedronのサイクル
    #  verbose_level: qcmodel.simplification_obj_edges()のverbose_level
    n1,n2,n3=utils.obj_volume_6d(obj)
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
        obj_new=obj_new.reshape(int(len(obj_new)/72),4,6,3)
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
        #tmp=mics.simplification_obj_edges(tmp.reshape(len(tmp)/72,4,6,3),num_cycle,verbose_level-1)
        tmp=mics.simplification_obj_edges_1(tmp.reshape(int(len(tmp)/72),4,6,3),num_cycle,verbose_level-1)
        if len(obj_new)==1:
            obj_new=tmp.reshape(len(tmp)*72)
        else:
            obj_new=np.append(obj_new,tmp)
    obj_new=obj_new.reshape(int(len(obj_new)/72),4,6,3)
    m1,m2,m3=utils.obj_volume_6d(obj_new)
    if verbose_level>0:
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

def simple_rondom(obj, num_cycle, combination_num, verbose_level = 0):
    """
    Legacy: simple
    """
    
    if verbose_level>0:
        print('  simplification_rondom()')
    else:
        pass
        
    n1,n2,n3=utils.obj_volume_6d(obj)
    obj_new=np.array([0])
    obj_tmp=obj
    for i in range(num_cycle):
        print('--------------')
        print('     %d-cycle'%(i))
        print('--------------')
        num=random.sample(range(len(obj_tmp)),combination_num)
        #print     num
        num=[num]
        obj_new=simple_special_1(obj_tmp,num,2,verbose_level-1)
        if len(obj_new)<len(obj_tmp):
            obj_tmp=obj_new
            if verbose_level>0:
                print('  reduced')
            else:
                pass
        else:
            pass
   
    m1,m2,m3=utils.obj_volume_6d(obj_new)
    if verbose_level>0:
        print(' volume of original obj  : %d %d %d (%10.8f)'%(n1,n2,n3,(n1+TAU*n2)/n3))
        print(' volume of simplified obj: %d %d %d (%10.8f)'%(m1,m2,m3,(m1+TAU*m2)/m3))
    if n1==m1 and n2==m2 and n3==m3:
        print(' succeed')
        return obj_new
    else:
        print(' fail, original obj returned.')
        return obj

def simpl_add_point(obj, point, verbose = 0):
    """
    Legacy: simple
    """
    # Parameters:
    #  obj: 単純化したいobject(dim4のnumpy.ndarray) ただし、凸多面体のみ
    #  point: obj内に追加したい点の6次元座標(dim2のnumpy.ndarray)
    obj_new=intsct.tetrahedralization_1(obj, point, verbose)
    return obj_new
    
def simpl_manual(obj, num, coordinates, verbose_level = 0):
    """
    Legacy: simple
    """
    # 複雑に四面体分割されたObjectを単純化する
    # 一部分（ただし、凸多面体に限る）だけ単純化したい場合．
    # この一部分はnumで指定
    # その頂点座標をcoordinatesで与える
    # Parameters:
    #  obj: 単純化したいobject(dim4のnumpy.ndarray)
    #  num: 凸多面体を構成する四面体の配列インデックスをnumで指定(dim2のlist)
    #  coordinates: 単純化したい部分の頂点（四面体分割に使う. dim3のnumpy.ndarrayを含むリスト）
    n1,n2,n3=utils.obj_volume_6d(obj)
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
        tmp4=intsct.tetrahedralization_points(coordinates[i1])
        if len(obj_new)==1:
            obj_new=tmp4.reshape(len(tmp4)*72)
        else:
            obj_new=np.append(obj_new,tmp4)
    obj_new=obj_new.reshape(len(obj_new)/72,4,6,3)
    m1,m2,m3=utils.obj_volume_6d(obj_new)
    if verbose_level>0:
         print(' volume of original obj  : %d %d %d (%10.8f)'%(n1,n2,n3,(n1+TAU*n2)/float(n3)))
         print(' volume of simplified obj: %d %d %d (%10.8f)'%(m1,m2,m3,(m1+TAU*m2)/float(m3)))
    if n1==m1 and n2==m2 and n3==m3:
        print(' succeed')
        return obj_new
    else:
        print(' fail, original obj returned.')
        return obj

def simpl_manual_2(obj, num, obj_partial, verbose_level = 0):
    """
    Legacy: simple
    """
    # 複雑に四面体分割されたObjectを単純化する
    # 四面分割を単純化したい部分が凸多面体でない場合、手で新たに四面体分割したobjを与える．
    # 一部分はnumで指定
    # 新しく四面体分割したobjを与える．
    # Parameters:
    #  obj: 単純化したいobject(dim4のnumpy.ndarray)
    #  num: 凸多面体を構成する四面体の配列インデックスをnumで指定(dim2のlist)
    #  obj_partial: 手で四面体分割したobj（四面体分割に使う. dim4のnumpy.ndarray）
    n1,n2,n3=utils.obj_volume_6d(obj)
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
    m1,m2,m3=utils.obj_volume_6d(obj_new)
    if verbose_level>0:
        print(' volume of original obj  : %d %d %d (%10.8f)'%(n1,n2,n3,(n1+TAU*n2)/float(n3)))
        print(' volume of simplified obj: %d %d %d (%10.8f)'%(m1,m2,m3,(m1+TAU*m2)/float(m3)))
    if n1==m1 and n2==m2 and n3==m3:
        print(' succeed')
        return obj_new
    else:
        print(' fail, original obj returned.')
        return obj

def simpl_manual(obj, num, verbose_level = 0):
    """
    Legacy: simple
    """
    # 複雑に四面体分割されたObjectを単純化する
    # 四面分割を単純化したい部分（凸多面体である必要がある）の四面体インデックス(m)とその頂点インデックス(n)を指定し、
    # その頂点集合を四面体分割する．
    # 新しく四面体分割したobjを与える．
    # Parameters:
    #  obj: 単純化したいobject(dim4のnumpy.ndarray)
    #  num: 凸多面体を構成する四面体インデックス(m)とその頂点インデックス(n) (dim2のlist)
    #        例　num = [[m1,n1],[m2,n2],...]
    return 0

if __name__ == "__main__":
    
    # import asymmetric part of STRT OD(occupation domain) located at origin,0,0,0,0,0,0.
    strt_asym = read_xyz(path='../xyz',basename='strt_aysmmetric')
    write(obj=strt_asym, path='.', basename = 'obj_seed', format='vesta', color = 'k')
    
    # generat STRT OD located at 0,0,0,0,0,0 by symmetric operations (m-3-5).
    pos0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
    strt = symmetric(asymmetric_part = strt_asym, position = pos0)
    
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
    
    