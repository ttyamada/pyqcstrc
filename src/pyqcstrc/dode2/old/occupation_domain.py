#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import timeit
import os
import sys
import numpy as np
#sys.path.append('.')

EPS=1e-6

try:
    import pyqcstrc.dode.math12 as math12
    #import pyqcstrc.dode.symmetry12 as symmetry12
    #import pyqcstrc.dode.intsct12 as intsct12
    #import pyqcstrc.dode.utils12 as utils12
    #import pyqcstrc.dode.strc12 as strc12
    #import pyqcstrc.dode.numericalc12 as numericalc12
except ImportError:
    print('import error\n')


SIN=np.sqrt(3)/2.0
V0=np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]])

def as_it_is(obj):
    """
    Returns an object as it is,
    
    Args:
        obj (numpy.ndarray): the shape is (num,3,6,3) or (num*3,6,3), where num=numbre_of_triangle.
    
    Returns:
        Occupation domains (numpy.ndarray): the shape is (num,3,6,3), where num=numbre_of_triangle.
    """
    
    if obj.ndim == 3:
        return obj.reshape(int(len(obj)/3),3,6,3)
    elif obj.ndim == 4:
        return obj
    else:
        return 1

def get_perp_component(vt):
    return math12.projection3(vt)
    
def write(obj, path = '.', basename = 'tmp', format = 'xyz', color = 'k', dmax = 5.0):
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
        dmax (float)
            Distance for bonds in case where 'vesta' format is selected.
            (default, dmax = 5.0)
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
            write_vesta(obj,path,basename,color,dmax)
        elif format == 'xyz':
            write_xyz(obj,path,basename)
        else:
            pass
        return 0

def write_vesta(obj, path, basename, color = 'k', dmax = 5.0, select = 'normal'):
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
        dmax (float)
            Distance for bonds in case where 'vesta' format is selected.
            (default, dmax = 5.0)
        select (str):
            (default, select = 'normal')
    Returns:
        int: 0 (succeed), 1 (fail)
    
    """
    file_name = '%s/%s.vesta'%(path,basename)
    f = open('%s'%(file_name),'w')

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
            a4,a5,a6 = math12.projection3(obj[i1][i2][0],obj[i1][i2][1],obj[i1][i2][2],obj[i1][i2][3],obj[i1][i2][4],obj[i1][i2][5])
            #print(a4,a5,a6)
            print('%4d Xx        Xx%d  1.0000    %8.6f %8.6f %8.6f        1'%\
            (i2+1,i2+1,(a4[0]+a4[1]*SIN)/a4[2],(a5[0]+a5[1]*SIN)/a5[2],(a6[0]+a6[1]*SIN)/a6[2]), file=f)
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
        if color == 'r':
            print('  1     Xx     Xx     0.00000     %3.2f  0  1  1  0  2  0.250  2.000 255 0 0'%(dmax), file=f)
        elif color == 'b':
            print('  1     Xx     Xx     0.00000     %3.2f  0  1  1  0  2  0.250  2.000 0 0 255'%(dmax), file=f)
        elif color == 'k':
            print('  1     Xx     Xx     0.00000     %3.2f  0  1  1  0  2  0.250  2.000 127 127 127'%(dmax), file=f)
        elif color == 'p':
            print('  1     Xx     Xx     0.00000     %3.2f  0  1  1  0  2  0.250  2.000 255 0 255'%(dmax), file=f)
        else:
            print('  1     Xx     Xx     0.00000     %3.2f  0  1  1  0  2  0.250  2.000 127 127 127'%(dmax), file=f)
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
    print('    written in %s'%(file_name))

    return 0

def write_xyz(obj, path, basename):
    """
    Export occupation domains in XYZ format.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangle.
        path (str): Path of the output XYZ file
        basename (str): Basename of the output XYZ file
        select (str)
            'triangle': set of triangle (default)
    Returns:
        int: 0 (succeed), 1 (fail)
    """
    
    def generator_xyz(obj,filename,num):
        """
        Generate object (set of triangle) object in XYZ format.
    
         Args:
            obj (numpy.ndarray): the occupation domain
                The shape is (num,3,6,3), where num=numbre_of_triangle.
            filename (str): filename of the output XYZ file
        
        Returns:
            int: 0 (succeed), 1 (fail)
        
        """
        f=open('%s'%(filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)*num))
        f.write('%s\n'%(filename))
        for i1 in range(len(obj)):
            for i2 in range(num):
                a4,a5,a6=math12.projection3(obj[i1][i2])
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the triangle %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                ((a4[0]+a4[1]*SIN)/(a4[2]),\
                (a5[0]+a5[1]*SIN)/(a5[2]),\
                (a6[0]+a6[1]*SIN)/(a6[2]),\
                i1,i2,\
                obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
                obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
                obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
                obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
                obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
                obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2]))
        if num>2:
            w1,w2,w3=utils12.obj_area_6d(obj)
            f.write('area = %d %d %d (%8.6f)\n'%(w1,w2,w3,(w1+SIN*w2)/(w3)))
            for i1 in range(len(obj)):
                [v1,v2,v3]=utils12.triangle_area_6d(obj[i1])
                f.write('%3d-the triangle, %d %d %d (%8.6f)\n'\
                        %(i1,v1,v2,v3,(v1+SIN*v2)/v3))
        else:
            pass
        f.closed
        return 0
    
    def generator_xyz_vertex(obj, filename):
        """
        Generate object (set of triangle) object in XYZ format.
    
         Args:
            obj (numpy.ndarray): the occupation domain
                The shape is (num,6,3), where num=numbre_of_vertex.
            filename (str): filename of the output XYZ file
        
        Returns:
            int: 0 (succeed), 1 (fail)
    
        """
        
        f=open('%s'%(filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)))
        f.write('%s\n'%(filename))
        for i1 in range(len(obj)):
            a4,a5,a6=math12.projection3(obj[i1][0],\
                                        obj[i1][1],\
                                        obj[i1][2],\
                                        obj[i1][3],\
                                        obj[i1][4],\
                                        obj[i1][5])
            f.write('Xx %8.6f %8.6f %8.6f # %3d-the vertex # # # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
            ((a4[0]+a4[1]*SIN)/(a4[2]),\
            (a5[0]+a5[1]*SIN)/(a5[2]),\
            (a6[0]+a6[1]*SIN)/(a6[2]),\
            i1,\
            obj[i1][0][0],obj[i1][0][1],obj[i1][0][2],\
            obj[i1][1][0],obj[i1][1][1],obj[i1][1][2],\
            obj[i1][2][0],obj[i1][2][1],obj[i1][2][2],\
            obj[i1][3][0],obj[i1][3][1],obj[i1][3][2],\
            obj[i1][4][0],obj[i1][4][1],obj[i1][4][2],\
            obj[i1][5][0],obj[i1][5][1],obj[i1][5][2]))
        f.closed
        return 0
    
    file_name='%s/%s.xyz'%(path,basename)
    
    if obj.ndim>3:
        generator_xyz(obj, file_name, obj.shape[1])
    elif obj.ndim==3:
        generator_xyz_vertex(obj, file_name)
    else:
        pass
    print('    written in %s/%s.xyz'%(path,basename))
    return 0

def read_xyz(path, basename, select='triangle'):
    """
    Load new occupation domain on input XYZ file.
    
    Args:
        path (str): Path of the input XYZ file
        basename (str): Basename of the input XYZ file
    
    Returns:
        Occupation domains (numpy.ndarray):
            Loaded occupation domains.
            The shape is (num,3,6,3), where num=numbre_of_triangle.
    
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
    
    if select=='triangle':
        num1=3
        return tmp.reshape(int(num/num1),num1,6,3)
    elif select=='egde':
        num1=2
        return tmp.reshape(int(num/num1),num1,6,3)
    else:
        num1=1
        return tmp.reshape(num1,int(num/num1),6,3)

def intersection_two_segments(segment1, segment2, verbose = 0):
    """
    Returns an intersecting point of two line segments.
    
    Args:
        segment1 (numpy.ndarray): the shape is (2,6,3).
        segment2 (numpy.ndarray): the shape is (2,6,3).
        verbose (int): 0 (defalt)
    Returns:
        intersecting point (numpy.ndarray): the shape is (6,3).
    """
    p=intsct12.intersection_two_segment(segment1[0],segment1[1],segment2[0],segment2[1],verbose)
    return p.reshape(6,3)

def symmetric_12(obj):
    """
    Generate symmteric occupation domain by symmetric elements of 12m on the input.
    
    Args:
        obj (numpy.ndarray):
            occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangle.
    
    Returns:
        Symmetric occupation domains (numpy.ndarray):
            The shape is (num,3,6,3), where num=numbre_of_triangle.
    
    """
    
    if asymmetric_part_obj.ndim == 3:
        return symmetry12.generator_obj_symmetric_triangle(asymmetric_part_obj, V0)
    elif asymmetric_part_obj.ndim == 4:
        asymmetric_part_obj=asymmetric_part_obj.reshape(len(asymmetric_part_obj)*3,6,3)
        return symmetry12.generator_obj_symmetric_triangle(asymmetric_part_obj, V0)

def site_symmetry(wyckoff_position, verbose=0):
    """
    Symmetry operators in the site symmetry group G and its left coset decomposition.
    
    Args:
        Wyckoff position (numpy.ndarray):
            4D coordinate.
            The shape is (6,3).
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
        symop=[]
        tmp=np.array([0,0,0,0,0,0])
        symop.append(tmp)
        for i1 in [-1,0,1]:
            for i2 in [-1,0,1]:
                for i3 in [-1,0,1]:
                    for i4 in [-1,0,1]:
                        tmp=np.array([i1,i2,i3,i4,0,0])
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
    
    symop=symmetry12.dodesymop()
    traop=translation()
    
     # List of index of symmetry operators of the site symmetry group G.
     # The symmetry operators leaves xyz identical.
    list1=[]
    
    # List of index of symmetry operators which are not in the G.
    list2=[]
    
    pos=wyckoff_position
    a1=(pos[0][0]+SIN*pos[0][1])/pos[0][2]
    a2=(pos[1][0]+SIN*pos[1][1])/pos[1][2]
    a3=(pos[2][0]+SIN*pos[2][1])/pos[2][2]
    a4=(pos[3][0]+SIN*pos[3][1])/pos[3][2]
    xyz=np.array([a1,a2,a3,a4,0,0])
    
    xyi=numericalc12.projection_1_numerical(xyz[0],xyz[1],xyz[2],xyz[3],xyz[4],xyz[5])
    xi=xyi[3]
    yi=xyi[4]
    if verbose>0:
        print(' site coordinates: %3.2f %3.2f %3.2f %3.2f'%(xyz[0],xyz[1],xyz[2],xyz[3]))
        print('         in Epar : %5.3f %5.3f'%(xyi[0],xyi[1]))
        print('         in Eperp: %5.3f %5.3f'%(xyi[3],xyi[4]))
    else:
        pass
    
    for i2 in range(len(symop)):
        flag=0
        for i1 in range(len(traop)):
            xyz1=np.dot(symop[i2],xyz)
            xyz2=xyz1+traop[i1]
            a=numericalc12.projection_1_numerical(xyz2[0],xyz2[1],xyz2[2],xyz2[3],xyz2[4],xyz2[5])
            if abs(a[3]-xi)<EPS and abs(a[4]-yi)<EPS:
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

def symmetric(asymmetric_unit, site, verbose = 0):
    """
    Generate symmterical occupation domain from its asymmetric unit by applying site simmetry.
    
    Args:
        asymmetric_unit (numpy.ndarray):
            Asymmetric unit of the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangle.
        site (numpy.ndarray):
            6d coordinate of the site of which the occupation domain centres.
            The symmetric centre.
            The shape is (6,3)
    
    Returns:
        Symmetric occupation domains (numpy.ndarray):
            The shape is (num,3,6,3), where num=numbre_of_triangle.
    
    """
    
    # get site symmetry
    #a1=(site[0][0]+SIN*site[0][1])/site[0][2]
    #a2=(site[1][0]+SIN*site[1][1])/site[1][2]
    #a3=(site[2][0]+SIN*site[2][1])/site[2][2]
    #a4=(site[3][0]+SIN*site[3][1])/site[3][2]
    #xyz=np.array([a1,a2,a3,a4,0,0])
    num_symop,num_coset=site_symmetry(site,verbose)
    
    if asymmetric_unit.ndim == 3:
        asym_obj=asymmetric_unit.reshape(int(len(asymmetric_unit)/3),3,6,3)
    else:# asymmetric_unit.ndim == 4:
        asym_obj=asymmetric_unit
   
    for i in range(len(num_symop)):
        obj1=symmetry12.generator_symmetric_obj_0(obj=asym_obj,centre=site,numop=num_symop[i])
        if i==0:
            objs=obj1
        else:
            objs=np.append(objs,obj1)
    objs=objs.reshape(int(len(objs)/54),3,6,3)
    
    return objs

# dev, currently this does not work...
def asymmetric(symmetric_obj, position, vecs):
    """
    Asymmetric part of occupation domain.
    
    Args:
        symmetric_obj (numpy.ndarray):
            Occupation domain of which the asymmetric part is calculated.
            The shape is (num,2,6,3), where num=numbre_of_triangle.
        position (numpy.ndarray):
            6d coordinate of the site of which the occupation domain centres.
            The shape is (6,3)
        vecs (numpy.ndarray):
            Three vectors that defines the asymmetric part.
            The shape is (2,6,3)
    
    Returns:
        Asymmetric part of the occupation domains (numpy.ndarray):
            The shape is (num,3,6,3), where num=numbre_of_triangle.
    """
    v0 = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    vecs = math12.mul_vectors(vecs,[10,0,1])
    # vecs multiplied by [a,b,c], where [a,b,c]=(a+TAU*b)/c. 
    # [a,b,c] has to be defined so that the tetrahedron whose vertices are defined 
    # by v0, and vecs covers the asymmetric unit of the ocuppation domains.
    # (default) [a,b,c]=[5,0,1].
    tmp = np.append(v0,vecs).reshape(3,6,3)
    aum = as_it_is(tmp)
    aum = shift(aum,position)
    print(aum.ndim)
    print(aum)
    od_asym = intersection(symmetric_obj,aum,0)
    od_asym_1=outline1(od_asym)
    return od_asym_1

def outline(obj,verbose = 0):
    """
    Generate outline of the occupation domain.
    
    Args:
        obj (numpy.ndarray): the shape is (num,3,6,3), where num=numbre_of_triangle.
        verbose (int): 0 (defalt)
    
    Returns:
        Outline of the occupation domain (numpy.ndarray):
            The shape is (num,2,6,3), where num=number of the outlines.
    
    """
    return utils12.generator_obj_outline(obj,verbose)

def outline1(obj, num_cycle = 20, verbose = 0):
    """
    Generate outline of the occupation domain.
    
    Args:
        obj (numpy.ndarray): the shape is (num,3,6,3), where num=numbre_of_triangle.
        num_cycle (int): 10 (defalt)
        verbose (int): 0 (defalt)
    
    Returns:
        Outline of the occupation domain (numpy.ndarray):
            The shape is (num,2,6,3), where num=number of the outlines.
    
    """
    return utils12.surface_cleaner(obj,num_cycle,verbose)

def triangulation(obj, position, num_cycle = 20, verbose = 0):
    """
    Triangulation of the occupation domain.
    
    Args:
        obj (numpy.ndarray): the shape is (num,3,6,3), where num=numbre_of_triangle.
        position (numpy.ndarray): 6d vector. the shape is (6,3)
        num_cycle (int): 10 (defalt)
        verbose (int): 0 (defalt)
    
    Returns:
        Outline of the occupation domain (numpy.ndarray):
            The shape is (num,3,6,3), where num=number of triangles.
    
    """
    a=utils12.surface_cleaner(obj,num_cycle,verbose)
    num=len(a)
    for i in range(0,num):
        b=np.append(position,a[i])
        if i==0:
            c=b
        else:
            c=np.append(c,b)
    return c.reshape(num,3,6,3)

def shift(obj, shift, verbose = 0):
    """
    Shift the occupation domain.
    
    Args:
        obj (numpy.ndarray):
            The occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangle.
        shift (numpy.ndarray):
            6d coordinate to which the occupation domain is shifted.
            The shape is (6,3)
        verbose (int):
            verbose = 0 (silent, default)
            verbose = 1 (normal)
    
    Returns:
        Shifted occupation domains (numpy.ndarray):
            The shape is (num,3,6,3), where num=numbre_of_triangle.
    
    """
    return utils12.shift_object(obj, shift, verbose)

def intersection(obj1,obj2,verbose = 0):
    
    tmp=intsct12.intersection_two_obj(obj1,obj2,verbose)
    
    if tmp.tolist()!=[[[[0]]]]:
        pass
    else:
        if verbose>0:
            print('empty')
        else:
            pass
    
    return tmp

def similarity(obj,order):
    if obj.ndim == 3:
        return symmetry12.similarity_obj(obj,order)
    elif obj.ndim == 4:
        obj=obj.reshape(len(obj)*3,6,3)
        return symmetry12.similarity_obj(obj,order)
    else:
        return 1
    
def qcstrc(obj, positions, path, basename, atm, phason_matrix, nmax, shift, origin_shift, verbose = 1, option = 0):
    
    #if phason_matrix == 0 or phason_matrix==[[[0]]] or phason_matrix==[[0]] or phason_matrix==[0]:
    if np.all(phason_matrix)==0:
        phason_matrix = np.array([[0.0]])
    else:
        pass
    
    if len(obj)==len(atm) and len(obj)==len(positions):
        #ishift=[0.00178250, 0.00137613, 0.003987675, 0.2387783, 0, 0]
        
        lst=[]
        a=strc12.strc(obj,positions,phason_matrix,nmax,shift,origin_shift,verbose)
        f=open('%s/%s.xyz'%(path,basename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(a)))
        f.write('%s.xyz\n'%(basename))
        for i1 in range(len(a)):
            #print(a[i1][3])
            if option == 0:
                f.write('%s %8.6f %8.6f %8.6f\n'%(atm[int(a[i1][3])],a[i1][0],a[i1][1],a[i1][2]))
            elif option == 1: # Eperp, x, y
                #f.write('%s %8.6f %8.6f %8.6f # %8.6f %8.6f %8.6f\n'%(atm[int(a[i1][3])],a[i1][0],a[i1][1],a[i1][2],a[i1][4],a[i1][5],a[i1][6]))
                f.write('%s %8.6f %8.6f %8.6f # %3d %3d %3d %3d\n'%(atm[int(a[i1][3])],a[i1][0],a[i1][1],a[i1][2],a[i1][4],a[i1][5],a[i1][6],a[i1][7]))
            else:
                pass
        f.closed
        print('    written in %s/%s'%(path,basename))
    else:
        pass
    return 0

# dev.
def genstrc(l1):
    """
    Generate atomic structure of DD QC.
    
    Args:
        l1 (list):
            List contains structural dataThe occupation domain, which contains following:
            
            (1) The shape is (num,3,6,3), where num=numbre_of_triangle.
            
    Returns:
        0 (int):
    
    """
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
            for i3 in range(3):
                if i2==0 and i3==0:
                    a=obj[vlist[i1][i2][0]-1]
                else:
                    a=np.append(a,obj[vlist[i1][i2][i3]-1])
        a=a.reshape(int(len(a)/54),3,6,3)
        if len(vlist)==1:
            basename1='%s'%(basename)
        else:
            basename1='%s_part%d'%(basename,i1)
        write_xyz(obj=a, path=path, basename=basename1)
        write_vesta(obj=a, path=path, basename=basename1, color='p')
    return 0

if __name__ == "__main__":
    
    # Three 6D vectors which define the asymmetric part of the occupation domain for Nizeki-Gahler dodecagonal tiling
    # Note that 5-th and 6-th components of each 6D vectors are dummy, and they correspond to Z coordinate in Epar and Eperp, respectively.
    v0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,0,0,0)
    v1=np.array([[ 0,-2, 3],[ 0, 2, 3],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) #
    v2=np.array([[ 0,-2, 3],[ 0, 2, 3],[ 3,-4, 3],[-3, 4, 3],[ 0, 0, 1],[ 0, 0, 1]]) #
    asym_obj=np.vstack([v0,v1,v2]).reshape(3,6,3)
    
    # OBJ_1
    obj1=symmetric(asym_obj,v0)
    # Shift vector for OBJ_2
    pos=np.array([[ 0, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # ( 0,1,0,0)

    # OBJ_1
    #obj1=asym_obj.reshape(1,3,6,3)
    # Shift vector for OBJ_2
    #pos=np.array([[ 0, 0, 1],[ 1, 0, 4],[ 1, 0, 4],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # ( 0,1/4,1/4,0)
    
    print('symmetric obj:')
    for i in range(len(obj1)):
        vi=obj1[i]
        for j in range(len(vi)):
            vj=vi[j]
            v1i,v2i,v3i=math12.projection3(vj[0],vj[1],vj[2],vj[3],vj[4],vj[5])
            print('%8.6f %8.6f %8.6f'%(((v1i[0]+v1i[1]*SIN)/v1i[2],(v2i[0]+v2i[1]*SIN)/v2i[2],(v3i[0]+v3i[1]*SIN)/v3i[2])))
    
    print('\noutline of obj:')
    a=outline(obj1)
    print('number of line segments = %d'%(len(a)))
    for i in range(len(a)):
        ai=a[i].reshape(2,6,3)
        for j in range(2):
            v1i,v2i,v3i=math12.projection3(ai[j][0],ai[j][1],ai[j][2],ai[j][3],ai[j][4],ai[j][5])
            print('%8.6f %8.6f %8.6f'%(((v1i[0]+v1i[1]*SIN)/v1i[2],(v2i[0]+v2i[1]*SIN)/v2i[2],(v3i[0]+v3i[1]*SIN)/v3i[2])))
    
    print('Shift obj:')
    obj2=shift(obj1,pos)
    #
    print('\noutline of shifted obj:')
    a=outline(obj2)
    print('number of line segments = %d'%(len(a)))
    for i in range(len(a)):
        ai=a[i].reshape(2,6,3)
        for j in range(2):
            v1i,v2i,v3i=math12.projection3(ai[j][0],ai[j][1],ai[j][2],ai[j][3],ai[j][4],ai[j][5])
            print('%8.6f %8.6f %8.6f'%(((v1i[0]+v1i[1]*SIN)/v1i[2],(v2i[0]+v2i[1]*SIN)/v2i[2],(v3i[0]+v3i[1]*SIN)/v3i[2])))
    
    print('\nIntersection of two objs:')
    obj3=intersection(obj1,obj2,verbose=0)
    if obj3.tolist()!=[[[[0]]]]:
        print('number of common triangles = %d'%(len(obj3)))
        for i in range(len(obj3)):
            for j in range(3):
                v1i,v2i,v3i=math12.projection3(obj3[i][j][0],obj3[i][j][1],obj3[i][j][2],obj3[i][j][3],obj3[i][j][4],obj3[i][j][5])
                print('%8.6f %8.6f %8.6f'%(((v1i[0]+v1i[1]*SIN)/v1i[2],(v2i[0]+v2i[1]*SIN)/v2i[2],(v3i[0]+v3i[1]*SIN)/v3i[2])))
    else:
        print(' no common part found')
    
    """
    vj=np.array([[-1,0,2],[1,0,2],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    v1i,v2i,v3i=math12.projection3(vj[0],vj[1],vj[2],vj[3],vj[4],vj[5])
    print('%8.6f %8.6f %8.6f'%(((v1i[0]+v1i[1]*SIN)/v1i[2],(v2i[0]+v2i[1]*SIN)/v2i[2],(v3i[0]+v3i[1]*SIN)/v3i[2])))
    
    vj=np.array([[-1,0,1],[1,0,1],[-1,1,1],[1,-1,1],[0,0,1],[0,0,1]])
    v1i,v2i,v3i=math12.projection3(vj[0],vj[1],vj[2],vj[3],vj[4],vj[5])
    print('%8.6f %8.6f %8.6f'%(((v1i[0]+v1i[1]*SIN)/v1i[2],(v2i[0]+v2i[1]*SIN)/v2i[2],(v3i[0]+v3i[1]*SIN)/v3i[2])))
    """