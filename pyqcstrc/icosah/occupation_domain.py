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

TAU=(1+np.sqrt(5))/2.0

def as_it_is(obj):
    if obj.ndim == 3:
        return obj.reshape(int(len(obj)/4),4,6,3)
    elif obj.ndim == 4:
        return obj
    else:
        return 1
    
def symmetric(asymmetric_part_obj, position):
    """
    This makes symmterical obj by symmetric operations of point fgroup of m-3-5 on asymmetric part of obj.
    """
    if asymmetric_part_obj.ndim == 3:
        return symmetry.generator_obj_symmetric_tetrahedron(asymmetric_part_obj, position)
    elif asymmetric_part_obj.ndim == 4:
        return symmetry.generator_obj_symmetric_tetrahedron(asymmetric_part_obj.reshape(len(asymmetric_part_obj)*4,6,3), position)
    else:
        return 1
    
def shift(obj, shift, vorbose = 0):
    return utils.shift_object(obj, shift, vorbose)

def generate_edges(obj):
    obj_surface = mics.generator_surface_1(obj)
    #obj_surface = mics.generator_surface(obj)
    return mics.generator_edge(obj_surface)

def gen_surface(obj):
    return mics.generator_surface(obj)

def write(obj, path = '.', basename = 'tmp', format = 'xyz', color = 'k', dmax = 5.0):
    
    if os.path.exists(path) == False:
        os.makedirs(path)
    else:
        pass
        
    if obj.tolist()==[[[[0]]]]:
        print('    Empty OD')
        return 1
    else:
        if format == 'vesta' or format == 'v' or format == 'VESTA':
            write_vesta(obj, path, basename, color, dmax)
        elif format == 'xyz':
            write_xyz(obj, path, basename)
        else:
            pass
        return 0
    
def write_vesta(obj, path, basename, color = 'k', dmax = 5.0):
    
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
            a1,a2,a3,a4,a5,a6 = math1.projection(obj[i1][i2][0],obj[i1][i2][1],obj[i1][i2][2],obj[i1][i2][3],obj[i1][i2][4],obj[i1][i2][5])
            print('%4d Xx        Xx%d  1.0000    %8.6f %8.6f %8.6f        1'%\
            (i2+1,i2+1,(a4[0]+a4[1]*TAU)/float(a4[2]),(a5[0]+a5[1]*TAU)/float(a5[2]),(a6[0]+a6[1]*TAU)/float(a6[2])), file=f)
            #print >>f, '%4d Xx        Xx%d  1.0000    %8.6f %8.6f %8.6f        1 # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'%\
            #(i2,i2,\
            #(a4[0]+a4[1]*TAU)/float(a4[2]),(a5[0]+a5[1]*TAU)/float(a5[2]),(a6[0]+a6[1]*TAU)/float(a6[2]),\
            #obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
            #obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
            #obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
            #obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
            #obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
            #obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2])
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
    print('    written in %s'%(file_name))
    
    return 0

def write_xyz(obj, path, basename):

    def generator_xyz_dim4_tetrahedron(obj, filename):
        f=open('%s'%(filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)*4))
        f.write('%s\n'%(filename))
        for i1 in range(len(obj)):
            for i2 in range(4):
                a1,a2,a3,a4,a5,a6=math1.projection(obj[i1][i2][0],\
                                                    obj[i1][i2][1],\
                                                    obj[i1][i2][2],\
                                                    obj[i1][i2][3],\
                                                    obj[i1][i2][4],\
                                                    obj[i1][i2][5])
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                ((a4[0]+a4[1]*TAU)/float(a4[2]),\
                (a5[0]+a5[1]*TAU)/float(a5[2]),\
                (a6[0]+a6[1]*TAU)/float(a6[2]),\
                i1,i2,\
                obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
                obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
                obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
                obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
                obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
                obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2]))
        w1,w2,w3=utils.obj_volume_6d(obj)
        f.write('volume = %d %d %d (%8.6f)\n'%(w1,w2,w3,(w1+TAU*w2)/float(w3)))
        for i1 in range(len(obj)):
            [v1,v2,v3]=utils.tetrahedron_volume_6d(obj[i1])
            f.write('%3d-the tetrahedron, %d %d %d (%8.6f)\n'\
                    %(i1,v1,v2,v3,(v1+TAU*v2)/float(v3)))
        f.closed
        return 0
    
    file_name='%s/%s.xyz'%(path,basename)
    generator_xyz_dim4_tetrahedron(obj, file_name)
    print('    written in %s/%s.xyz'%(path,basename))
    
    return 0

def read_xyz(path,basename):
    """
     import xyz file (set of tetrahedra, dim=4)
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
    
    return tmp.reshape(int(num/4),4,6,3)

def simple(obj, select, num_cycle = 3, verbose = 0, num_cycle_234 = [3,0,0], num_shuffle_234 = [1,0,0]):
    if select == 0:
        obj_new = mics.simplification_obj_edges(obj, num_cycle, verbose)
    elif select == 1:
        obj_new = mics.simplification_obj_edges_1(obj, num_cycle, verbose)
    elif select == 2:
        obj_new=mics.simplification_obj_smart(obj, num_cycle, verbose)
    elif select == 3:
        obj_new = mics.simplification_convex_polyhedron(obj, num_cycle, verbose)
    elif select == 4:
        [num2_of_cycle,num3_of_cycle,num4_of_cycle] = num_cycle_234
        [num2_of_shuffle,num3_of_shuffle,num4_of_shuffle] = num_shuffle_234
        obj_new = mics.simplification(obj,num2_of_cycle,num3_of_cycle,num4_of_cycle,num2_of_shuffle,num3_of_shuffle,num4_of_shuffle,verbose)
    else:
        obj_new=mics.simplification_obj_smart(obj, num_cycle, verbose)
    return obj_new

def simple_special(obj, num, num_cycle, verbose_level = 0):
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
        tmp=mics.simplification_convex_polyhedron(tmp.reshape(int(len(tmp)/72),4,6,3),num_cycle,verbose_level)
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
    # Parameters:
    #  obj: 単純化したいobject(dim4のnumpy.ndarray) ただし、凸多面体のみ
    #  point: obj内に追加したい点の6次元座標(dim2のnumpy.ndarray)
    obj_new=intsct.tetrahedralization_1(obj, point, verbose)
    return obj_new
    
def simpl_manual(obj, num, coordinates, verbose_level = 0):
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

def simpl_manual_1(obj, num, verbose_level = 0):
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
    
    