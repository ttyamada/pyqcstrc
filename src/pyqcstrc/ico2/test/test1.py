#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
import sys
sys.path.append('../')
import numpy as np
import time

try:
    import occupation_domain as od
    import two_occupation_domains as tod
    import utils
    import numericalc
except ImportError:
    print('import error\n')


if __name__ == "__main__":
    
    opath='.'
    xyzpath='../../../../pyqcstrc/xyz'
    #--------------
    # Object A
    #--------------
    strt_aysmmetric=od.read_xyz(path=xyzpath,basename='strt_aysmmetric',select='tetrahedron',verbose=0)
    #od.write(obj=strt_aysmmetric, path=opath, basename='strt_aysmmetric', format='vesta')
    #obj_surface=utils.generator_surface_1(strt_aysmmetric)
    
    #strt_aysmmetric_convex_hull=utils.generate_convex_hull(strt_aysmmetric)
    #od.write(obj=strt_aysmmetric_convex_hull, path=opath, basename='strt_aysmmetric_convex_hull', format='vesta')
    
    #--------------
    # Object B
    #--------------
    # generat STRT OD located at 0,0,0,0,0,0 by symmetric operations (m-3-5).
    pos0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    strt_sym=od.symmetric(strt_aysmmetric,pos0)
    #
    # move STRT OD to a position 1 0 0 0 0 0.
    POS_B1=np.array([[ 1, 0, 1],[ 1, 0, 1],[ 1, 0, 1],[ 0, 0, 1],[-1, 0, 1],[ 0, 0, 1]]) # b_1
    strt_sym_pos1=od.shift(strt_sym,POS_B1)
    #od.write(obj=strt_sym_pos1, path=opath, basename='obj_strt1', format='xyz')
    #od.write(obj=strt_sym_pos1, path=opath, basename='obj_strt1', format='vesta')
    
    #--------------
    # A AND B = C
    #--------------
    print('    intersection starts')
    start=time.time()
    ###
    common=tod.intersection(strt_aysmmetric,strt_sym_pos1)
    ###
    end=time.time()
    time_diff=end-start
    print('                 ends in %4.3f sec'%time_diff)
    
    #--------------
    # Simplification
    #--------------
    print('    simplification starts')
    start=time.time()
    ###
    common_simple=od.simplification(common)
    od.write(obj=common_simple, path=opath, basename='common_simplified', format='vesta')
    ###
    end=time.time()
    time_diff=end-start
    print('                 ends in %4.3f sec'%time_diff)
    
    #--------------
    # A NOT B
    #--------------
    print('    subtraction starts')
    start=time.time()
    ###
    a=tod.subtraction(strt_aysmmetric,strt_sym_pos1,verbose=0)
    volume=utils.obj_volume_6d(a)
    print('     volume=',volume)
    #a=tod.object_subtraction(strt_aysmmetric[0].reshape(1,4,6,3),strt_sym_pos1,flag=1)
    #od.write(obj=a, path=opath, basename='strt_subtracted_aysmmetric_1', format='xyz')
    od.write(obj=a, path=opath, basename='strt_subtracted_aysmmetric_1', format='vesta')
    ###
    end=time.time()
    time_diff=end-start
    print('                 ends in %4.3f sec'%time_diff)  # 処理にかかった時間データ
    
    #---------------------------
    # A NOT B = A NOT C
    #---------------------------
    print('    object_subtraction starts')
    start=time.time()
    ###
    a=tod.subtraction(strt_aysmmetric,common,verbose=2)
    volume=utils.obj_volume_6d(a)
    print('     volume=',volume)
    #a=tod.object_subtraction(strt_aysmmetric[0].reshape(1,4,6,3),common_simple,flag=1)
    #od.write(obj=a, path=opath, basename='strt_subtracted_aysmmetric_2', format='xyz')
    od.write(obj=a, path=opath, basename='strt_subtracted_aysmmetric_2', format='vesta')
    ###
    end=time.time()
    time_diff=end-start
    print('                 ends in %4.3f sec'%time_diff)  # 処理にかかった時間データ
    
    
    
    #od.write(obj=common, path=opath, basename='common', format='vesta')
    #od.write(obj=common, path=opath, basename='common', format='xyz')
    
    
    #common=od.read_xyz(path='.',basename='common',select='tetrahedron',verbose=0)
    
    #triangle_surface=utils.generator_surface_1(common)
    #od.write(obj=triangle_surface, path=opath, basename='triangle_surface', format='vesta')
    
    #edge_surface=utils.surface_cleaner(triangle_surface)
    #od.write(obj=edge_surface, path=opath, basename='edge_surface', format='vesta')
    
    #common_convex_hull=utils.generate_convex_hull(common)
    #od.write(obj=common_convex_hull, path=opath, basename='common_convex_hull', format='vesta')
    
