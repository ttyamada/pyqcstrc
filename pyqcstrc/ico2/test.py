#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
import sys
sys.path.append('.')
import numpy as np
import time

try:
    import occupation_domain as od
    import two_occupation_domains as tod
    
    import utils
    
except ImportError:
    print('import error\n')


if __name__ == "__main__":
    
    # import asymmetric part of STRT OD(occupation domain) located at origin,0,0,0,0,0,0.
    strt_asym=od.read_xyz(path='../xyz',basename='strt_aysmmetric')
    od.write(obj=strt_asym, path='.', basename='strt_aysmmetric', format='xyz')
    od.write(obj=strt_asym, path='.', basename='strt_aysmmetric', format='vesta')
    
    surface_triangles=utils.generator_surface_1(strt_asym)
    od.write(obj=surface_triangles, path='.', basename='strt_asym_surface_triangles', format='vesta',select='normal')
    od.write(obj=surface_triangles, path='.', basename='strt_asym_surface_triangles', format='xyz', select='triangle')
    surface_edges=utils.generator_unique_edges(surface_triangles)
    od.write(obj=strt_asym, path='.', basename='strt_asym_surface_edges', format='vesta',select='normal')
    
    
    # generat STRT OD located at 0,0,0,0,0,0 by symmetric operations (m-3-5).
    pos0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    strt_sym=od.symmetric(strt_asym,pos0)
    od.write(obj=strt_sym, path='.', basename='obj_strt0', format='xyz')
    od.write(obj=strt_sym, path='.', basename='obj_strt0', format='vesta')
    
    # move STRT OD to a position 1 1 1 0 -1 0.
    pos_b1=np.array([[1,0,1],[1,0,1],[1,0,1],[0,0,1],[-1,0,1],[0,0,1]])#b_1
    strt_pos1=od.shift(strt_sym,pos_b1)
    od.write(obj=strt_pos1, path='.', basename='obj_strt1', format='xyz')
    od.write(obj=strt_pos1, path='.', basename='obj_strt1', format='vesta')
    
    # intersection of "asymmetric part of strt" and "strt at position pos_b1"
    print('    intersection starts')
    start = time.time()
    ###
    common=tod.intersection(strt_asym,strt_pos1)
    ###
    end=time.time()
    time_diff=end-start
    print('                 ends in %4.3f sec'%time_diff)  # 処理にかかった時間データを使用
    
    print('    generator_surface_1 starts')
    start = time.time()
    ###
    # Export common in vesta
    od.write(obj=common, path='.', basename='common', format='xyz')
    od.write(obj=common, path='.', basename='common', format='vesta')
    #common=od.read_xyz(path='.',basename='common')
    surface_triangles=utils.generator_surface_1(common)
    od.write(obj=surface_triangles, path='.', basename='common_surface_triangles', format='vesta',select='normal')
    surface_edges=utils.generator_unique_edges(surface_triangles)
    od.write(obj=surface_edges, path='.', basename='common_surface_edges', format='vesta',select='normal')
    ###
    end=time.time()
    time_diff=end-start
    print('                 ends in %4.3f sec'%time_diff)  # 処理にかかった時間データを使用
    
    
    #"""
    # TEST intersection_convex()
    print('    intersection_convex starts')
    start = time.time()
    ###
    common1=tod.intersection_convex(strt_sym,strt_pos1)
    od.write(obj=common1, path='.', basename='common_convex', format='xyz')
    od.write(obj=common1, path='.', basename='common_convex', format='vesta')
    ###
    end=time.time()
    time_diff=end-start
    print('                 ends in %4.3f sec'%time_diff)  # 処理にかかった時間データを使用
    #"""