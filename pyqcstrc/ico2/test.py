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
    
    opath='./output'
    # generate asymmetric part of RT OD(occupation domain) located at origin,0,0,0,0,0,0.
    v0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
    v1 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
    v2 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
    v3 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 0, 0, 2],[-1, 0, 2],[ 0, 0, 2]])
    rt_asym = np.vstack([v0,v1,v2,v3]).reshape(1,4,6,3)
    od.write(obj=rt_asym, path=opath, basename='rt_asymmeric', format='xyz')
    od.write(obj=rt_asym, path=opath, basename='rt_asymmeric', format='vesta')
    
    #a=utils.merge_two_tetrahedra(strt_asym[0],strt_asym[1])
    #print(a)
    
    
    surface_triangles=utils.generator_surface_1(rt_asym)
    od.write(obj=surface_triangles, path=opath, basename='rt_asym_surface_triangles', format='vesta',select='normal')
    #od.write(obj=surface_triangles, path='.', basename='rt_asym_surface_triangles', format='xyz', select='triangle')
    surface_edges=utils.generator_unique_edges(surface_triangles)
    od.write(obj=rt_asym, path=opath, basename='rt_asym_surface_edges', format='vesta',select='normal')
     
    # generat STRT OD located at 0,0,0,0,0,0 by symmetric operations (m-3-5).
    pos0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    rt_sym=od.symmetric(rt_asym,pos0)
    #rt_sym=od.symmetric_0(rt_asym,pos0,7)
    #rt_sym=od.symmetric_0(rt_asym,pos0,113)
    od.write(obj=rt_sym, path=opath, basename='obj_rt0', format='xyz')
    od.write(obj=rt_sym, path=opath, basename='obj_rt0', format='vesta')
    
    # move STRT OD to a position 1 0 0 0 0 0.
    pos_1=np.array([[1,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    rt_sym_pos1=od.shift(rt_sym,pos_1)
    od.write(obj=rt_sym_pos1, path=opath, basename='obj_rt1', format='xyz')
    od.write(obj=rt_sym_pos1, path=opath, basename='obj_rt1', format='vesta')
    #
    #surface_triangles=utils.generator_surface_1(rt_sym_pos1)
    #od.write(obj=surface_triangles, path=opath, basename='obj_rt1_surface_triangles', format='vesta',select='normal')
    #surface_edges=utils.generator_unique_edges(surface_triangles)
    #od.write(obj=surface_edges, path=opath, basename='obj_rt1_surface_edges', format='vesta',select='normal')
    
    
    
    # intersection of "asymmetric part of rt" and "rt at position pos_b1"
    print('    intersection starts')
    start = time.time()
    ###
    #common=tod.intersection(rt_asym,rt_sym_pos1)
    common=tod.intersection(rt_sym,rt_sym_pos1)
    ###
    end=time.time()
    time_diff=end-start
    print('                 ends in %4.3f sec'%time_diff)  # 処理にかかった時間データを使用
    
    # Export common in vesta
    od.write(obj=common, path=opath, basename='common', format='xyz')
    od.write(obj=common, path=opath, basename='common', format='vesta')
    #common=od.read_xyz(path=opath,basename='common')
    
    print('    generator_surface_1 starts')
    start = time.time()
    ###
    surface_triangles=utils.generator_surface_1(common)
    od.write(obj=surface_triangles, path=opath, basename='common_surface_triangles', format='vesta',select='normal')
    surface_edges=utils.generator_unique_edges(surface_triangles)
    od.write(obj=surface_edges, path=opath, basename='common_surface_edges', format='vesta',select='normal')
    ###
    end=time.time()
    time_diff=end-start
    print('                 ends in %4.3f sec'%time_diff)  # 処理にかかった時間データを使用
    
    
    """
    # TEST intersection_convex()
    print('    intersection_convex starts')
    start = time.time()
    ###
    common1=tod.intersection_convex(rt_sym,rt_pos1)
    od.write(obj=common1, path='.', basename='common_convex', format='xyz')
    od.write(obj=common1, path='.', basename='common_convex', format='vesta')
    ###
    end=time.time()
    time_diff=end-start
    print('                 ends in %4.3f sec'%time_diff)  # 処理にかかった時間データを使用
    """