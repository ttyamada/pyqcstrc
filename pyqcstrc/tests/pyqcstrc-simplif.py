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
import pyqcstrc.icosah.occupation_domain as od
import pyqcstrc.icosah.two_occupation_domains as ods

path_work='./test1'


# import asymmetric part of STRT OD(occupation domain) located at origin,0,0,0,0,0,0.
obj_common = od.read_xyz(path=path_work,basename='common')

# simplification
common_new=od.simple(obj=obj_common, select=0, num_cycle=3, verbose=1)

# export common part in xyz file
od.write(obj=common_new, path=path_work, basename='common_new', format='xyz')
od.write(obj=common_new, path=path_work, basename='common_new', format='vesta', color='r')
