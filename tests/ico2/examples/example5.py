#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import timeit
import os
import sys
import numpy as np
import ico2.occupation_domain as od
import ico2.two_occupation_domains as ods

V0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
BC = np.array([[ 1, 0, 2],[ 1, 0, 2],[ 1, 0, 2],[ 1, 0, 2],[ 1, 0, 2],[ 1, 0, 2]])
EC = np.array([[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])

common_od_asym=od.read_xyz(path='./example3', basename='common_od_asym_smpl')
# back to the origin
v4 = np.array([[-1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
common_od_asym=od.shift(common_od_asym,v4)

od.obj2podatm(obj=common_od_asym,serial_number=1,path='./example5',basename='tmp',shift=[0,0,0,0,0,0])
