#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import os
import sys
import numpy as np
sys.path.append('..')
import occupation_domain as od
import utils2

SIN=np.sqrt(3)/2.0

a=od.read_xyz(path='./test1',basename='obj3')
b=utils2.triangle_area_6d(a[0])
print(b)
print((b[0]+SIN*b[1])/b[2])

print((10619-12260*SIN)/24)
