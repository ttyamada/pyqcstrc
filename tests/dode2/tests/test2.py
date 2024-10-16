#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

#import timeit
#import os
import sys
import numpy as np
import pyqcstrc.dode2.symmetry as sym
import pyqcstrc.dode2.numericalc as numcalc

verbose=0
#verbose=1

ndim=4
#ndim=5

site_1a=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
site_6a=np.array([[ 0, 0, 1],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
site_4b=np.array([[ 0, 0, 1],[ 2, 0, 3],[ 0, 0, 1],[ 1, 0, 3],[ 0, 0, 1],[ 0, 0, 1]])
site_3a=np.array([[ 0, 0, 1],[ 1, 0, 2],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
site_6b=np.array([[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1]])
#site=site_1a
site=site_6a
#site=site_4b
#site=site_3a
#site=site_6b

vn=numcalc.numerical_vector(site)
print(' site coordinates: %3.2f %3.2f %3.2f %3.2f %3.2f %3.2f'%(vn[0],vn[1],vn[2],vn[3],vn[4],vn[5]))

# list1: symmetry operators of site symmetry group, G
# list2: symmetry operators of the left coset representative of G.
list1=sym.site_symmetry(site,ndim)
list2=sym.coset(site,ndim)
print('     multiplicity:',len(list1))
print('    site symmetry:',list1)
print('       left coset:',list2)
#list3,list4=sym.site_symmetry_and_coset(site,ndim,verbose)
#print('     multiplicity:',len(list3))
#print('    site symmetry:',list3)
#print('       left coset:',list4)
