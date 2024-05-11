#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# PyQC - Python tools for Quasi-Crystallography
# Copyright (c) 2020 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import timeit
import os
import sys
import numpy as np
import pyqcstrc.ico.occupation_domain as od

# --------------------------------------
# PREDIFINED 6D POSITIONs in TAU style
# --------------------------------------

# Origin 0,0,0,0,0,0
POS_O0=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
POS_EC=np.array([[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])

# --------------------------------------
# Asymmetric part of m35
# Difinition of three vertces of asymmetric part of three times larger RT OD
# --------------------------------------

VRTX_ASYM1=np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
VRTX_ASYM2=np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
VRTX_ASYM3=np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 0, 0, 2],[-1, 0, 2],[ 0, 0, 2]])
tmp=np.append(POS_O0,VRTX_ASYM1)
tmp=np.append(tmp,VRTX_ASYM2)
tmp=np.append(tmp,VRTX_ASYM3)
asym=tmp.reshape(1,4,6,3)

# --------------------------------------
# Asymmetric part of m-5
# --------------------------------------
#VRTX_ASYM1 =np.array([[ 3 ,0, 2],[-3, 0, 2],[-3, 0, 2],[-3, 0, 2],[-3, 0, 2],[-3, 0, 2]])
#VRTX_ASYM4 =np.array([[ 0, 0, 1],[-3, 0, 4],[-3, 0, 4],[ 3, 0, 2],[-3, 0, 2],[ 3, 0, 2]])
#VRTX_ASYM5 =np.array([[ 0, 0, 1],[ 3, 0, 4],[-3, 0, 2],[ 3, 0, 2],[-3, 0, 2],[ 3, 0, 4]])
VRTX_ASYM1=np.array([[ 1, 0, 2],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
VRTX_ASYM4 =np.array([[0, 0, 1],[ 1, 0, 2],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[ 0, 0, 1]])
VRTX_ASYM5 =np.array([[0, 0, 1],[ 0, 0, 1],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[ 1, 0, 2]])
tmp=np.append(POS_O0,VRTX_ASYM1)
tmp=np.append(tmp,VRTX_ASYM4)
tmp=np.append(tmp,VRTX_ASYM5)
asym_m5=tmp.reshape(1,4,6,3)

if __name__ == '__main__':
   
   working_path='./test'
   if os.path.exists(working_path) == False:
       os.makedirs(working_path)
   else:
       pass
   
   #-------------------
   # Asymmetric part m35
   #-------------------
   file_name='RT_asymmeric'
   od.write_xyz(asym,working_path,file_name)
   od.write_vesta(asym,working_path,file_name)
   
   #-------------------
   # Symmetric part m35
   #-------------------
   sym=od.symmetric(asym,POS_O0)
   file_name='RT_symmeric'
   od.write_xyz(sym,working_path,file_name)
   od.write_vesta(sym,working_path,file_name)
   
    
   #-------------------------------
   # Asymmetric part m-5
   #-------------------------------
   file_name='EC_asymmeric'
   od.write_xyz(asym_m5,working_path,file_name)
   od.write_vesta(asym_m5,working_path,file_name)
   
