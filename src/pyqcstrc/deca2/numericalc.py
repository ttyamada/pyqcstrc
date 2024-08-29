#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import numpy as np
from numpy.typing import NDArray
import random

EPS=1e-6 # tolerance
PI = np.pi
TAU = (np.sqrt(5)+1)/2.0
CONST1 =  1/np.sqrt(5)

C1 = np.cos(2*PI*1/5)
C2 = np.cos(2*PI*2/5)
C3 = np.cos(2*PI*3/5)
C4 = np.cos(2*PI*4/5)
C5 = np.cos(2*PI*5/5)
S1 = np.sin(2*PI*1/5)
S2 = np.sin(2*PI*2/5)
S3 = np.sin(2*PI*3/5)
S4 = np.sin(2*PI*4/5)
S5 = np.sin(2*PI*5/5)
C21 = np.cos(4*PI*1/5)
C22 = np.cos(4*PI*2/5)
C23 = np.cos(4*PI*3/5)
C24 = np.cos(4*PI*4/5)
C25 = np.cos(4*PI*5/5)
S21 = np.sin(4*PI*1/5)
S22 = np.sin(4*PI*2/5)
S23 = np.sin(4*PI*3/5)
S24 = np.sin(4*PI*4/5)
S25 = np.sin(4*PI*5/5)

def projection_5d(vn: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    parallel component of 5D direct lattice vector, x and y in 2*a/np.sqrt(5) unit, and z in c unit.
    input
    list:r, 5D reflection index
    """
    mat=np.array([
        [ C1-1, C2-1, C3-1, C4-1, 0],\
        [   S1,   S2,   S3,   S4, 0],\
        [    0,    0,    0,    0, 1],\
        [ C2-1, C4-1, C1-1, C3-1, 0],\
        [   S2,   S4,   S1,   S3, 0],\
    ])
    return mat@vn
    
def projection3_5d(vn: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    perpendicular component of 5D reciprocal lattice vector, in 2*a/np.sqrt(5) unit.
    input
    list:r, 5D reflection index
    float:a, lattice constant
    """
    mat=np.array([
        [ C2-1, C4-1, C1-1, C3-1, 0],\
        [   S2,   S4,   S1,   S3, 0],\
    ])
    return mat@vn

def projection_6d(vn: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    parallel component of 6D direct lattice vector, x and y in 2*a/np.sqrt(5) unit, and z in c unit.
    
    """
    mat=np.array([
        [ C1,  C2,  C3,  C4,  C5, 0],\
        [ S1,  S2,  S3,  S4,  S5 ,0],\
        [  0,   0,   0,   0,   0, 1],\
        [C21, C22, C23, C24, C25, 0],\
        [S21, S22, S23, S24, S25, 0],\
    ])
    return mat@vn
    
def projection3_6d(vn: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    perpendicular component of 6D direct lattice vector, x and y in 2*a/np.sqrt(5) unit.
    """
    mat=np.array([
        [C21, C22, C23, C24, C25, 0],\
        [S21, S22, S23, S24, S25, 0],\
    ])
    return mat@vn
    
def transform6to5(vn: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    this function transforms a 6-dim coordinates to its corresponding 5-dim coordinates.
    """
    mat = np.array([
        [ 4,-1,-1,-1,-1, 0],\
        [-1, 4,-1,-1,-1, 0],\
        [-1,-1, 4,-1,-1, 0],\
        [-1,-1,-1, 4,-1, 0],\
        [ 0, 0, 0, 0, 0, 5],\
    ])/5.0
    return mat@x

def projection_sets_numerical(vns: NDArray[np.float64]) -> NDArray[np.float64]:
    """parallel and perpendicular components of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    num=len(vns)
    m=np.zeros((num,6),dtype=np.float64)
    for i in range(num):
        m[i]=projection_6d(vns[i])
    return m
    
def projection3_sets_numerical(vns: NDArray[np.float64]) -> NDArray[np.float64]:
    """perpendicular component of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    ndim=vns.ndim
    if ndim==2:
        n1,_=vns.shape
        m=np.zeros((n1,2),dtype=np.float64)
        for i in range(n1):
            m[i]=projection3_6d(vns[i])
    elif ndim==3:
        n1,n2,_=vns.shape
        m=np.zeros((n1,n2,2),dtype=np.float64)
        for i1 in range(n1):
            for i2 in range(n2):
                m[i1][i2]=projection3_6d(vns[i1][i2])
    elif ndim==4:
        n1,n2,n3,_=vns.shape
        m=np.zeros((n1,n2,n3,2),dtype=np.float64)
        for i1 in range(n1):
            for i2 in range(n2):
                for i3 in range(n3):
                    m[i1][i2][i3]=projection3_6d(vns[i1][i2][i3])
    elif ndim==5:
        n1,n2,n3,n4,_=vns.shape
        m=np.zeros((n1,n2,n3,n4,2),dtype=np.float64)
        for i1 in range(n1):
            for i2 in range(n2):
                for i3 in range(n3):
                    for i4 in range(n4):
                    m[i1][i2][i3][i4]=projection3_6d(vns[i1][i2][i3][i4])
    return m
