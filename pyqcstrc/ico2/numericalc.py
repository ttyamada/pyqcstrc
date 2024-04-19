#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import numpy as np

TAU=(1+np.sqrt(5))/2.0
EPS=1e-6

def numeric_value(a):
    """Numeric value of a TAU-style value, a.

    Parameters
    ----------
    a: array
        value in TAU-style
    
    Returns
    -------
    float
    """
    return (a[0]+a[1]*TAU)/a[2]

def numerical_vector(v):
    """Numeric value of a TAU-style vector, v.

    Parameters
    ----------
    v: array
        vector in TAU-style
    
    Returns
    -------
    array
    """
    n=len(v)
    w=np.zeros(n,dtype=np.float64)
    for i in range(n):
        w[i]=numeric_value(v[i])
    return w


def int inside_outside_tetrahedron(point,tetrahedron):
    """this function judges whether the point is inside a traiangle or not
        
    Parameters
    ----------
    point: array
        vertex coordinates of the triangle,xyz0, xyz1, xyz2, xyz3
    tetrahedron: array
        coordinate of the point,xyz4
    """
    volume0=tetrahedron_volume_6d_numerical(tetrahedron)
    #
    tetrahedron1=np.append(point,tetrahedron_v2)
    tetrahedron1=np.append(tetrahedron1,tetrahedron_v3)
    tetrahedron1=np.append(tetrahedron1,tetrahedron_v4)
    tetrahedron1=tetrahedron1.reshape(4,6,3)
    volume1=tetrahedron_volume_6d_numerical(tetrahedron1)
    #
    tetrahedron2=np.append(tetrahedron_v1,point)
    tetrahedron2=np.append(tetrahedron2,tetrahedron_v3)
    tetrahedron2=np.append(tetrahedron2,tetrahedron_v4)
    tetrahedron2=tetrahedron2.reshape(4,6,3)
    volume2=tetrahedron_volume_6d_numerical(tetrahedron2)
    #
    tetrahedron3=np.append(tetrahedron_v1,tetrahedron_v2)
    tetrahedron3=np.append(tetrahedron3,point)
    tetrahedron3=np.append(tetrahedron3,tetrahedron_v4)
    tetrahedron3=tetrahedron3.reshape(4,6,3)
    volume3=tetrahedron_volume_6d_numerical(tetrahedron3)
    #
    tetrahedron4=np.append(tetrahedron_v1,tetrahedron_v2)
    tetrahedron4=np.append(tetrahedron4,tetrahedron_v3)
    tetrahedron4=np.append(tetrahedron4,point)
    tetrahedron4=tetrahedron4.reshape(4,6,3)
    volume4=tetrahedron_volume_6d_numerical(tetrahedron4)
    #
    if abs(volume0-volume1-volume2-volume3-volume4)<EPS*volume0:
        return 0 # inside
    else:
        return 1 # outside




def matrix_dot(m1,m2):
    return np.dot(m1,m2)
    
def matrixpow(ma,n):
    """
    """
    (mx,my)=ma.shape
    if mx==my:
        if n==0:
            return np.identity(mx)
        elif n<0:
            tmp=np.identity(mx)
            inva = np.linalg.inv(ma)
            for i in range(-n):
                tmp=np.dot(tmp,inva)
            return tmp
        else:
            tmp=np.identity(mx)
            for i in range(n):
                tmp=np.dot(tmp,ma)
            return tmp
    else:
        print('matrix has not regular shape')
        return 


###########################
#     Numerical Calc      #
###########################

def obj_volume_6d_numerical(obj):
    """This function returns volume of an object (set of tetrahedra).
        
    Parameters
    ----------
    object: array
        6-dimensional vertex coordinates of the tetrahedron, in TAU-style.
    """
    vol=0
    for tetrahedron in obj:
        vol+=tetrahedron_volume_6d_numerical(tetrahedron)
    return vol

def tetrahedron_volume_6d_numerical(tetrahedron):
    """This function returns volume of a tetrahedron
        
    Parameters
    ----------
    tetrahedron: array
        6-dimensional vertex coordinates of the tetrahedron, in TAU-style.
    """
    xyz0=get_internal_component_numerical(tetrahedron[0])
    xyz1=get_internal_component_numerical(tetrahedron[1])
    xyz2=get_internal_component_numerical(tetrahedron[2])
    xyz3=get_internal_component_numerical(tetrahedron[3])
    tmp1=np.stack([xyz0, xyz1])
    tmp2=np.stack([xyz2, xyz3])
    tmp1=np.stack([tmp1, tmp2])
    return tetrahedron_volume_numerical(tmp1)

def tetrahedron_volume_numerical(tetrahedron):
    """This function returns volume of a tetrahedron
        
    Parameters
    ----------
    tetrahedron: array
        vertex coordinates of the tetrahedron, (x0,y0,z0),(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)
    """
    xyz1=tetrahedron[1]-tetrahedron[0]
    xyz2=tetrahedron[2]-tetrahedron[0]
    xyz3=tetrahedron[3]-tetrahedron[0]
    xyz1=np.stack([xyz1, xyz2])
    xyz1=np.stack([xyz1, xyz3])
    detm = np.linalg.det(xyz1)
    return abs(detm)/6.0
    
def get_internal_component_numerical(vt):
    """
    Parameters
    ----------
    vt: array
        6-dimensional vector in TAU-style.
    """
    vn=numerical_vector(vt)
    return projection3_numerical(vn)

def projection_numerical(vn):
    """parallel and perpendicular components of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vn: array
        6-dimensional vector, numerial values
    """
    v1 =  (vn[0]-vn[4]) + TAU*(vn[1]+vn[2]) # x in Epar
    v2 =  (vn[3]+vn[5]) + TAU*(vn[0]+vn[4]) # y in Epar
    v3 =  (vn[1]-vn[2]) - TAU*(vn[3]-vn[5]) # z in Epar
    v4 = -(vn[1]+vn[2]) + TAU*(vn[0]-vn[4]) # x in Eperp
    v5 = -(vn[0]+vn[4]) + TAU*(vn[3]+vn[5]) # y in Eperp
    v6 =  (vn[3]-vn[5]) + TAU*(vn[1]-vn[2]) # z in Eperp
    return np.array([v1,v2,v3,v4,v5,v6],dtype=np.float64)

def projection3_numerical(vn):
    """perpendicular components of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vn: array
        6-dimensional vector, numerial values
    """
    v4 = -(vn[1]+vn[2]) + TAU*(vn[0]-vn[4]) # x in Eperp
    v5 = -(vn[0]+vn[4]) + TAU*(vn[3]+vn[5]) # y in Eperp
    v6 =  (vn[3]-vn[5]) + TAU*(vn[1]-vn[2]) # z in Eperp
    return np.array([v4,v5,v6],dtype=np.float64)

if __name__ == '__main__':
    
    a=np.array([1,1,1])
    a=numeric_value(a)
    print(a)