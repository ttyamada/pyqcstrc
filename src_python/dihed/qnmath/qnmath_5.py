import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import qnndarray as qna
from numpy.typing import(NDArray)

def centroid(obj: qnv.Qnvec) -> qnv.Qnvec:
    """geometric center, centroid of tetrahedron, triangle or edge, in qnvec.

    Parameters
    ----------
    obj: array
        6-dimensional vector in qnvec
    
    Returns
    -------
    centroid in qnvec
    """
    #N=obj[0].N
    n_=obj.shape[0]
    num=len(obj) # length of obj
    v2=qnv.Qnvec(n_) # nD zero qnvector
    qnnum=qnn.Qnnum([1,0,num]) # 1/num
    for i1 in range(num):
        v2=v2+obj[i1]
    v0=v2*qnnum
    return v0

# needless???
def centroid_obj(obj: qnv.Qnvec) -> qnv.Qnvec:
    """geometric center, centroid of tetrahedron, in qnvec.

    Parameters
    ----------
    tetrahedron: array
        6-dimensional vector in qnvec
    
    Returns
    -------
    centroid in qnvec
    """
    #print('centroid_obj')
    
    #  geometric center, centroid of OBJ
    #N=obj[0].N
    shape=qnv.shape
    n=shape[1]
    len=shape[0]  # 1/len(obj)
    tmp=qnv.Qnvec(n) # zero vector
    for thd in obj:
        tmp=tmp+thd
    tmp=tmp*len
    return tmp

def det_matrix(mtx: qnm.Qnmat, n_:np.int64) -> qnn.Qnnum:
    if n_==2:
        return det_matrix_2d(mtx)
    elif n_==3:
        return det_matrix_3d(mtx)

def det_matrix_3d(mtx: qnm.Qnmat) -> qnn.Qnnum:
    """Determinant of 3x3 matrix, mtx, in qnnumber
    
    Parameters
    ----------
    mtx: array
        3x3 matrix in qnnumer

    Returns
    -------
    determinant in qnnumber
    """
    #N=mtx.N
    shape=mtx.shape
    if shape[0]!=3:
        print("shape of mtx in det_matrix_3d should be (3,3) but",shape); exit(0)
    det=qnn.Qnnum([0,0,1]) # zero qnnumber
    det=det+mtx[0][0]*mtx[1][1]*mtx[2][2]
    det=det+mtx[0][1]*mtx[1][2]*mtx[2][0]    
    det=det+mtx[0][2]*mtx[1][0]*mtx[2][1]
    det=det-mtx[0][2]*mtx[1][1]*mtx[2][0]
    det=det-mtx[0][1]*mtx[1][0]*mtx[2][2]    
    det=det-mtx[0][0]*mtx[1][2]*mtx[2][1]

    return det
