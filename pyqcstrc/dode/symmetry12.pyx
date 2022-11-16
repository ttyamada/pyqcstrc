#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
cimport numpy as np
cimport cython

#from pyqcstrc.dode.math2 cimport add
from pyqcstrc.dode.math12 cimport add,sub
from pyqcstrc.dode.utils12 cimport remove_doubling_dim3, remove_doubling_dim3_in_perp_space

DTYPE_double = np.float64
DTYPE_int = int

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list symop_obj(np.ndarray[DTYPE_int_t,ndim=2] symop,
                    np.ndarray[DTYPE_int_t, ndim=3] obj,
                    np.ndarray[DTYPE_int_t, ndim=2] centre):
    cdef int i
    cdef list obj1
    obj1=[]
    for i in range(len(obj)):
        obj1.extend(symop_vec(symop,obj[i],centre))
    return obj1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list symop_vec(np.ndarray[DTYPE_int_t,ndim=2] symop,
                    np.ndarray[DTYPE_int_t,ndim=2] vec,
                    np.ndarray[DTYPE_int_t, ndim=2] centre):
    cdef np.ndarray[DTYPE_int_t,ndim=2] vec1,vec2
    vec1=shift_vec_pull(vec,centre)
    vec2=symop_vec_0(symop,vec1)
    vec1=shift_vec_push(vec2,centre)
    return vec1.tolist()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray shift_vec_push(np.ndarray[DTYPE_int_t,ndim=2] vec,
                               np.ndarray[DTYPE_int_t, ndim=2] shift):
    cdef int i1
    cdef long n1,n2,n3
    cdef list a,b
    a=[]
    for i1 in range(6):
        b=add(vec[i1][0],vec[i1][1],vec[i1][2],shift[i1][0],shift[i1][1],shift[i1][2])
        a.append(b)
    return np.array(a)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray shift_vec_pull(np.ndarray[DTYPE_int_t,ndim=2] vec,
                               np.ndarray[DTYPE_int_t, ndim=2] shift):
    cdef int i1
    cdef long n1,n2,n3
    cdef list a,b
    a=[]
    for i1 in range(6):
        b=sub(vec[i1][0],vec[i1][1],vec[i1][2],shift[i1][0],shift[i1][1],shift[i1][2])
        a.append(b)
    return np.array(a)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray symop_vec_0(np.ndarray[DTYPE_int_t,ndim=2] symop,
                            np.ndarray[DTYPE_int_t,ndim=2] vec):
    """
    vector: from the origin to a point defined by vec
    symmetric operation on the vector. Note that the symmetric center is the origin.
    """
    cdef int j,k,
    cdef long a1,a2,a3,b1,b2,b3
    cdef list vec1
    vec1=[]
    for k in range(6):
        b1,b2,b3=0,0,1
        for j in range(6):
            a1=vec[j][0]*symop[k][j]
            a2=vec[j][1]*symop[k][j]
            a3=vec[j][2]
            [b1,b2,b3]=add(b1,b2,b3,a1,a2,a3)
        vec1.extend([b1,b2,b3])
    return np.array(vec1).reshape(6,3)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_obj_symmetric_triangle(np.ndarray[DTYPE_int_t, ndim=3] obj,
                                                    np.ndarray[DTYPE_int_t, ndim=2] centre):
    cdef int i
    cdef list od,mop
    #cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4
    od=[]
    mop=dodesymop()
    for i in range(len(mop)):
        od.extend(symop_obj(mop[i],obj,centre))
    #tmp4=np.array(od).reshape(int(len(od)/54),3,6,3) # 54=3*6*3
    #return tmp4
    return np.array(od).reshape(int(len(od)/54),3,6,3) # 54=3*6*3

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_obj_symmetric_triangle_specific_symop(np.ndarray[DTYPE_int_t, ndim=3] obj,
                                                                    np.ndarray[DTYPE_int_t, ndim=2] centre,
                                                                    list symmetry_operation):
    cdef int i
    cdef list od
    #cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4
    od=[]
    for i in range(len(symmetry_operation)):
        od.extend(symop_obj(symmetry_operation[i],obj,centre))
    #tmp4=np.array(od).reshape(int(len(od)/54),3,6,3) # 54=3*6*3
    #return tmp4
    return np.array(od).reshape(int(len(od)/54),3,6,3) # 54=3*6*3
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_symmetric_obj_0(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                            np.ndarray[DTYPE_int_t, ndim=2] centre,
                                            int numop):
    cdef int i
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,tmp4b
    
    for i in range(len(obj)):
        tmp4a=generator_obj_symmetric_triangle_0(obj[i],centre,numop)
        if i==0:
            tmp4b=tmp4a
        else:
            tmp4b=np.vstack([tmp4b,tmp4a])
    return tmp4b

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_obj_symmetric_triangle_0(np.ndarray[DTYPE_int_t, ndim=3] obj,
                                                    np.ndarray[DTYPE_int_t, ndim=2] centre,
                                                    int numop):
    cdef int i
    cdef list od,mop
    #cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4
    od=[]
    mop=dodesymop()
    od.extend(symop_obj(mop[numop],obj,centre))
    #tmp4=np.array(od).reshape(int(len(od)/54),3,6,3) # 54=3*6*3
    #return np.array(od).reshape(int(len(od)/54),3,6,3) # 54=3*6*3
    return np.array(od).reshape(1,3,6,3) # 54=3*6*3
    #return tmp4

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_obj_symmetric_vec(np.ndarray[DTYPE_int_t, ndim=3] vectors,
                                            np.ndarray[DTYPE_int_t, ndim=2] centre):
    cdef int i1,i2
    cdef list od,mop
    #cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3
    
    od=[]
    mop=dodesymop()
    for i1 in range(len(mop)):
        for i2 in range(len(vectors)):
            od.extend(symop_vec(mop[i1],vectors[i2],centre))
    #tmp3=np.array(od).reshape(24,int(len(od)/18/24),6,3) # 18=6*3
    return np.array(od).reshape(24,int(len(od)/18/24),6,3) # 18=6*3
    #return tmp3

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_obj_symmetric_vec_0(np.ndarray[DTYPE_int_t, ndim=2] vector, int numop):
    cdef int i1,i2
    cdef list od,mop
    cdef np.ndarray[DTYPE_int_t,ndim=2] centre
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3
    
    centre=np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,0,0,0)
    mop=dodesymop()
    od=symop_vec(mop[numop],vector,centre)
    return np.array(od).reshape(6,3)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray similarity_obj(np.ndarray[DTYPE_int_t, ndim=3] obj, int m):
    cdef int i
    cdef list obj1
    obj1=[]
    for i in range(len(obj)):
        obj1.extend(similarity_vec(obj[i],m))
    return np.array(obj1).reshape(int(len(obj1)/54),3,6,3)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list similarity_vec(np.ndarray[DTYPE_int_t, ndim=2] vector, int m):
    cdef int j,k,
    cdef long a1,a2,a3,b1,b2,b3
    cdef list vec1
    cdef np.ndarray[DTYPE_int_t,ndim=2] op
    
    vec1=[]
    op=similarity(m)
    for k in range(6):
        b1,b2,b3=0,0,1
        for j in range(6):
            a1=vector[j][0]*op[k][j]
            a2=vector[j][1]*op[k][j]
            a3=vector[j][2]
            [b1,b2,b3]=add(b1,b2,b3,a1,a2,a3)
        vec1.extend([b1,b2,b3])
    return vec1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_equivalent_vec(np.ndarray[DTYPE_int_t, ndim=2] vector,
                                            np.ndarray[DTYPE_int_t, ndim=2] centre):
    cdef int i1,i2
    cdef list od,mop
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3
    
    od=[]
    mop=dodesymop()
    for i1 in range(len(mop)):
        od.extend(symop_vec(mop[i1],vector,centre))
    tmp3=np.array(od).reshape(int(len(od)/18),6,3) # 18=6*3
    return remove_doubling_dim3(tmp3)
    #return remove_doubling_dim3_in_perp_space(tmp3)
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list site_symmetry(np.ndarray[DTYPE_int_t, ndim=2] site):
    
    cdef np.ndarray[DTYPE_int_t, ndim=2] centre,dev,vec,v1
    cdef np.ndarray[DTYPE_int_t, ndim=3] tmp3a,tmp3b,trans
    cdef long b1,b2,b3
    cdef int i,k
    cdef list vec1,mop,numlst
    
    vec1=[]
    centre=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    dev=np.array([[-3,4,1],[2,-4,1],[-4,3,1],[3,1,1],[0,0,1],[0,0,1]])
    
    # translation
    v1=np.array([[ 1, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (1,0,0,0)
    trans=generator_equivalent_vec(v1,centre)
    
    for k in range(6):
        [b1,b2,b3]=add(site[k][0],site[k][1],site[k][2],dev[k][0],dev[k][1],dev[k][2])
        vec1.extend([b1,b2,b3])
    vec=np.array(vec1).reshape(6,3)
    tmp3a=generator_equivalent_vec(vec,centre)
    
    # translational symmetry
    vec1=[]
    for i in range(len(tmp3a)):
        for j in range(len(trans)):
            for k in range(6):
                [b1,b2,b3]=sub(tmp3a[i][k][0],tmp3a[i][k][1],tmp3a[i][k][2],trans[j][k][0],trans[j][k][1],trans[j][k][2])
                vec1.extend([b1,b2,b3])
    tmp3b=np.array(vec1).reshape(int(len(vec1)/18),6,3)
    tmp3a=np.vstack([tmp3a,tmp3b])
    
    #print('1: len(tmp3a)=',len(tmp3a))
    vec1=[]
    for i in range(len(tmp3a)):
        for k in range(6):
            [b1,b2,b3]=sub(tmp3a[i][k][0],tmp3a[i][k][1],tmp3a[i][k][2],site[k][0],site[k][1],site[k][2])
            vec1.extend([b1,b2,b3])
    tmp3a=np.array(vec1).reshape(len(tmp3a),6,3)
    #print(vec1)
    
    #print('2: len(tmp3a)=',len(tmp3a))
    mop=dodesymop()
    
    numlst=[]
    for i in range(len(tmp3a)):
        for k in range(len(mop)):
            vec1=symop_vec(mop[k],dev,centre)
            if np.array_equal(np.array(vec1).reshape(6,3),tmp3a[i]):
                numlst.extend([k])
            else:
                pass
    return numlst

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray similarity(int m):
    # Similarity transformation of Dodecagonal QC
    cdef np.ndarray[DTYPE_int_t, ndim=2] m1
    if m>0:
        m1=np.array([[ 1, 0, 0, -1, 0, 0],\
                    [ 1, 1, 0, 0, 0, 0],\
                    [ 0, 1, 1, 1, 0, 0],\
                    [ 0, 0, 1, 1, 0, 0],\
                    [ 0, 0, 0, 0, 1, 0],\
                    [ 0, 0, 0, 0, 0, 1]])
        return matrix_pow(m1.T,m)
    elif m<0:
        m1=np.array([[ 0, 1,-1, 1, 0, 0],\
                   [ 0, 0, 1,-1, 0, 0],\
                   [ 1,-1, 1, 0, 0, 0],\
                   [-1, 1,-1, 1, 0, 0],\
                   [ 0, 0, 0, 0, 1, 0],\
                   [ 0, 0, 0, 0, 0, 1]])
        return matrix_pow(m1.T,-m)
    else:
        m1=np.array([[ 1, 0, 0, 0, 0, 0],\
                   [ 0, 1, 0, 0, 0, 0],\
                   [ 0, 0, 1, 0, 0, 0],\
                   [ 0, 0, 0, 1, 0, 0],\
                   [ 0, 0, 0, 0, 1, 0],\
                   [ 0, 0, 0, 0, 0, 1]])
        return m1.T

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list dodesymop():
    # icosahedral symmetry operations
    cdef int i1,i2
    cdef np.ndarray[DTYPE_int_t, ndim=2] s1,s2,m1,m2
    cdef list symop
    # c12
    m1=np.array([[ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [-1, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]])
    # mirror
    m2=np.array([[ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 1, 0, 0, 0, 0],\
                [ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]])
    symop=[]
    for i1 in range(2):
        for i2 in range(12):
            s1=matrix_pow(m1,i2) # c12
            s2=matrix_pow(m2,i1) # mirror
            tmp=matrix_dot(s2,s1)
            symop.append(tmp)
    return symop

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray matrix_pow(np.ndarray[DTYPE_int_t, ndim=2] array_1, int n):
    cdef Py_ssize_t mx
    cdef Py_ssize_t my
    cdef Py_ssize_t x,y,z
    cdef int i
    mx = array_1.shape[0]
    my = array_1.shape[1]
    array_2 = np.identity(mx, dtype=int)
    if mx == my:
        if n == 0:
            return np.identity(mx, dtype=int)
        elif n<0:
            return np.zeros((mx,mx), dtype=int)
        else:
            for i in range(n):
                tmp = np.zeros((6, 6), dtype=int)
                for x in range(array_2.shape[0]):
                    for y in range(array_1.shape[1]):
                        for z in range(array_2.shape[1]):
                            tmp[x][y] += array_2[x][z] * array_1[z][y]
                array_2 = tmp
            return array_2
    else:
        print('ERROR: matrix has not regular shape')
        return np.array([0])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray matrix_dot(np.ndarray[DTYPE_int_t, ndim=2] array_1, np.ndarray[DTYPE_int_t, ndim=2] array_2):
    cdef Py_ssize_t mx1,my1,mx2,my2
    cdef Py_ssize_t x,y,z    
    mx1 = array_1.shape[0]
    my1 = array_1.shape[1]
    mx2 = array_2.shape[0]
    my2 = array_2.shape[1]
    array_3 = np.zeros((mx1,my2), dtype=int)
    for x in range(array_1.shape[0]):
        for y in range(array_2.shape[1]):
            for z in range(array_1.shape[1]):
                array_3[x][y] += array_1[x][z] * array_2[z][y]
    return array_3

