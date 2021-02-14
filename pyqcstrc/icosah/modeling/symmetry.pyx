import sys
import numpy as np
cimport numpy as np
cimport cython

from pyqcstrc.icosah.modeling.math1 cimport add

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
    cdef long j,k,a1,a2,a3,b1,b2,b3
    cdef list vec1
    vec1=[]
    for k in range(6):
        b1,b2,b3=0,0,1
        for j in range(6):
            a1=vec[j][0]*symop[k][j]
            a2=vec[j][1]*symop[k][j]
            a3=vec[j][2]
            #print a1,a2,a3,b1,b2,b3
            [b1,b2,b3]=add(b1,b2,b3,a1,a2,a3)
        [b1,b2,b3]=add(b1,b2,b3,centre[k][0],centre[k][1],centre[k][2])
        vec1.extend([b1,b2,b3])
    return vec1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_obj_symmetric_surface(np.ndarray[DTYPE_int_t, ndim=3] obj,
                                                np.ndarray[DTYPE_int_t, ndim=2] centre):
    cdef int i
    cdef list od,mop
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4
    od=[]
    mop=icosasymop()
    for i in range(len(mop)):
        od.extend(symop_obj(mop[i],obj,centre))
    tmp4=np.array(od).reshape(int(len(od)/54),3,6,3) # 54=3*6*3
    #print(' Number of triangles on POD surface: %d'%(len(tmp4)))
    return tmp4

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_obj_symmetric_tetrahedron(np.ndarray[DTYPE_int_t, ndim=3] obj,
                                                    np.ndarray[DTYPE_int_t, ndim=2] centre):
    cdef int i
    cdef list od,mop
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4
    #print('Generating symmetric POD')
    od=[]
    mop=icosasymop()
    for i in range(len(mop)):
        od.extend(symop_obj(mop[i],obj,centre))
    tmp4=np.array(od).reshape(int(len(od)/72),4,6,3) # 72=4*6*3
    #print(' Number of tetrahedron: %d'%(len(tmp4)))
    return tmp4

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_obj_symmetric_tetrahedron_specific_symop(np.ndarray[DTYPE_int_t, ndim=3] obj,
                                                                    np.ndarray[DTYPE_int_t, ndim=2] centre,
                                                                    list symmetry_operation):
    # using specific symmetry operations
    cdef int i
    cdef list od
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4
    #print('Generating symmetric POD')
    od=[]
    for i in range(len(symmetry_operation)):
        od.extend(symop_obj(symmetry_operation[i],obj,centre))
    tmp4=np.array(od).reshape(int(len(od)/72),4,6,3) # 72=4*6*3
    #print(' Number of tetrahedron: %d'%(len(tmp4)))
    return tmp4

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_obj_symmetric_tetrahedron_0(np.ndarray[DTYPE_int_t, ndim=3] obj,
                                                        np.ndarray[DTYPE_int_t, ndim=2] centre,
                                                        int numop):
    cdef int i
    cdef list od,mop
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4
    od=[]
    #print('Generating asymmetric POD')
    mop=icosasymop()
    od.extend(symop_obj(mop[numop],obj,centre))
    tmp4=np.array(od).reshape(int(len(od)/72),4,6,3) # 72=4*6*3
    #print(' Number of tetrahedron: %d'%(len(tmp4)))
    return tmp4

@cython.boundscheck(False)
@cython.wraparound(False)
cdef icosasymop():
    # icosahedral symmetry operations
    cdef int i,j,k,l,m
    cdef np.ndarray[DTYPE_int_t, ndim=2] s1,s2,s3,s4,s5,m1,m2,m3,m4,m5
    cdef list symop
    m1=np.array([[ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 1, 0, 0, 0, 0]])
    # mirror
    m2=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0,-1],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0,-1, 0, 0, 0]])
    # c2
    m3=np.array([[ 0, 0, 0, 0, 0,-1],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [-1, 0, 0, 0, 0, 0]])
    # c3
    m4=np.array([[ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0]])
    # inversion
    m5=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0,-1]])
    symop=[]
    for m in range(2):
        for l in range(3):
            for k in range(2):
                for j in range(2):
                    for i in range(5):
                        s1=matrix_pow(m1,i) # c5
                        s2=matrix_pow(m2,j) # mirror
                        s3=matrix_pow(m3,k) # c2
                        s4=matrix_pow(m4,l) # c3
                        s5=matrix_pow(m5,m) # inversion
                        tmp=matrix_dot(s5,s4)
                        tmp=matrix_dot(tmp,s3)
                        tmp=matrix_dot(tmp,s2)
                        tmp=matrix_dot(tmp,s1)
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

