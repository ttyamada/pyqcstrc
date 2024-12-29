#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
cimport numpy as np

ctypedef np.float64_t DTYPE_double_t
ctypedef long DTYPE_int_t

cpdef np.ndarray mul_vectors(np.ndarray[DTYPE_int_t, ndim=3],list)

cpdef np.ndarray mul_vector(np.ndarray[DTYPE_int_t, ndim=2],list)

cpdef np.ndarray add_vectors(np.ndarray[DTYPE_int_t, ndim=2],np.ndarray[DTYPE_int_t, ndim=2])

cpdef np.ndarray sub_vectors(np.ndarray[DTYPE_int_t, ndim=2],np.ndarray[DTYPE_int_t, ndim=2])

cpdef list projection(np.ndarray[DTYPE_int_t, ndim=1],
                    np.ndarray[DTYPE_int_t, ndim=1],
                    np.ndarray[DTYPE_int_t, ndim=1],
                    np.ndarray[DTYPE_int_t, ndim=1],
                    np.ndarray[DTYPE_int_t, ndim=1],
                    np.ndarray[DTYPE_int_t, ndim=1])

cpdef list projection3(np.ndarray[DTYPE_int_t, ndim=1],
                    np.ndarray[DTYPE_int_t, ndim=1],
                    np.ndarray[DTYPE_int_t, ndim=1],
                    np.ndarray[DTYPE_int_t, ndim=1],
                    np.ndarray[DTYPE_int_t, ndim=1],
                    np.ndarray[DTYPE_int_t, ndim=1])

cpdef list add(DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t) # A+B

cpdef list sub(DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t) # A-B

cpdef list mul(DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t) # A*B

cpdef list div(DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t) # A/B

cpdef int gcd(DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t) # A/B

cpdef list det_matrix(np.ndarray[DTYPE_int_t, ndim=2],
                     np.ndarray[DTYPE_int_t, ndim=2],
                     np.ndarray[DTYPE_int_t, ndim=2])

cpdef list dot_product(np.ndarray[DTYPE_int_t, ndim=1],
                        np.ndarray[DTYPE_int_t, ndim=1],
                        np.ndarray[DTYPE_int_t, ndim=1],
                        np.ndarray[DTYPE_int_t, ndim=1],
                        np.ndarray[DTYPE_int_t, ndim=1],
                        np.ndarray[DTYPE_int_t, ndim=1])

cpdef np.ndarray outer_product(np.ndarray[DTYPE_int_t, ndim=2],
                                np.ndarray[DTYPE_int_t, ndim=2])
