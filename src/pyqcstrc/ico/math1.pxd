#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
cimport numpy as np

ctypedef np.float64_t DTYPE_double_t
ctypedef long DTYPE_int_t

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

cpdef list projection_perp(np.ndarray[DTYPE_int_t, ndim=1],
                            np.ndarray[DTYPE_int_t, ndim=1],
                            np.ndarray[DTYPE_int_t, ndim=1],
                            np.ndarray[DTYPE_int_t, ndim=1],
                            np.ndarray[DTYPE_int_t, ndim=1],
                            np.ndarray[DTYPE_int_t, ndim=1])

cpdef np.ndarray outer_product(np.ndarray[DTYPE_int_t, ndim=2],
                                np.ndarray[DTYPE_int_t, ndim=2])

cpdef list inner_product(np.ndarray[DTYPE_int_t, ndim=2],
                        np.ndarray[DTYPE_int_t, ndim=2])

cpdef int coplanar_check(np.ndarray[DTYPE_int_t, ndim=3])

cpdef list det_matrix(np.ndarray[DTYPE_int_t, ndim=2],
                     np.ndarray[DTYPE_int_t, ndim=2],
                     np.ndarray[DTYPE_int_t, ndim=2])

cpdef list dot_product(np.ndarray[DTYPE_int_t, ndim=1],
                        np.ndarray[DTYPE_int_t, ndim=1],
                        np.ndarray[DTYPE_int_t, ndim=1],
                        np.ndarray[DTYPE_int_t, ndim=1],
                        np.ndarray[DTYPE_int_t, ndim=1],
                        np.ndarray[DTYPE_int_t, ndim=1])

cpdef np.ndarray centroid(np.ndarray[DTYPE_int_t, ndim=3])

cpdef double triangle_area(np.ndarray[DTYPE_int_t, ndim=2],
                            np.ndarray[DTYPE_int_t, ndim=2],
                            np.ndarray[DTYPE_int_t, ndim=2])

cpdef list add(DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t) # A+B

cpdef list sub(DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t) # A-B

cpdef list mul(DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t) # A*B

cpdef list div(DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t ,DTYPE_int_t) # A/B

cpdef np.ndarray mul_vectors(np.ndarray[DTYPE_int_t, ndim=3], list)

cpdef np.ndarray mul_vector(np.ndarray[DTYPE_int_t, ndim=2], list)

cpdef np.ndarray add_vectors(np.ndarray[DTYPE_int_t, ndim=2], np.ndarray[DTYPE_int_t, ndim=2])
