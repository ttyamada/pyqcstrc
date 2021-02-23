#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
cimport numpy as np

ctypedef np.float64_t DTYPE_double_t
ctypedef long DTYPE_int_t

cpdef np.ndarray intersection_two_obj(np.ndarray[DTYPE_int_t, ndim=4],
                                    np.ndarray[DTYPE_int_t, ndim=4],
                                    int,
                                    int,
                                    int,
                                    int)

cpdef list intersection_two_obj_convex(np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=4],
                                        int)

cpdef np.ndarray tetrahedralization(np.ndarray[DTYPE_int_t, ndim=3],
                                    np.ndarray[DTYPE_int_t, ndim=3],
                                    int)

cpdef np.ndarray tetrahedralization_1(np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=2],
                                        int)

cpdef np.ndarray tetrahedralization_points(np.ndarray[DTYPE_int_t, ndim=3],
                                            int)

cpdef np.ndarray intersection_using_tetrahedron_4(np.ndarray[DTYPE_int_t, ndim=4],
                                                    np.ndarray[DTYPE_int_t, ndim=4],
                                                    int,
                                                    int)

cpdef np.ndarray intersection_two_tetrahedron_4(np.ndarray[DTYPE_int_t, ndim=3],
                                                np.ndarray[DTYPE_int_t, ndim=3],
                                                int)

cpdef np.ndarray intersection_tetrahedron_obj_4(np.ndarray[DTYPE_int_t, ndim=3],
                                                np.ndarray[DTYPE_int_t, ndim=4],
                                                int,
                                                int)
