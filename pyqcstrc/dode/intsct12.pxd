#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
cimport numpy as np

ctypedef np.float64_t DTYPE_double_t
ctypedef long DTYPE_int_t

cpdef np.ndarray intersection_two_obj(np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=4],
                                        int)

cpdef np.ndarray intersection_triangle_obj(np.ndarray[DTYPE_int_t, ndim=3],
                                            np.ndarray[DTYPE_int_t, ndim=4],
                                            int)

cpdef np.ndarray triangulation(np.ndarray[DTYPE_int_t, ndim=3],
                                np.ndarray[DTYPE_int_t, ndim=3],
                                int)

cpdef np.ndarray intersection_line_segment_triangle(np.ndarray[DTYPE_int_t, ndim=3],
                                                    np.ndarray[DTYPE_int_t, ndim=3],
                                                    int)

cpdef np.ndarray intersection_two_segment(np.ndarray[DTYPE_int_t, ndim=2],
                                    np.ndarray[DTYPE_int_t, ndim=2],
                                    np.ndarray[DTYPE_int_t, ndim=2],
                                    np.ndarray[DTYPE_int_t, ndim=2],
                                    int)

cpdef np.ndarray intersection_two_triangles(np.ndarray[DTYPE_int_t, ndim=3],
                                            np.ndarray[DTYPE_int_t, ndim=3],
                                            int)

cpdef np.ndarray triangulation_points(np.ndarray[DTYPE_int_t, ndim=3],
                                        int)

