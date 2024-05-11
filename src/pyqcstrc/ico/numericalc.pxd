#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
cimport numpy as np

ctypedef np.float64_t DTYPE_double_t
ctypedef long DTYPE_int_t

cpdef int point_on_segment(np.ndarray[DTYPE_int_t, ndim=2] point,
                            np.ndarray[DTYPE_int_t, ndim=2],
                            np.ndarray[DTYPE_int_t, ndim=2])
                            
cpdef double obj_volume_6d_numerical(np.ndarray[DTYPE_int_t, ndim=4])

cpdef double tetrahedron_volume_6d_numerical(np.ndarray[DTYPE_int_t, ndim=3])

cpdef list projection_numerical(DTYPE_double_t, DTYPE_double_t, DTYPE_double_t, DTYPE_double_t, DTYPE_double_t, DTYPE_double_t)

cpdef list projection3_numerical(DTYPE_double_t, DTYPE_double_t, DTYPE_double_t, DTYPE_double_t, DTYPE_double_t, DTYPE_double_t)

cpdef int inside_outside_obj(list,np.ndarray[DTYPE_int_t, ndim=4])

cpdef int inside_outside_tetrahedron(np.ndarray[DTYPE_int_t, ndim=2],
                                    np.ndarray[DTYPE_int_t, ndim=2],
                                    np.ndarray[DTYPE_int_t, ndim=2],
                                    np.ndarray[DTYPE_int_t, ndim=2],
                                    np.ndarray[DTYPE_int_t, ndim=2])

cpdef int check_intersection_segment_surface_numerical(np.ndarray[DTYPE_int_t, ndim=2],
                                                        np.ndarray[DTYPE_int_t, ndim=2],
                                                        np.ndarray[DTYPE_int_t, ndim=2],
                                                        np.ndarray[DTYPE_int_t, ndim=2],
                                                        np.ndarray[DTYPE_int_t, ndim=2])
