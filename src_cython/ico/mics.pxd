#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
cimport numpy as np

ctypedef np.float64_t DTYPE_double_t
ctypedef long DTYPE_int_t


cpdef np.ndarray generate_convex_hull(np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=2], 
                                        int,
                                        int)

cpdef np.ndarray simplification_obj_edges_1(np.ndarray[DTYPE_int_t, ndim=4], int, int)

cpdef np.ndarray simplification_obj_edges(np.ndarray[DTYPE_int_t, ndim=4], int, int)

cpdef np.ndarray simplification_obj_smart(np.ndarray[DTYPE_int_t, ndim=4], int, int)

cpdef np.ndarray simplification_obj_edges_using_parents(np.ndarray[DTYPE_int_t, ndim=4] ,
                                                    np.ndarray[DTYPE_int_t, ndim=4] ,
                                                    np.ndarray[DTYPE_int_t, ndim=4] ,
                                                    int,
                                                    int)

cpdef np.ndarray simplification(np.ndarray[DTYPE_int_t, ndim=4],int, int, int, int, int, int, int)

cpdef np.ndarray simplification_convex_polyhedron(np.ndarray[DTYPE_int_t, ndim=4], int, int, int)


cpdef np.ndarray object_subtraction(np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=4],
                                        int)

cpdef np.ndarray object_subtraction_2(np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=4],
                                        int)

cpdef np.ndarray object_subtraction_new(np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=4],
                                        int)

cpdef np.ndarray object_subtraction_dev(np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=4],
                                        int)

cpdef np.ndarray object_subtraction_dev1(np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=4],
                                        int)

cpdef np.ndarray object_subtraction_dev2(np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=4],
                                        int)

cpdef np.ndarray object_subtraction_3(np.ndarray[DTYPE_int_t, ndim=4],
                                        np.ndarray[DTYPE_int_t, ndim=4],
                                        int verbose)

