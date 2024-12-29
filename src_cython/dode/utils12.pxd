#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
cimport numpy as np

ctypedef np.float64_t DTYPE_double_t
ctypedef long DTYPE_int_t

cpdef np.ndarray generator_obj_outline(np.ndarray[DTYPE_int_t, ndim=4], int)

cpdef list obj_area_6d(np.ndarray[DTYPE_int_t, ndim=4])

cpdef list triangle_area_6d(np.ndarray[DTYPE_int_t, ndim=3])

cpdef np.ndarray shift_object(np.ndarray[DTYPE_int_t, ndim=4],
                            np.ndarray[DTYPE_int_t, ndim=2],
                            int)

cpdef np.ndarray remove_doubling_dim4_in_perp_space(np.ndarray[DTYPE_int_t, ndim=4])

cpdef np.ndarray remove_doubling_dim3_in_perp_space(np.ndarray[DTYPE_int_t, ndim=3])

cpdef np.ndarray surface_cleaner(np.ndarray[DTYPE_int_t, ndim=4], int, int)

cpdef np.ndarray remove_doubling_dim4(np.ndarray[DTYPE_int_t, ndim=4])

cpdef np.ndarray remove_doubling_dim3(np.ndarray[DTYPE_int_t, ndim=3])
