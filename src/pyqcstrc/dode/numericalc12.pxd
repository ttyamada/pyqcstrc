#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
cimport numpy as np

ctypedef np.float64_t DTYPE_double_t
ctypedef long DTYPE_int_t

cpdef list get_internal_component_numerical(np.ndarray[DTYPE_int_t, ndim=2])

cpdef list get_internal_component_phason_numerical(np.ndarray[DTYPE_int_t, ndim=2], np.ndarray[double, ndim=2])

cpdef int inside_outside_triangle(np.ndarray[DTYPE_int_t, ndim=2],np.ndarray[DTYPE_int_t, ndim=3],int)

cpdef int check_intersection_line_segment_triangle(np.ndarray[DTYPE_int_t, ndim=3],np.ndarray[DTYPE_int_t, ndim=3],int)

cpdef int check_intersection_two_triangles(np.ndarray[DTYPE_int_t, ndim=3],np.ndarray[DTYPE_int_t, ndim=3],int)

cpdef int check_intersection_two_segment_numerical(np.ndarray[DTYPE_int_t, ndim=3],np.ndarray[DTYPE_int_t, ndim=3],int)

cpdef int point_on_segment(np.ndarray[DTYPE_int_t, ndim=2],np.ndarray[DTYPE_int_t, ndim=2],np.ndarray[DTYPE_int_t, ndim=2])

cpdef int inout_occupation_domain_numerical(np.ndarray[DTYPE_int_t, ndim=4],np.ndarray[DTYPE_double_t, ndim=1],int)

cpdef int inout_occupation_domain_phason_numerical(np.ndarray[DTYPE_int_t, ndim=4],np.ndarray[DTYPE_double_t, ndim=1],np.ndarray[double, ndim=2],int)

cpdef list projection_numerical(np.ndarray[DTYPE_int_t,ndim=2])

cpdef list projection_1_numerical(double,double,double,double,double,double)

cpdef list projection_2_numerical(double,double,double,double,double,double,double,double,double,double)

#cpdef list projection_numerical_phason(np.ndarray[DTYPE_int_t, ndim=2],np.ndarray[DTYPE_int_t, ndim=3])

cpdef list projection_numerical_phason_1(np.ndarray[DTYPE_int_t, ndim=2],np.ndarray[double, ndim=2])
