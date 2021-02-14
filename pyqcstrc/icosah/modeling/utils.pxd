cimport numpy as np

ctypedef np.float64_t DTYPE_double_t
ctypedef long DTYPE_int_t

cpdef list obj_volume_6d(np.ndarray[DTYPE_int_t, ndim=4])

cpdef list tetrahedron_volume_6d(np.ndarray[DTYPE_int_t, ndim=3])

cpdef np.ndarray shift_object(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                np.ndarray[DTYPE_int_t, ndim=2], 
                                int)

cpdef np.ndarray remove_doubling_dim4(np.ndarray[DTYPE_int_t, ndim=4])

cpdef np.ndarray remove_doubling_dim3(np.ndarray[DTYPE_int_t, ndim=3])

cpdef np.ndarray remove_doubling_dim4_in_perp_space(np.ndarray[DTYPE_int_t, ndim=4])

cpdef np.ndarray remove_doubling_dim3_in_perp_space(np.ndarray[DTYPE_int_t, ndim=3])

cpdef int generator_xyz_dim4(np.ndarray[DTYPE_int_t, ndim=4],
                            char)

cpdef int generator_xyz_dim4_triangle(np.ndarray[DTYPE_int_t, ndim=4],
                                        char)

cpdef int generator_xyz_dim4_tetrahedron(np.ndarray[DTYPE_int_t, ndim=4],
                                        char,
                                        int)

cpdef int generator_xyz_dim4_tetrahedron_select(np.ndarray[DTYPE_int_t, ndim=4],
                                                char,
                                                int)

cpdef int generator_xyz_dim3(np.ndarray[DTYPE_int_t, ndim=3],
                            char)
