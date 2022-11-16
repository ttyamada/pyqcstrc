#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
cimport numpy as np

ctypedef np.float64_t DTYPE_double_t
ctypedef long DTYPE_int_t

#cpdef list symop_vec(np.ndarray[DTYPE_int_t,ndim=2],
#                    np.ndarray[DTYPE_int_t,ndim=2],
#                    np.ndarray[DTYPE_int_t, ndim=2])

#cpdef list symop_obj(np.ndarray[DTYPE_int_t,ndim=2],
#                    np.ndarray[DTYPE_int_t, ndim=3],
#                    np.ndarray[DTYPE_int_t, ndim=2])

cpdef np.ndarray generator_obj_symmetric_vec(np.ndarray[DTYPE_int_t, ndim=3],
                                            np.ndarray[DTYPE_int_t, ndim=2])
                                            
cpdef np.ndarray generator_obj_symmetric_surface(np.ndarray[DTYPE_int_t, ndim=3],
                                                np.ndarray[DTYPE_int_t, ndim=2])

cpdef np.ndarray generator_obj_symmetric_tetrahedron(np.ndarray[DTYPE_int_t, ndim=3],
                                                    np.ndarray[DTYPE_int_t, ndim=2])

cpdef np.ndarray generator_obj_symmetric_tetrahedron_specific_symop(np.ndarray[DTYPE_int_t, ndim=3],
                                                                    np.ndarray[DTYPE_int_t, ndim=2],
                                                                    list)

cpdef np.ndarray generator_obj_symmetric_tetrahedron_0(np.ndarray[DTYPE_int_t, ndim=3],
                                                        np.ndarray[DTYPE_int_t, ndim=2],
                                                        int numop)

cpdef list symop_vec(np.ndarray[DTYPE_int_t,ndim=2],
                    np.ndarray[DTYPE_int_t,ndim=2],
                    np.ndarray[DTYPE_int_t, ndim=2])

cpdef list symop_obj(np.ndarray[DTYPE_int_t,ndim=2],
                    np.ndarray[DTYPE_int_t, ndim=3],
                    np.ndarray[DTYPE_int_t, ndim=2])

cpdef np.ndarray generator_equivalent_vec(np.ndarray[DTYPE_int_t, ndim=3],
                                            np.ndarray[DTYPE_int_t, ndim=2])
