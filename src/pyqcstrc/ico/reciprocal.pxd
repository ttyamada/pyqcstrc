#
# PyQCdiff - Python library for Quasi-Crystal diffraction
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
cimport numpy as np

ctypedef np.float64_t DTYPE_double_t
ctypedef long DTYPE_int_t

cpdef list generate_refection_list_ico(int,double,double,int)
cpdef list projection_qpar_ico_elser(int,int,int,int,int,int)
cpdef list projection_qperp_ico_elser(int,int,int,int,int,int)
cpdef list projection_qpar_ico_csg(int,int,int,int,int,int)
cpdef list projection_qperp_ico_csg(int,int,int,int,int,int)



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
