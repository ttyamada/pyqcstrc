#
# PyQCstrc - Python tools for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
cimport numpy as np

ctypedef np.float64_t DTYPE_double_t
ctypedef long DTYPE_int_t

cpdef np.ndarray strc(np.ndarray[DTYPE_int_t, ndim=4],
                        np.ndarray[DTYPE_int_t, ndim=3],
                        int,
                        int)

