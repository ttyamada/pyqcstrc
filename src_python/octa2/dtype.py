from numpy.typing import NDArray
import cython

DTYPE_int = cython.long
DARRAY_int = cython.typedef(NDArray[DTYPE_int])
#DARRAY_int = cython.typedef(cython.int)
