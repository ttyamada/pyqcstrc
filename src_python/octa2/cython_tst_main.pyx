import cython_tst
import numpy
arr = numpy.arange(1000000000, dtype=numpy.int64)
cython_tst.do_calc(arr)
