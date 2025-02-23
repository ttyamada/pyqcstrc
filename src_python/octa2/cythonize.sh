#cythonize -i -3 -X annotate=True,infer_types=True,boundscheck=False,wraparound=True,profile=True $1
cythonize -i -3 -X infer_types=True,boundscheck=False,wraparound=True,profile=True $1

#for profiling use
#profile=False
# but this is default


