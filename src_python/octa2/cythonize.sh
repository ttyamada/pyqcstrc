cythonize -i -3 -X infer_types=True,boundscheck=False,wraparound=False,profile=True $1
#cythonize -i -3 -X infer_types=True,boundscheck=False,wraparound=False $1

#for no-profiling use
#profile=False
# but this is default


