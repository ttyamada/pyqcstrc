# for profiling
#cythonize -i -3 -X profile=True,infer_types=True,boundscheck=False,wraparound=False $1

# for no-profiling
#cythonize -i -3 -X infer_types=True,boundscheck=False,wraparound=False $1

# for creating html file use -a as a cython compiler option
cython -a -i -3 -X infer_types=True,boundscheck=False,wraparound=False $1

# for viewing html file use firefox or any other browser


