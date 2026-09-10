export PYQCSTRC=$HOME/git/pyqcstrc
export CYTHON=$PYQCSTRC/src_cython
export DODE2=$PYQCSTRC/src_python/dode2
export PYTHONPATH=$PYQCSTRC/tests/dode2/tests:$DODE2:CYTHON:$PYTHONPATH
# for profiling
python -m cProfile -o test1.pstats test1.py
