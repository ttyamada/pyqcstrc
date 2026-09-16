export PYQCSTRC=$HOME/git/pyqcstrc/src_python
#export PYTHONPATH=$PYQCSTRC:$PYTHONPATH
export CRSYS=$PYQCSTRC/crsys
export QNNUM=$PYQCSTRC/qnnum
export QNVEC=$PYQCSTRC/qnvec
export PYTHONPATH=$CRSYS:$QNNUM:$QNVEC:$PYTHONPATH

# for profiling
#python -m cProfile -o test1.pstats test1.py

# for no-profileing
python test2.py