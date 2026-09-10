export PYQCSTRC=$HOME/git/pyqcstrc/src_python
export CRSYS=$PYQCSTRC/crsys
export QNNUM=$PYQCSTRC/qnnum
export QNVEC=$PYQCSTRC/qnvec
export QNVEC=$PYQCSTRC/qnmat
export QNVEC=$PYQCSTRC/qnprj
export PYTHONPATH=$CRSYS:$QNNUM:$QNVEC:$QNMATH:$QNPRJ:$PYTHONPATH
#export PYTHONPATH=$PYQCSTRC:$PYTHONPATH
# for profiling
#python -m cProfile -o test1.pstats test1.py

# for no-profileing
python test3.py