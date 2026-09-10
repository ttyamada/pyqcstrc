export PYQCSTRC=$HOME/git/pyqcstrc/src_python
export CRSYS=$PYQCSTRC/crsys
export QNNUM=$PYQCSTRC/qnnum
export PYTHONPATH=$CRSYS:$QNNUM:$PYTHONPATH
#export PYTHONPATH=$PYQCSTRC:$PYTHONPATH
# for profiling
#python -m cProfile -o test1.pstats test1.py

# for no-profileing
python test1.py