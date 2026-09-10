export PYQCSTRC=$HOME/git/pyqcstrc
export OCTA2=$PYQCSTRC/src_python/octa2
export OCCDM=$PYQCSTRC/src_python/occdom
export TWOODS=$PYQCSTRC/src_python/twoods
export PYTHONPATH=$OCTA2:$OCCDM:$TWOODS:$PYTHONPATH
#export PYTHONPATH=$PYQCSTRC:$PYTHONPATH
# for profiling
python -m cProfile -o test2.pstats test2.py
