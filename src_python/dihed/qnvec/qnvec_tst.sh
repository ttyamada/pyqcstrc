export PYQCSTRC=$HOME/git/pyqcstrc/src_python
export PYTHONPATH=$PYQCSTRC:$PYTHONPATH

#export OCTA2=$PYQCSTRC/octa2
#export OCCDM=$PYQCSTRC/occdom
#export TWOODS=$PYQCSTRC/twoods
#export PYTHONPATH=$OCTA2:$OCCDM:$TWOODS:$PYTHONPATH

# for profiling
#python -m cProfile -o test1.pstats test1.py

# for no-profileing
python qnvec_tst.py
