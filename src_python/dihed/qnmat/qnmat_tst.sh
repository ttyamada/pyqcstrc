export PYQCSTRC=$HOME/git/pyqcstrc/src_python
export PYTHONPATH=$PYQCSTRC:$PYTHONPATH

# for profiling
#python -m cProfile -o test1.pstats test1.py

# for no-profileing
python qnmat_tst.py
