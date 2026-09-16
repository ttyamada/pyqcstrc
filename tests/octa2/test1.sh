export PYQCSTRC=$HOME/git/pyqcstrc/src_python
#export PYTHONPATH=$PYQCSTRC:$PYTHONPATH
export OCTA2=$PYQCSTRC/octa2
export CRSYS=$PYQCSTRC/crsys
export OCCDM=$PYQCSTRC/occdom
export TWOODS=$PYQCSTRC/twoods
export QNNUM=$PYQCSTRC/qnnum
export QNVEC=$PYQCSTRC/qnvec
export QNMATH=$PYQCSTRC/qnmath
export QNMAT=$PYQCSTRC/qnmat
export QNNDARRAY=$PYQCSTRC/qnndarray
export GEOM=$PYQCSTRC/geometry
export NUMER=$PYQCSTRC/numeric
export MATH1=$PYQCSTRC/math1
export UTILS=$PYQCSTRC/utils
export QNSYM=$PYQCSTRC/qnsym
export INTSCT=$PYQCSTRC/intsct
export PRJOP=$PYQCSTRC/qnprj
export LATTICE=$PYQCSTRC/lattice
export SITESYM=$PYQCSTRC/sitesym
export OFF=$PYQCSTRC/off

export PYTHONPATH=$OCTA2:$CRSYS:$OCCDM:$TWOODS:$PNPRJ:$QNNUM:\
$QNVEC:$QNMATH:$QNMAT:$QNNDARRAY:$GEOM:$NUMER:$MATH1:$UTILS:$QNSYJ:\
$INTSCT:$PRJOP:$LATTICE:$SITESYM:$OFF\
$PYTHONPATH

# for profiling
#python -m cProfile -o test1.pstats test1.py

# for no-profileing
python test1.py
