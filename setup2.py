from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext_modules = [
    Extension('pyqcstrc.icosah.modeling.math1', 
    sources=['pyqcstrc/icosah/modeling/math1.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.modeling.numericalc', 
    sources=['pyqcstrc/icosah/modeling/numericalc.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.modeling.symmetry', 
    sources=['pyqcstrc/icosah/modeling/symmetry.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.modeling.utils', 
    sources=['pyqcstrc/icosah/modeling/utils.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.modeling.intsct', 
    sources=['pyqcstrc/icosah/modeling/intsct.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.modeling.mics', 
    sources=['pyqcstrc/icosah/modeling/mics.pyx'],
    include_dirs=['.', get_include()]
    )
]

setup(name="pyqcstruc",
        ext_modules = cythonize(ext_modules,compiler_directives={'language_level':"3"})
)