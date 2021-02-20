from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

extensions = [
    Extension('pyqcstrc.icosah.math1', 
    sources=['pyqcstrc/icosah/math1.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.numericalc', 
    sources=['pyqcstrc/icosah/numericalc.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.symmetry', 
    sources=['pyqcstrc/icosah/symmetry.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.utils', 
    sources=['pyqcstrc/icosah/utils.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.intsct', 
    sources=['pyqcstrc/icosah/intsct.pyx'],
    include_dirs=['.', get_include()]
    ),
    Extension('pyqcstrc.icosah.mics', 
    sources=['pyqcstrc/icosah/mics.pyx'],
    include_dirs=['.', get_include()]
    )
]

setup(
    name="pyqcstruc",
    ext_modules = cythonize(extensions,compiler_directives={'language_level':"3"}),
    scripts = ['pyqcstrc/icosah/occupation_domain.py',
                'pyqcstrc/icosah/two_occupation_domains.py'],
    include_package_data=True,
    )