# PyQCstrc: python library for quasi-crystal structure

### Installation

via PIP:

```
pip install --upgrade pip
pip install pyqcstrc
```

or in local:
```
conda create -n pyqc python=3.10 pip
conda activate pyqc
gh repo clone ttyamada/pyqcstrc
cd pyqcstrc
python3 setup.py bdist_wheel
pip3 install ./dist/pyqcstrc-XXX.whl
```

Fro more information, see [Docs](https://www.rs.tus.ac.jp/tsunetomo.yamada/pyqcstrc/).

### Supported versions

- Python>=3.7
- Operating systems: Linux, macOS, and Windows

### Publications

If you use PyQCstrc in your research please cite the corresponding [paper](https://doi.org/10.1107/S1600576721005951):
```BibTeX
@article{yamada2021pyqcstrc,
  title={PyQCstrc. ico: a computing package for structural modelling of icosahedral quasicrystals},
  author={Yamada, Tsunetomo},
  journal={Journal of Applied Crystallography},
  volume={54},
  number={4},
  pages={1252--1255},
  year={2021},
  publisher={International Union of Crystallography}
}
```

### Documentation

Documentation can be found on [Docs](https://www.rs.tus.ac.jp/tsunetomo.yamada/pyqcstrc/index.html).


### Requirements

- Cython>=0.29.21
- numpy>=1.20.0
- scipy>=1.6.0

### License
PyQCstrc is released under a [MIT license](https://opensource.org/licenses/mit-license.php).
