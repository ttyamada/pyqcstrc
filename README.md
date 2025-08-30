# PyQCstrc: python library for quasi-crystal structure

### Installation

via PIP:

```
pip install --upgrade pip
pip install pyqcstrc
```

or in local:
```
conda create -n pyqc python=3.10
conda activate pyqc
git clone -b develop https://github.com/ttyamada/pyqcstrc.git
cd pyqcstrc
python3 setup.py bdist_wheel
pip install ./dist/pyqcstrc-XXX.whl
pip install -e .
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

- numpy>=1.20.0
- scipy>=1.6.0

### License
PyQCstrc is released under a [MIT license](https://opensource.org/licenses/mit-license.php).
