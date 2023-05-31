Installation
============
*PyQCstrc* supports python version ``3.7`` and above.

PyQCstrc is available on the `PyPI <https://pypi.python.org/pypi/pyqcstrc>`_ as binary wheels packages provided for common platforms, macOS, linux and Windows.

It is recommended to instal PyQCstrc into a virtual environment we will call myvenv.

.. code-block:: bash

   python3 -m venv ~/myvenv
   source ~/myvenv/bin/activate

The binary wheel packages of PyQCstrc can be then installed using PIP. PIP is the package installer for Python, and we recommend to upgrade pip and its related utilities before installing PyQCstrc.

.. code-block:: bash

   pip install setuptools wheel pip --upgrade
   pip install pyqcstrc

Installation has been tested on macOS (10.14), Linux (Ubuntu20.04, Fedora32 and CentOS8) and Windows 10.

Dependencies
------------
Requirements:

* Python packages (all installable using pip):

 * numpy (version>=1.20.0)
 * scipy (version>=1.6.0)
 * cython (version>=0.29.21)


Installation under Windows10
----------------------------
Python is not installed by default under Windows 10. We suggest you install `Python3 <http://python.org>`_ from the official web page.

The virtual environment can be created using the `venv module <https://docs.python.org/3/library/venv.html>`_, which is done by executing the command:

.. code-block:: bash

    python3 -m virtualenv myvenv

Once the virtual enviromen is created, activate it by executing a ``activate.bat`` script file:

.. code-block::

    myvenv\Scripts\activate.bat
    
Then, the PyQCstrc can be installed as described above.
