Installation
============

.. highlight:: console

Requirements
------------
CorrHOD is designed to work on python 3.6 and above. It requires the following packages:

* `numpy <http://www.numpy.org/>`_
* `pandas <https://pandas.pydata.org/>`_
* `healpy <https://healpy.readthedocs.io/en/latest/>`_
* `densitysplit <https://github.com/epaillas/densitysplit>`_ (OpenMP branch)
* AbacusHOD (from `abacusutils <https://abacusutils.readthedocs.io/en/latest/>`_)
* `cosmoprimo <https://cosmoprimo.readthedocs.io/en/latest/>`_
* `pycorr <https://py2pcf.readthedocs.io/en/latest/>`_
* `mockfactory <https://github.com/cosmodesi/mockfactory>`_

.. warning::
    The pycorr module required here is not the same as the one in the pip repository.
    If installed manually, it must be installed with ::

        $ pip install pycorr[mpi,jackknife,corrfunc] @ git+https://github.com/cosmodesi/pycorr

    (see `pycorr documentation <https://py2pcf.readthedocs.io/en/latest/>`_ for more details)

.. note::
    The CorrHOD code is designed to wort either with the main and the OpenMP branch of densitysplit. 
    However, the OpenMP branch is highly recommended as it is much faster, and compatible with MPI.


Pip Installation
----------------
For access to the Python functionality of CorrHOD, you can either install via pip
or clone from GitHub. The pip installation is recommended if you don't need to modify
the source:
::

    $ pip install git+https://github.com/SBouchard01/CorrHOD

This will install most dependencies

.. warning::
    Some packages might need to be installed manually beforehand, as they require C compilation.
    Some examples are ``Corrfunc``, or ``mpi4py``, as they are included in the ``pycorr`` package.
    ``mockfactory`` or ``cosmoprimo`` also have some C-dependencies that might need to be installed manually.
    If a ``ModuleNotFoundError`` is raised during the installation, please install the missing package manually and try again.


For Developers
--------------

Installing from Cloned Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to hack on the CorrHOD source code, it is best to clone the GitHub 
repository install the package in pip "editable mode":

::

    $ git clone https://github.com/SBouchard01/CorrHOD.git
    $ cd CorrHOD
    $ python setup.py develop --user

.. note::
    You can also git clone the package without developer mode ::

        $ git clone https://github.com/SBouchard01/CorrHOD.git
        $ cd CorrHOD
        $ python setup.py install --user


Building documentation
~~~~~~~~~~~~~~~~~~~~~~
The documentation is built using Sphinx. 
The ``setup.py`` file is designed to ignore and mock the packages that have C dependencies if the environment name is ``READTHEDOCS``.
This *should* allow the documentation to be built on ReadTheDocs without any problem.