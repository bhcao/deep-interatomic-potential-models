.. _installation:

Installation
=================

DIPM
----

The *dipm* library for CUDA 12 and training only can be installed via pip:

.. code-block:: bash

    pip install "jax[cuda12]"
    pip install dipm

If you just want to install the regular CPU version of JAX, please ignore the first line.
This line installs the necessary versions of `jaxlib <https://pypi.org/project/jaxlib/>`_.
See the `installation guide of JAX <https://docs.jax.dev/en/latest/installation.html>`_ for
more information.

Note that using the TPU version of *jaxlib* is, in principle, also supported by
this library. However, it has not been thoroughly tested and should therefore be
considered an experimental feature.

Simulation related tasks such as MD or energy minimization will require
`JAX-MD <https://github.com/jax-md/jax-md>`_ and `ASE <https://gitlab.com/ase/ase>`_
as dependencies. ASE can be installed as an optional dependency while the newest
version of JAX-MD must be installed directly from the GitHub repository to avoid
critical bugs. Here is the installation commands:

.. code-block:: bash

    pip install git+https://github.com/jax-md/jax-md.git
    pip install -U dipm[md]

DIPM-Cvt
--------

We provide a dataset / model file conversion tool called *dipm_cvt*, which is independent of dipm.

This package can be installed via pip with all optional dependencies:

.. code-block:: bash

    pip install dipm-cvt[all]

Typically, not all dependencies are needed. For example, if you don't want to convert model
file or some LMDB datasets, there is no need to install PyTorch. Therefore, we provide options
to install dependencies at your discretion.

The optional argument can be one or more of the following:

* `lmdb`: to enable support for LMDB datasets.
* `ase`: to enable support for ExtXYZ datasets.
* `web`: to enable support for web download.
* `gdrive`: to enable support for Google Drive download.
* `hf`: to enable support for huggingface hub datasets.
* `openqdc`: to enable support for OpenQDC datasets.
* `all`: to enable all optional dependencies above.
