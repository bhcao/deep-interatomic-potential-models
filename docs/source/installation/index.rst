.. _installation:

Installation
=================

DIPM
----

The *dipm* library for training only can be installed via pip:

.. code-block:: bash

    pip install dipm

However, this command **only installs the regular CPU version** of JAX.
We recommend that the library is run on GPU.
Use this command instead to install the GPU-compatible version:

.. code-block:: bash

    pip install "dipm[cuda]"

**This command installs the CUDA 12 version of JAX.** For different versions, please
install *dipm* without the `cuda` flag and install the desired JAX version via pip.

Note that using the TPU version of JAX is, in principle, also supported by
this library. You need to install it separately via pip. However, it has not been
thoroughly tested and should therefore be considered an experimental feature.

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
