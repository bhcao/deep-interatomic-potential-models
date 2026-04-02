.. _user_guide:

User guide
==========

.. _getting_started:

Getting started
---------------

This is a enhanced version of `MLIP <https://github.com/instadeepai/mlip>`_ library.

The *dipm* library consists of multiple submodules targeted towards different
parts of a complete force field pipeline:

* ``data``: Code related to dataset reading and preprocessing. Its main purpose is
  to go from datasets stored on a file system to instances of
  :py:class:`GraphDataset <dipm.data.helpers.graph_dataset.GraphDataset>`
  that can be directly used for training or batched inference tasks.

* ``loss``: Code related to the loss functions used for training force models.

* ``layers``: Code related to the commonly used neural network layers for force models.

* ``models``: Code related to the force models, which can be wrapped as
  :py:class:`ForceFieldPredictor <dipm.models.ForceFieldPredictor>` objects for easy interfacing with
  other ``dipm`` submodules like ``training`` or ``simulation``.
  This module also contains the loss definition and utilities for
  parameter loading of trained models.

* ``training``: Code related to training force models. The main class for this task
  is the :py:class:`TrainingLoop <dipm.training.training_loop.TrainingLoop>`.

* ``simulation``: Code related to running MD simulations or energy minimizations with
  force models. We support the `JAX-MD <https://jax-md.readthedocs.io/>`_
  and `ASE <https://wiki.fysik.dtu.dk/ase/>`_ backends.

* ``utils``: Utility functions used in different modules.

**Each of these modules is designed to allow a user to set up their own experiment
scripts or notebooks with minimal effort, while also supporting customization,
especially for topics such as logging (e.g., to a remote storage location
like S3 or GCS) or adding new losses, force model architectures, or dataset
readers for customized data preprocessing.**

We provide some example Jupyter notebooks as tutorials
to help you with the onboarding process to the library.
Furthermore, we provide in-depth tutorials for each of the four main modules of the
library along with some other more advanced topics.

.. _notebook_tutorials:

Jupyter Notebook Tutorials
--------------------------

We provide Jupyter notebooks with example code that may serve as templates
to build your own more complex force pipelines. These can be used alongside the
deep-dive tutorials below to help you with getting onboarded to the *dipm* library.
These tutorials can be found in the GitHub repository:

* `Inference and simulation <https://github.com/bhcao/deep-interatomic-potential-models/blob/main/tutorials/simulation_tutorial.ipynb>`_
* `Model training <https://github.com/bhcao/deep-interatomic-potential-models/blob/main/tutorials/model_training_tutorial.ipynb>`_
* `Addition of new models <https://github.com/bhcao/deep-interatomic-potential-models/blob/main/tutorials/model_addition_tutorial.ipynb>`_

To run the tutorials, just install Jupyter notebooks via pip and launch it from
a directory that contains the notebooks:

.. code-block:: bash

    pip install notebook && jupyter notebook

The installation of *mlip* itself is included within the notebooks. We recommend to
run these notebooks with GPU acceleration enabled.

.. _tutorials:

Deep-dive Tutorials
-------------------

Follow the links below for more in-depth tutorials for each of the available
tasks supported by the library.

.. toctree::
   :maxdepth: 2

   dataset_preparation
   training
   simulations
   advanced/index
