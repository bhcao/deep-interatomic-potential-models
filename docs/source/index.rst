.. MLIP documentation master file:
   You can adapt this file completely to your liking,
   but it should at least
   contain the root `toctree` directive.

Welcome to the documentation of DIPM!
=====================================

*dipm* is a enhancement of `MLIP <https://github.com/instadeepai/mlip>`_ standing
for **Deep Interatomic Potentials Models (DIPM)**. It contains the following features:

* Multiple NNX model architectures (for now: MACE, NequIP, ViSNet, LiTEN, EquiformerV2 and UMA)
* Dataset loading and preprocessing
* Training and fine-tuning force models
* Batched inference with trained force models
* MD simulations and energy minimizations with JAX-MD and ASE backend.

As a first step, we recommend that you check out our page on :ref:`installation`
and our :ref:`user_guide` which contains several tutorials on how to use the library.
Furthermore, we also provide an :ref:`api_reference`.

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation <installation/index>
   User guide <user_guide/index>
   API reference <api_reference/index>
