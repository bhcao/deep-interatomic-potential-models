.. _chemical_dataset:

.. module:: dipm.data.chemical_datasets.dataset

Dataset interface
=================

This interface defines a pytorch-like dataset interface for loading, combining and slicing
datasets.

.. autoclass:: Dataset

    .. automethod:: __getitem__

    .. automethod:: __len__

    .. automethod:: release

.. autoclass:: ConcatDataset

    .. automethod:: __init__

    .. automethod:: __getitem__

    .. automethod:: __len__

    .. automethod:: release

.. autoclass:: Subset

    .. automethod:: __init__

    .. automethod:: __getitem__

    .. automethod:: __len__

    .. automethod:: release
