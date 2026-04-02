.. _data_processing:

Data processing
===============

Set up graph dataset builder
----------------------------

In order to train a model or run batched inference, one needs to process the data
into objects of type
:py:class:`GraphDataset <dipm.data.helpers.graph_dataset.GraphDataset>`.
This can be achieved by using the
:py:class:`GraphDatasetBuilder <dipm.data.graph_dataset_builder.GraphDatasetBuilder>`
class, which can be instantiated from its associated pydantic config and a
tuple of datasets that is derived from the
:py:class:`Dataset <dipm.data.chemical_datasets.dataset.Dataset>`
base class:

.. code-block:: python

    from dipm.data import GraphDatasetBuilder

    # datasets is a tuple of train, validation and test datasets
    datasets = _get_datasets()  # this is a placeholder for the moment
    builder_config = GraphDatasetBuilder.Config(
        graph_cutoff_angstrom=5.0,
        max_n_node=None,
        max_n_edge=None,
        batch_size=16,
    )
    graph_dataset_builder = GraphDatasetBuilder(datasets, builder_config)

In the example above, we set some example values for the settings in the
:py:class:`GraphDatasetBuilderConfig <dipm.data.configs.GraphDatasetBuilderConfig>`.
For simpler code, we allow to access this config object directly via
``GraphDatasetBuilder.Config``. Check out the API reference of the class to see the
full set of configurable values and for which values we have defaults available.

The datasets is a tuple of instances of
:py:class:`Dataset <dipm.data.chemical_datasets.dataset.Dataset>`
class.
This class allows to read a dataset into lists of
:py:class:`ChemicalSystem <dipm.data.chemical_system.ChemicalSystem>` objects via
its ``__getitem__`` method. You can either implement your own derived class to do
this for your custom dataset format, or you can use built-in
:py:func:`create_datasets <dipm.data.chemical_datasets.hdf5_dataset.create_datasets>`.

.. code-block:: python

    from dipm.data import create_datasets, ChemicalDatasetsConfig

    datasets_config = ChemicalDatasetsConfig(
        train_dataset_paths = "...",
        valid_dataset_paths = "...",
        test_dataset_paths = "...",
    )

    # If data is stored locally
    datasets = create_datasets(datasets_config)

If you have multiple datasets in different formats and would like to combine them,
or you want to slice a dataset without loading all of it, you can do so by instead
using the
:py:class:`ConcatDataset <dipm.data.chemical_datasets.dataset.ConcatDataset>` and
:py:class:`Subset <dipm.data.chemical_datasets.dataset.Subset>` classes.

.. code-block:: python

    from dipm.data import ConcatDataset, Subset

    datasets = _get_list_of_individual_chemical_datasets()  # placeholder
    combined_dataset = ConcatDataset(datasets)

    start, length = 0, 1000  # example slice
    subset_dataset = Subset(combined_dataset, start, length)

This resulting dataset can then also be used as an input to the
:py:class:`GraphDatasetBuilder <dipm.data.graph_dataset_builder.GraphDatasetBuilder>`.

Built-in graph dataset: data formats
--------------------------------------------

We only provide one built-in core dataset:
:py:class:`Hdf5Dataset <dipm.data.chemical_datasets.hdf5_dataset.Hdf5Dataset>`.

To train an force model, we need a dataset of atomic systems
with the following features per system with specific units:

* the positions (i.e., coordinates) of the atoms in the structure in Angstrom
* the element numbers of the atoms
* the forces of the atoms in eV / Angstrom
* the energy of the structure in eV
* (optional) the stress of the structure  in eV / Angstrom\ :sup:`3`
* (optional) the periodic boundary conditions

For a detailed description of the data format that the
:py:class:`Hdf5Dataset <dipm.data.chemical_datasets.hdf5_dataset.Hdf5Dataset>`.
requires, see :ref:`here <hdf5_dataset>`.

If you want to use different data formats or units, it is recommended to use our
:ref:`dataset conversion tool <dataset_preparation>` to convert your data into
the required format. You can also implement your own derived class to read your
custom dataset format.

Start preprocessing
-------------------

Once you have the ``graph_dataset_builder`` set up, you can start the preprocessing and
fetch the resulting datasets:

.. code-block:: python

    graph_dataset_builder.prepare_datasets()

    splits = graph_dataset_builder.get_splits()
    train_set, validation_set, test_set = splits

The resulting datasets are of type
:py:class:`GraphDataset <dipm.data.helpers.graph_dataset.GraphDataset>`
as mentioned above. For example, to process the batches in the training set, one
can execute:

.. code-block:: python

    num_graphs = len(train_set.graphs)
    num_batches = len(train_set)

    for batch in train_set:
        _process_batch_in_some_way(batch)

Get sharded batches
-------------------

If one wants to generate batches that are sharded across devices and prefetched, the
arguments to the ``get_splits()`` member of the
:py:class:`GraphDatasetBuilder <dipm.data.graph_dataset_builder.GraphDatasetBuilder>`
must be set to the following:

.. code-block:: python

    splits = graph_dataset_builder.get_datasets(
        prefetch=True, devices=jax.local_devices()
    )
    train_set, valid_set, test_set = splits

Now, the datasets are not of type
:py:class:`GraphDataset <dipm.data.helpers.graph_dataset.GraphDataset>` anymore,
but of type
:py:class:`PrefetchIterator <dipm.data.helpers.data_prefetching.PrefetchIterator>`
instead which implements batch prefetching on top of the
:py:class:`ParallelGraphDataset <dipm.data.helpers.data_prefetching.ParallelGraphDataset>`
class. It can be iterated over to obtain the sharded batches in the same way, however,
note that it does not have a ``graphs`` member that can be accessed directly.

.. _get_dataset_info:

Get dataset info
----------------

Furthermore, the builder class also populates a dataclass of type
:py:class:`DatasetInfo <dipm.data.dataset_info.DatasetInfo>`, which contains
metadata about the dataset which are relevant to the models while training and must be
stored together with the models for these to be usable. The populated instance of this
dataclass can be accessed easily like this:

.. code-block:: python

    # Note: this will call prepare_datasets() and give a warning if accessed
    # before prepare_datasets() is run
    dataset_info = graph_dataset_builder.dataset_info
