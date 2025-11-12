.. _hdf5_dataset:

.. module:: dipm.data.chemical_datasets.hdf5_dataset

HDF5 Dataset
============

This class reads data from a `MACE <https://github.com/ACEsuit/mace>`_ compatible
`HDF5 format <https://docs.h5py.org/en/>`_ file organized in the following way. The data
must be grouped into batches containing the same number of data points, with each batch
name being `config_batch_{idx}`. It's fine to put all data into a single batch group.
Every data point must defined as a group named `config_{idx}`. Below, we provide
an example of how to read the data from such a compliant HDF5 file to demonstrate
how the data is organized:

.. code-block:: python

    def unpack_value(value):
        '''If the value is a string of "None", return None, otherwise return the value.'''
        if isinstance(value, bytes) and value == b"None":
            return None
        return value
    
    def get_value(group, name):
        '''If the attribute exists, unpack and return its value, otherwise return None.'''
        if name in group:
            return unpack_value(group[name][()])
        return None

    with h5py.File(hdf5_dataset_path, "r") as h5file:
        # Deciding the batch index and data point index to load
        batch_index = 0
        data_index = 0

        # Get the group containing the data point
        batch_group = h5file[f"config_batch_{batch_index}"]
        data_group = batch_group[f"config_{data_index}"]

        # Attributes that must exist
        positions = data_group["positions"][()]
        atomic_numbers = data_group["atomic_numbers"][()]

        # Attributes that can be None
        pbc = unpack_value(data_group["pbc"][()])
        cell = unpack_value(data_group["cell"][()])

        # Attributes contained in the "properties" group
        forces = get_value(data_group["properties"], "forces")
        energy = get_value(data_group["properties"], "energy")
        stress = get_value(data_group["properties"], "stress")

See below for the API reference to the associated loader class.

.. autoclass:: Hdf5Dataset

    .. automethod:: __init__

    .. automethod:: __getitem__

    .. automethod:: __len__

    .. automethod:: release

.. autofunction:: create_datasets
