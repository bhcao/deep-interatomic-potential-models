.. _dataset_preparation:

Dataset preparation
===================

We only support HDF5 format datasets for training (compatible with HDF5 used in
`MACE <https://github.com/ACEsuit/mace>`_, see :ref:`hdf5_dataset` for details).
You should either use datasets from MACE or convert your own dataset to this format.

We provided a dataset conversion toolkit for this purpose. We recommend to install it in
a different environment than *dipm* to avoid conflicts. We provided a command-line
interface `dipm-cvt-cli` for user-friendly usage.

To convert a dataset, run the following command (requires `dipm_cvt` to be installed):

.. code-block:: bash

    dipm-cvt-cli -d <input_path> <output_path>

Options
-------

Input path can any of the following:

* A single local dataset file.
* A local directory containing multiple dataset files.
* A compressed local dataset file (e.g. `*.tar.gz`, `*.zip`).
* A URL to a dataset file (e.g. `https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300.tar.gz`)
* A huggingface hub dataset identifier (e.g. `hf://colabfit/rMD17`).
* An OpenQDC dataset identifier (e.g. `openqdc://SpiceV2`).

Output path must be a directory except single file input.

Supported dataset formats:

* LMDB (PyG format in extension `.lmdb`, see `OC20 and OC22 <https://fair-chem.github.io/catalysts/datasets/oc20.html>`_)
* LMDB (Compressed JSON format in extension `.lmdbase`, see `OMOL25 <https://fair-chem.github.io/molecules/datasets/omol25.html>`_)
* ExtXYZ (Extended XYZ in extension `.xyz` or `.extxyz`, see `ASE extxyz format <https://ase-lib.org/ase/io/formatoptions.html#extxyz>`_)

Available options:

+--------------------------+---------------------------------------------------------------------+
| Options                  | Description                                                         |
+==========================+=====================================================================+
|`--ref_energy_path <path>`| Path to the reference energies file for OC20 dataset. See `OC22     |
|                          | website <https://fair-chem.github.io/catalysts/datasets/oc22.html>`_|
|                          | for details.                                                        |
+--------------------------+---------------------------------------------------------------------+
| `--split`                | Whether to split every file in the dataset into multiple smaller    |
|                          | files and save them in the subdirectory with the same name as the   |
|                          | original file. Cannot be used with `--merge`.                       |
+--------------------------+---------------------------------------------------------------------+
| `--merge`                | Whether to merge multiple dataset files into one, cannot be used    |
|                          | with `--split`.                                                     |
+--------------------------+---------------------------------------------------------------------+
| `--size <size>`          | Approximate size (in MB) of each resulting file. When `--split` is  |
|                          | specified, the original file will be split into `total_size / size` |
|                          | files and default is 512 MB. When `--merge` is specified, default   |
|                          | is to merge all files of every subdirectory into one file.          |
+--------------------------+---------------------------------------------------------------------+
| `--download_dir <dir>`   | The original downloaded dataset file will be removed after          |
|                          | conversion. If you want to keep it, specify `--download_dir` to the |
|                          | directory you want to save it.                                      |
+--------------------------+---------------------------------------------------------------------+
| `--energy_unit <unit>`   | Energy unit of the original dataset. Will be converted to `eV`.     |
|                          | Options are `eV` (default), `kJ/mol`, `kcal/mol`, `Hartree`.        |
+--------------------------+---------------------------------------------------------------------+
| `--distance_unit <unit>` | Distance unit of the original dataset. Will be converted to         |
|                          | `Angstrom`. Options are `Angstrom` (default), `Bohr`.               |
+--------------------------+---------------------------------------------------------------------+

Examples
--------

To convert local datasets, split into 512 MB files and convert energy unit from kcal/mol to eV:

.. code-block:: bash

    dipm-cvt-cli -d /home/user/datasets /home/user/converted_datasets --split_file --split_size 512 --energy_unit kcal/mol

To use a OpenQDC dataset and cache the downloaded OpenQDC files:

.. code-block:: bash

    dipm-cvt-cli -d openqdc://SpiceV2 /home/user/converted_datasets --download_dir /home/user/download_dir

To use a url to a compressed dataset file:

.. code-block:: bash

    dipm-cvt-cli -d https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300.tar.gz /home/user/converted_datasets
