# ⚛️ DIPM-Cvt: Dataset and Model File Conversion Tools for DIPM 🔄

This package contains tools for converting datasets and model files for DIPM.

For compatibility and efficiency reasons, we do not support LMDB and ExtXYZ datasets directly during training. LMDB (OCP style) depends on `torch_geometric` and indirectly on `torch`, leading to unnecessary dependencies that may cause compatibility problems. ExtXYZ is large and slow for loading, making it unsuitable as a dataset format. Therefore, we provide this script for dataset format conversion.

## 📦 Installation

To install the latest release of `dipm_cvt`, run the following command:

```
pip install dipm-cvt[opt_dep]
```

or for development:

```
git clone https://github.com/bhcao/deep-interatomic-potential-models.git
cd deep-interatomic-potential-models/dipm-conversion-tools
pip install -e .[opt_dep]
```

The `opt_dep` argument can be one or more of the following:
- `lmdb`: to enable support for LMDB datasets.
- `ase`: to enable support for ExtXYZ datasets.
- `web`: to enable support for web download.
- `gdrive`: to enable support for Google Drive download.
- `hf`: to enable support for huggingface hub datasets.
- `openqdc`: to enable support for OpenQDC datasets.
- `all`: to enable all optional dependencies above.

After installation, you can use the `dipm-cvt-cli` script, which should be in the usual place that pip places binaries (or you can explicitly run `python3 <path_to_cloned_dir>/dipm-conversion-tools/src/dipm_cvt/cli/main.py`).

## 📌 Usage

### 📊 Dataset Conversion

To convert a dataset, run the following command:

```
dipm-cvt-cli -d <input_path> <output_path>
```

Input path can any of the following:
- A single local dataset file.
- A local directory containing multiple dataset files.
- A compressed local dataset file (e.g. `*.tar.gz`, `*.zip`).
- A URL to a dataset file (e.g. `https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300.tar.gz`)
- A huggingface hub dataset identifier (e.g. `hf://colabfit/rMD17`).
- An OpenQDC dataset identifier (e.g. `openqdc://SpiceV2`).

Output path must be a directory except single file input.

Explanation of options:

| Option | Description |
| --- | --- |
| `--ref_energy_path` | Path to the reference energies file for OC20 dataset. See OC22 website for details. |
| `--split_file` | Whether to split every file in the dataset into multiple smaller files and save them in the subdirectory with the same name as the original file. Cannot be used with `--merge_dir`. |
| `--split_size` | Approximate size of each split file. The original file will be split into `total_size / split_size` files. Default is 512 MB. |
| `--merge_dir` | Whether to merge multiple dataset files into one, cannot be used with `--split_file`. |
| `--merge_size` | Approximate total size of the merged file. Default is to merge all files of every subdirectory into one file. |
| `--download_dir` | The original downloaded dataset file will be removed after conversion. If you want to keep it, specify `--download_dir` to the directory you want to save it. |
| `--energy_unit` | Energy unit of the dataset. Default is `eV`. |
| `--distance_unit` | Distance unit of the dataset. Default is `Angstrom`. |
| `--overwrite_existing` | Whether to overwrite existing files. Cannot be used with `--ignore_existing`. |
| `--ignore_existing` | Whether to ignore existing files. If your script is interrupted, you can use this option to skip existing files. |

### 🧠 Model Conversion

To convert a model checkpoint, run the following command:

```
dipm-cvt-cli -m <checkpoint_path> <safetensors_path>
```