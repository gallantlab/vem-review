# Voxelwise Encoding Model Review

This repository contains the code and analysis scripts to reproduce the figures for the Voxelwise Encoding Model (VEM) review paper.

## Overview

This codebase implements neural response modeling using fMRI data with a focus on voxelwise encoding models. The project demonstrates how to fit banded ridge regression models using motion energy and WordNet semantic features to predict neural responses across different brain regions.

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended for model fitting)

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd vem-review
```

2. Install the package in development mode:
```bash
cd src
pip install -e .
```

### Dependencies

The project requires several scientific computing and neuroimaging libraries:

**Core dependencies:**
- numpy, scipy, scikit-learn
- matplotlib (visualization)
- h5py (data storage)
- torch (GPU acceleration)

**Neuroimaging libraries:**
- pycortex (cortical surface visualization)
- himalaya (ridge regression models)
- pymoten (motion energy features)
- voxelwise_tutorials (utilities)

**Data management:**
- datalad (automatic data downloading)

All dependencies are automatically installed via `pip install -e .`

## Data

The project uses experimental data from the Gallant Lab's short clips dataset, which includes:

- **fMRI responses**: Neural responses for 5 subjects (S01-S05) recorded during natural movie viewing
- **Motion energy features**: Spatiotemporal filters capturing visual motion patterns
- **WordNet semantic features**: Semantic category representations
- **Cortical mappers**: Surface topology data for brain visualization

Data is automatically downloaded from https://gin.g-node.org/gallantlab/shortclips when you run the analysis scripts.

## Usage

### Basic Workflow

1. **Fit encoding models**:
   ```bash
   python scripts/01_fit-banded-ridge.py S01
   ```
   This script fits banded ridge regression models using motion energy and WordNet features for subject S01. Replace `S01` with other subjects (`S02`, `S03`, `S04`, `S05`) as needed.

2. **Generate visualizations**:
   ```bash
   python scripts/02_plot-banded-ridge.py S01
   ```
   Creates cortical surface visualizations and analysis plots for the fitted models.

### Jupyter Notebooks

The `notebooks/` directory contains example analyses:

- `example-fir-model.ipynb`: Demonstrates finite impulse response (FIR) model fitting
- `simulate-noise-ceiling.ipynb`: Simulates noise ceiling analysis for model evaluation
- `utils.py`: Utility functions for notebook analyses

## Model Architecture

The encoding models use **banded ridge regression** with multiple kernel types:

1. **Motion energy features**: Spatiotemporal Gabor filters capturing local motion patterns
2. **WordNet semantic features**: Hierarchical semantic category representations
3. **Delayed features**: Temporal delays to capture hemodynamic response

### Key Processing Steps

1. **Data preprocessing**:
   - Z-score fMRI responses within runs
   - Average test responses across repetitions
   - Demean features within runs

2. **Cross-validation**:
   - Leave-one-run-out validation respecting temporal structure
   - Hyperparameter optimization using grid search

3. **Evaluation**:
   - Normalized correlation (CCnorm) between predicted and actual responses
   - R² scores for model performance assessment

## Directory Structure

```
vem-review/
├── README.md                    # This file
├── LICENSE.md                   # License information
├── CLAUDE.md                    # Development instructions
├── src/
│   ├── setup.py                 # Package installation
│   └── vemreview/
│       ├── __init__.py
│       ├── config.py            # Configuration and paths
│       ├── io.py                # Data loading functions
│       └── utils.py             # Visualization utilities
├── scripts/
│   ├── 01_fit-banded-ridge.py  # Main model fitting script
│   └── 02_plot-banded-ridge.py # Visualization generation
├── notebooks/
│   ├── example-fir-model.ipynb
│   ├── simulate-noise-ceiling.ipynb
│   └── utils.py
└── figures/                     # Generated visualizations (created by scripts)
```

## Output

Running the analysis scripts will generate:

- **tutorial-data/**: Downloaded experimental data and fitted model results
- **figures/**: Cortical surface visualizations and analysis plots
- **Model weights**: Saved in HDF5 format for further analysis

## Citation

If you use this code in your research, please cite:

```bibtex
@article{vem-review,
  title={Voxelwise Encoding Model Review},
  author={[Authors]},
  journal={[Journal]},
  year={[Year]}
}
```

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE.md file for details.

## Support

For questions or issues, please open an issue on the GitHub repository.

## Acknowledgments

This work builds upon the excellent neuroimaging analysis tools:
- [Himalaya](https://github.com/gallantlab/himalaya) for ridge regression
- [Pycortex](https://github.com/gallantlab/pycortex) for cortical visualization
- [Voxelwise Tutorials](https://github.com/gallantlab/voxelwise_tutorials) for analysis utilities