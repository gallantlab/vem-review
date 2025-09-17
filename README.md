# Encoding models in functional magnetic resonance imaging: the Voxelwise Encoding Model framework

This repository contains the code and analysis scripts to reproduce the figures from the guide on the Voxelwise Encoding Model (VEM) framework:

> Visconti di Oleggio Castello\*, M., Deniz\*, F., Dupré la Tour, T., & Gallant, J. L. (2025). Encoding models in functional magnetic resonance imaging: the Voxelwise Encoding Model framework. *PsyArXiv*. https://doi.org/10.31234/osf.io/nt2jq_v1
> 
> *equal contribution


## Overview

This codebase implements a basic VEM analaysis on public data. The project demonstrates how to fit banded ridge regression models using motion energy and WordNet semantic features to predict neural responses across different brain regions. It is heavily based on the Voxelwise Encoding Model tutorials available [here](https://gallantlab.org/voxelwise_tutorials/).

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for model fitting; fitting on CPU is possible but it will take a very long time!)
- [uv](https://docs.astral.sh/uv/) - Modern Python package manager

### Setup

1. Clone this repository:
```bash
git clone https://github.com/gallantlab/vem-review.git
cd vem-review
```

2. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install the package and dependencies:
```bash
uv sync
```

This will automatically create a virtual environment and install all required dependencies, including Jupyter for running notebooks.

### Dependencies

The project requires several scientific computing and neuroimaging libraries:

**Core dependencies:**
- numpy, scipy, scikit-learn
- matplotlib (visualization)
- h5py (data storage)
- torch (GPU acceleration)

**Neuroimaging libraries:**
- pycortex (cortical surface visualization)
- himalaya (efficient ridge regression models)
- pymoten (motion energy features)
- voxelwise_tutorials (utilities)

**Data management:**
- datalad (automatic data downloading)

All dependencies are automatically installed via `uv sync` and defined in `pyproject.toml`

## Data

The project uses experimental data from the Gallant Lab's short clips dataset.
The necessary data are automatically downloaded from https://gin.g-node.org/gallantlab/shortclips when you run the analysis scripts.
The first time you run the script it will need to download approximately 5GB of data. 

## Usage

### Basic Workflow

1. **Fit encoding models**:
   ```bash
   uv run python scripts/01_fit-banded-ridge.py S01
   ```
   This script fits banded ridge regression models with motion energy and WordNet features for subject S01. Replace `S01` with other subjects (`S02`, `S03`, `S04`, `S05`) as needed.

2. **Generate visualizations**:
   ```bash
   uv run python scripts/02_plot-banded-ridge.py S01
   ```
   Creates cortical surface visualizations and analysis plots for the fitted models. These are the plots shown in the figures of the paper.

### Jupyter Notebooks

The `notebooks/` directory contains simulations that are used in Boxes 3 and 4.

- `example-fir-model.ipynb`: Show how finite impulse response (FIR) models work
- `simulate-noise-ceiling.ipynb`: Explains how the normalized correlation coefficient can account for different levels of noise
- `utils.py`: Utility functions for notebook analyses

To run notebooks with the proper environment:
```bash
uv run jupyter notebook notebooks/
```

## Output

Running the analysis scripts will generate:

```
tutorial-data/          # Created by scripts, contains downloaded data and results
├── shortclips/         # Downloaded experimental data
│   ├── features/       # Motion energy and WordNet features
│   ├── mappers/        # Cortical surface mappers per subject
│   ├── responses/      # fMRI responses per subject
│   ├── stimuli/        # Training and test stimuli
│   └── utils/          # WordNet categories and graph structure
├── results/           # Fitted model results
│   └── {subject}_bandedridge.hdf  # Cross-validation scores, model weights
└── figures/           # Generated visualizations per subject
    └── {subject}/
        ├── {subject}_ev.png                    # Explained variance visualization
        ├── {subject}_joint_r2_scores.png       # Joint model R² scores
        ├── {subject}_split_r2_cvscores.png     # Split model cross-validation scores
        ├── {subject}_split_r2_scores.png       # Split model R² scores
        ├── {subject}_wordnet_flatmap_pc1.png   # WordNet PC1 flatmap visualization
        ├── {subject}_wordnet_flatmap_pc234.png # WordNet PC2-4 flatmap visualization
        └── {subject}_wordnet_graph_*.png       # WordNet semantic graphs for specific ROIs
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{vem-review,
  title={Encoding models in functional magnetic resonance imaging: the Voxelwise Encoding Model framework},
  author={{Visconti di Oleggio Castello}, Matteo and Deniz, Fatma and {Dupré la Tour}, Tom and Gallant, Jack L.},
  journal={PsyArXiv},
  year={2025},
  doi={10.31234/osf.io/nt2jq_v1},
  url={https://doi.org/10.31234/osf.io/nt2jq_v1}
}
```

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE.md file for details.

## Support

For questions or issues, please open an issue on the GitHub repository.

## Acknowledgments

This work builds upon these existing neuroimaging analysis tools:
- [Himalaya](https://github.com/gallantlab/himalaya) for ridge regression
- [Pycortex](https://github.com/gallantlab/pycortex) for cortical visualization
- [Voxelwise Tutorials](https://github.com/gallantlab/voxelwise_tutorials) for analysis utilities
