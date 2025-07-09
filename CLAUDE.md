# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a scientific research codebase for reproducing figures from a Voxelwise Encoding Model (VEM) review paper. The project uses fMRI data from the Gallant Lab to fit banded ridge regression models using motion energy and WordNet semantic features for predicting neural responses.

## Installation and Setup

This is a Python package that must be installed in development mode:

```bash
cd src
pip install -e .
```

The package automatically handles data downloading from https://gin.g-node.org/gallantlab/shortclips via datalad when scripts are run.

## Key Commands

### Model Fitting
```bash
# Fit banded ridge models for a subject (S01-S05)
python scripts/01_fit-banded-ridge.py S01

# Generate visualizations for fitted models
python scripts/02_plot-banded-ridge.py S01
```

### Testing
No automated test suite is configured. The package includes pytest in extras_require but no tests are present.

## Architecture

### Core Package (`src/vemreview/`)
- **`config.py`**: Defines data directory paths (DATA_DIR, shortclips_dir, results_dir, figures_dir)
- **`io.py`**: Data loading functions with automatic datalad downloading
  - `load_data_for_fitting()`: Main function that loads and preprocesses fMRI responses and features
  - `load_features()`: Loads motion energy and WordNet features
  - `load_mapper()`: Loads cortical surface mappers for visualization
- **`utils.py`**: Visualization utilities
  - `get_alpha()`: Creates alpha channels for visualization
  - `scale_weights()`: Scales regression coefficients by R² scores

### Analysis Scripts (`scripts/`)
- **`01_fit-banded-ridge.py`**: Main model fitting script
  - Uses Himalaya library for banded ridge regression
  - Supports GPU acceleration via PyTorch CUDA backend
  - Implements leave-one-run-out cross-validation
  - Saves results to `tutorial-data/results/{subject}_bandedridge.hdf`
- **`02_plot-banded-ridge.py`**: Visualization generation
  - Creates cortical surface visualizations using pycortex
  - Generates WordNet semantic space visualizations
  - Saves figures to `tutorial-data/figures/{subject}/`

### Notebooks (`notebooks/`)
- **`example-fir-model.ipynb`**: Finite impulse response model fitting example
- **`simulate-noise-ceiling.ipynb`**: Noise ceiling analysis simulation
- **`utils.py`**: Notebook-specific utility functions

## Data Processing Pipeline

1. **Data Loading**: `load_data_for_fitting()` automatically downloads and loads:
   - fMRI responses for subjects S01-S05
   - Motion energy features (spatiotemporal Gabor filters)
   - WordNet semantic features (hierarchical categories)
   - Cortical surface mappers

2. **Preprocessing**:
   - Z-score fMRI responses within runs
   - Average test responses across repetitions
   - Demean features within runs
   - Cast to float32 for GPU processing

3. **Model Fitting**:
   - Banded ridge regression with multiple kernels
   - Leave-one-run-out cross-validation
   - Hyperparameter optimization via grid search

4. **Evaluation**:
   - Correlation scores between predicted and actual responses
   - R² scores for model performance assessment

## Key Dependencies

- **Himalaya**: Ridge regression and kernel methods
- **PyTorch**: GPU acceleration for model fitting
- **Pycortex**: Cortical surface visualization
- **Voxelwise Tutorials**: Data loading and analysis utilities
- **Datalad**: Automatic data downloading
- **Pymoten**: Motion energy feature extraction

## File Structure

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

## Subject Data

The codebase works with 5 subjects: S01, S02, S03, S04, S05. All scripts accept subject ID as command-line argument, defaulting to S01.

## GPU Support

The model fitting script automatically attempts to use CUDA if available via `set_backend("torch_cuda", on_error="warn")`. Falls back to CPU if GPU is unavailable.