"""This scripts fits a banded ridge model using motion-energy and WordNet features.
It saves the wordnet weights and the model scores."""

import os
import sys

import numpy as np
import torch
from himalaya.backend import set_backend
from himalaya.kernel_ridge import (
    ColumnKernelizer,
    Kernelizer,
    MultipleKernelRidgeCV,
    WeightedKernelRidge,
)
from himalaya.progress_bar import bar
from himalaya.scoring import correlation_score, correlation_score_split, r2_score_split
from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer
from voxelwise_tutorials.io import save_hdf5_dataset
from voxelwise_tutorials.utils import generate_leave_one_run_out

from vemreview.config import results_dir
from vemreview.io import load_data_for_fitting

###############################################################################
# Set up backend and directories
backend = set_backend("torch_cuda", on_error="warn")
os.makedirs(results_dir, exist_ok=True)

###############################################################################
# Allow to pass a subject as a command-line argument
args = sys.argv[1:]
if len(args) == 0:
    subject = "S01"
else:
    subject = args[0]
if subject not in ["S01", "S02", "S03", "S04", "S05"]:
    raise ValueError("subject must be in ['S01', 'S02', 'S03', 'S04', 'S05']")

###############################################################################
# Load the data
(
    X_train,
    X_test,
    Y_train,
    Y_test,
    run_onsets,
    n_features_list,
    feature_names,
) = load_data_for_fitting(subject)

# Set nans to 0
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
Y_train = np.nan_to_num(Y_train)
Y_test = np.nan_to_num(Y_test)

###############################################################################
# Define the cross-validation scheme
n_samples_train = X_train.shape[0]
cv = generate_leave_one_run_out(n_samples_train, run_onsets)
cv = check_cv(cv)  # copy the cross-validation splitter into a reusable list

###############################################################################
# Define the pipeline
#
# First, Banded ridge
solver = "random_search"
n_iter = 400
alphas = np.logspace(1, 20, 20)
# Note: the following parameters should be changed depending on VRAM size
n_targets_batch = 40000
n_alphas_batch = 20
n_targets_batch_refit = 40000
solver_params = dict(
    n_iter=n_iter,
    alphas=alphas,
    n_targets_batch=n_targets_batch,
    n_alphas_batch=n_alphas_batch,
    n_targets_batch_refit=n_targets_batch_refit,
)
mkr_model = MultipleKernelRidgeCV(
    kernels="precomputed",
    solver=solver,
    solver_params=solver_params,
    cv=cv,
    random_state=42,
)
# Then add delays
delayer = Delayer(delays=[1, 2, 3, 4])
# Make preprocessing pipeline
preprocess_pipeline = make_pipeline(
    StandardScaler(with_mean=True, with_std=False),
    delayer,
    Kernelizer(kernel="linear"),
)
# Now make the column kernelizer for banded ridge
# Find the start and end of each feature space in the concatenated ``X_train``.
start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
slices = [
    slice(start, end) for start, end in zip(start_and_end[:-1], start_and_end[1:])
]
kernelizers_tuples = [
    (name, preprocess_pipeline, slice_) for name, slice_ in zip(feature_names, slices)
]
column_kernelizer = ColumnKernelizer(kernelizers_tuples)
# And finally define the full pipeline
pipeline = make_pipeline(
    column_kernelizer,
    mkr_model,
)

###############################################################################
# Fit the model
pipeline.fit(X_train, Y_train)
Y_pred = pipeline.predict(X_test)
Y_pred_split = pipeline.predict(X_test, split=True)

###############################################################################
# Store results
results = {
    "feature_names": feature_names,
    f"{subject}_joint_r2_scores": pipeline.score(X_test, Y_test),
    f"{subject}_joint_r_scores": correlation_score(Y_test, Y_pred),
    f"{subject}_split_r2_scores": r2_score_split(Y_test, Y_pred_split),
    f"{subject}_split_r_scores": correlation_score_split(Y_test, Y_pred_split),
}

# Extract the weights for wordnet
Xs_fit = column_kernelizer.get_X_fit()
primal_weights = mkr_model.get_primal_coef(Xs_fit)
# average over delays
primal_weights = [
    delayer.reshape_by_delays(weights, axis=0) for weights in primal_weights
]
primal_weights = [this.mean(axis=0) for this in primal_weights]

features_to_save = ["wordnet"]
for feature_name, primal_weights_ii in zip(feature_names, primal_weights):
    if feature_name not in features_to_save:
        continue
    results[f"{subject}_weights_{feature_name}"] = backend.to_numpy(
        primal_weights_ii
    ).astype(np.float32)

###############################################################################
# Recompute the cross-validation scores for the best alpha
cv_scores_r2 = []
cv_scores_r = []
cv_splits = list(cv.split(X=X_train, y=Y_train))
for train, val in bar(cv_splits, "Recomputing CV scores"):
    x_train, x_val = X_train[train], X_train[val]
    y_train, y_val = Y_train[train], Y_train[val]
    wkr = WeightedKernelRidge(
        alpha=1.0,
        deltas=torch.nan_to_num(mkr_model.deltas_),
        kernels="precomputed",
        solver_params=dict(n_targets_batch=n_targets_batch),
    )
    pipeline_wkr = make_pipeline(column_kernelizer, wkr)
    pipeline_wkr.fit(x_train, y_train)
    ypred = pipeline_wkr.predict(x_val, split=True)
    cv_scores_r2.append(backend.to_numpy(r2_score_split(y_val, ypred)))
    cv_scores_r.append(backend.to_numpy(correlation_score_split(y_val, ypred)))
cv_scores_r2 = np.stack(cv_scores_r2)
cv_scores_r = np.stack(cv_scores_r)
assert cv_scores_r2.ndim == 3
assert cv_scores_r.ndim == 3

results.update(
    {
        f"{subject}_split_r2_cvscores": cv_scores_r2,
        f"{subject}_split_r_cvscores": cv_scores_r,
    }
)

# Convert to numpy
for key, value in results.items():
    if isinstance(value, torch.Tensor):
        results[key] = backend.to_numpy(value).astype(np.float32)

# Save results
save_hdf5_dataset(os.path.join(results_dir, f"{subject}_bandedridge.hdf"), results)
