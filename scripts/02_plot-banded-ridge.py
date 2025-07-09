import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from sklearn.decomposition import PCA
from voxelwise_tutorials.io import load_hdf5_array
from voxelwise_tutorials.viz import (
    plot_2d_flatmap_from_mapper,
    plot_3d_flatmap_from_mapper,
    plot_flatmap_from_mapper,
)
from voxelwise_tutorials.wordnet import (
    DEFAULT_HIGHLIGHTED_NODES,
    apply_cmap,
    correct_coefficients,
    load_wordnet,
    plot_wordnet_graph,
    scale_to_rgb_cube,
)

from vemreview.config import figures_dir, results_dir, shortclips_dir
from vemreview.io import load_ev, load_mapper
from vemreview.utils import get_alpha, scale_weights

########################################################################################
# Setup

args = sys.argv[1:]
if len(args) == 0:
    subject = "S01"
else:
    subject = args[0]
if subject not in ["S01", "S02", "S03", "S04", "S05"]:
    raise ValueError("subject must be in ['S01', 'S02', 'S03', 'S04', 'S05']")
results_fn = os.path.join(results_dir, f"{subject}_bandedridge.hdf")
results = load_hdf5_array(results_fn)
figures_dir = os.path.join(figures_dir, subject)
os.makedirs(figures_dir, exist_ok=True)

voxels_to_show = {
    "S01": {
        "lPPA": 19911,
        "rFFA": 6526,
        "rPrecun": 59144,
        "rpSTS": 43933,
        "lpSTS": 40305,
        "rRSC": 40211,
    }
}

mapper, mapper_file = load_mapper(subject)

# Load wordnet
wordnet_graph, wordnet_categories = load_wordnet(directory=shortclips_dir)


########################################################################################
# Functions

# Manual renaming of some labels
rename_labels = {
    "geological_formation": "geol. formation",
    "wheeled_vehicle": "wheeled\nvehicle",
    "atmospheric_phenomenon": "atmospheric\nphenomenon",
}


def get_pos_default_highlighted_nodes():
    # Get the position of all nodes
    highlighted_node_positions = [
        wordnet_graph.nodes(data="pos")[node] for node in DEFAULT_HIGHLIGHTED_NODES
    ]
    # convert to numpy array
    highlighted_node_positions = np.array(
        [
            list(map(float, pos.strip('"').split(",")))
            for pos in highlighted_node_positions
        ]
    )
    return highlighted_node_positions


def plot_wordnet_graph_with_styles(
    node_sizes, node_colors, font_size=16, adjust_texts=True
):
    ax = plot_wordnet_graph(
        node_colors=node_colors, node_sizes=node_sizes, font_size=font_size
    )
    # Extract the texts
    texts = ax.texts
    # Fix labels so that underscores are replaced with spaces
    # And add 60 to the y position because plot_wordnet_graph plots the labels at y - 60
    # If I don't add 60 back, then the lines connecting the labels to the nodes are off
    for text in texts:
        label = text.get_text()
        label = rename_labels.get(label, label)
        text.set_text(label.replace("_", " "))
        text.set_y(text.get_position()[1] + 60)
    if adjust_texts:
        # Avoid overlapping with highlighted nodes
        default_node_pos = get_pos_default_highlighted_nodes()
        # Adjust the texts
        adjust_text(
            texts,
            x=default_node_pos[:, 0],
            y=default_node_pos[:, 1],
            ensure_inside_axes=True,
            arrowprops=dict(arrowstyle="-", lw=1, color="w"),
            min_arrow_len=10,
        )
        fig = ax.get_figure()
    return fig


########################################################################################
# Plot EV
if False:
    ev = load_ev(subject)
    alpha = get_alpha(ev)
    ax = plot_flatmap_from_mapper(
        ev, mapper_file, alpha=alpha, vmin=0.0, vmax=0.6, cmap="viridis"
    )
    ax.set_title(f"{subject} Explainable Variance", fontsize="x-large")
    fig = ax.get_figure()
    fig.savefig(
        os.path.join(figures_dir, f"{subject}_ev.png"), dpi=300, transparent=True
    )


########################################################################################
# Plot 2D flatmap with the full score
if False:
    full_scores = results[f"{subject}_joint_r2_scores"]
    full_scores[full_scores < 0] = 0
    alpha = get_alpha(full_scores)
    ax = plot_flatmap_from_mapper(
        full_scores,
        mapper_file,
        alpha=alpha,
        vmin=0,
        vmax=0.6,
        cmap="inferno",
        with_rois=False,
        with_colorbar=False,
    )
    ax.set_title(f"{subject} Joint R2 Scores", fontsize="x-large")
    fig = ax.get_figure()
    fig.savefig(
        os.path.join(figures_dir, f"{subject}_joint_r2_scores.png"),
        dpi=300,
        transparent=True,
    )

########################################################################################
# Plot 2D flatmap with the test split scores
if False:
    split_scores = results[f"{subject}_split_r2_scores"]
    full_scores = results[f"{subject}_joint_r2_scores"]
    full_scores[full_scores < 0] = 0
    alpha = get_alpha(full_scores)
    ax = plot_2d_flatmap_from_mapper(
        split_scores[1],
        split_scores[0],
        mapper_file,
        alpha=alpha,
        vmin=0,
        vmax=0.5,
        vmin2=0,
        vmax2=0.25,
        label_1=results["feature_names"][1].decode(),
        label_2=results["feature_names"][0].decode(),
        cmap="PU_BuOr_covar",
    )
    ax.set_title(f"{subject} Split R2 Scores", fontsize="x-large")
    fig = ax.get_figure()
    fig.savefig(
        os.path.join(figures_dir, f"{subject}_split_r2_scores.png"),
        dpi=300,
        transparent=True,
    )

########################################################################################
# Plot 2D flatmap with the train CV split scores
if False:
    split_scores = np.nanmean(results[f"{subject}_split_r2_cvscores"], 0)
    full_scores = split_scores.sum(0)
    full_scores[full_scores < 0] = 0
    alpha = get_alpha(full_scores)
    ax = plot_2d_flatmap_from_mapper(
        split_scores[1],
        split_scores[0],
        mapper_file,
        alpha=alpha,
        vmin=0,
        vmax=0.5,
        vmin2=0,
        vmax2=0.25,
        label_1=results["feature_names"][1].decode(),
        label_2=results["feature_names"][0].decode(),
        cmap="PU_BuOr_covar",
    )
    ax.set_title(f"{subject} Split R2 CV Scores", fontsize="x-large")
    fig = ax.get_figure()
    fig.savefig(
        os.path.join(figures_dir, f"{subject}_split_r2_cvscores.png"),
        dpi=300,
        transparent=True,
    )

########################################################################################
# Run PCA and plot PCs

# Load the weights and rescale them by the training CV scores
wordnet_weights = results[f"{subject}_weights_wordnet"].astype(float)
print("wordnet_weights.shape =", wordnet_weights.shape, "(n_features, n_voxels)")

feature_idx = list(results["feature_names"]).index(b"wordnet")
cvscores = results[f"{subject}_split_r2_cvscores"][:, feature_idx].astype(float)
cvscores = np.nanmean(cvscores, axis=0)
print("max cvscores =", cvscores.max())
wordnet_weights = scale_weights(wordnet_weights, cvscores)


# Compute a common alpha based on the norm of the weights for plotting
alpha = get_alpha(np.linalg.norm(wordnet_weights, axis=0))


if False and subject in voxels_to_show:
    # Plot wordnet graph for some specific voxels
    voxels = voxels_to_show[subject]
    for roi, idx in voxels.items():
        # Take the weight and correct it following Huth et al., 2012
        weight = wordnet_weights[:, idx]
        weight = correct_coefficients(weight, wordnet_categories)
        # Normalize it for plotting
        weight -= weight.mean()
        weight /= weight.std()
        node_sizes = np.abs(weight)
        node_colors = apply_cmap(weight, vmin=-2, vmax=2, cmap="coolwarm", n_colors=2)

        # WordNet graph for this voxel
        fig = plot_wordnet_graph_with_styles(node_sizes, node_colors)
        fig.savefig(
            os.path.join(figures_dir, f"{subject}_wordnet_graph_{roi}.png"),
            dpi=300,
            facecolor="k",
        )


if True:
    # Now perform PCA
    pca = PCA(n_components=4)
    pca.fit(wordnet_weights.T)
    components = pca.components_
    print("components.shape =", components.shape, "(n_components, n_features)")
    print("PCA explained variance =", pca.explained_variance_ratio_)

    # Project the weights on the PCs
    wordnet_weights_transformed = pca.transform(wordnet_weights.T).T
    # Compute a common vmax
    vmax = np.percentile(np.abs(wordnet_weights_transformed), 99.9)

    # Correct coefficients as in Huth et al., 2012
    components = correct_coefficients(components.T, wordnet_categories).T
    components -= components.mean(axis=1)[:, None]
    components /= components.std(axis=1)[:, None]

    ####################################################################################
    # Plot the first PC
    first_component = components[0]
    node_sizes = np.abs(first_component)
    node_colors = apply_cmap(
        first_component, vmin=-2, vmax=2, cmap="coolwarm", n_colors=2
    )

    # WordNet graph of the first PC
    fig = plot_wordnet_graph_with_styles(node_sizes, node_colors)
    fig.savefig(
        os.path.join(figures_dir, f"{subject}_wordnet_graph_pc1.png"),
        dpi=300,
        facecolor="k",
    )

    # Transformed weights on the first PC plotted on the flatmap
    ax = plot_flatmap_from_mapper(
        wordnet_weights_transformed[0],
        mapper_file,
        alpha=alpha,
        vmin=-vmax,
        vmax=vmax,
        cmap="coolwarm",
    )
    fig = ax.get_figure()
    fig.savefig(
        os.path.join(figures_dir, f"{subject}_wordnet_flatmap_pc1.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )

    ####################################################################################
    # Plot the second, third, and fourth PCs with an RGB colormap
    flip = np.array([-1, 1, 1])
    next_three_components = flip * components[1:4].T
    node_sizes = np.linalg.norm(next_three_components, axis=1)
    node_colors = scale_to_rgb_cube(next_three_components, clip=2)
    print("node_colors.shape =", node_colors.shape, "(n_nodes, n_channels)")

    # WordNet graph of the three PCs
    fig = plot_wordnet_graph_with_styles(node_sizes, node_colors)
    fig.savefig(
        os.path.join(figures_dir, f"{subject}_wordnet_graph_pc234.png"),
        dpi=300,
        facecolor="k",
    )

    # Transformed weights on the three PCs plotted on the flatmap
    voxel_colors = scale_to_rgb_cube(
        flip * wordnet_weights_transformed[1:4].T, clip=3
    ).T
    print("(n_channels, n_voxels) =", voxel_colors.shape)

    ax = plot_3d_flatmap_from_mapper(
        voxel_colors[0],
        voxel_colors[1],
        voxel_colors[2],
        alpha=alpha,
        mapper_file=mapper_file,
        vmin=0,
        vmax=1,
        vmin2=0,
        vmax2=1,
        vmin3=0,
        vmax3=1,
    )
    fig = ax.get_figure()
    fig.savefig(
        os.path.join(figures_dir, f"{subject}_wordnet_flatmap_pc234.png"),
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
