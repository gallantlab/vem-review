import numpy as np


def get_alpha(array, q=95):
    """
    Normalize an array to create an alpha channel for visualization.

    Parameters
    ----------
    array : numpy.ndarray
        The input array to normalize.
    q : int, optional
        Percentile value for normalization (0-100), default is 95.

    Returns
    -------
    numpy.ndarray
        Normalized array with values clipped between 0 and 1.
        NaN values in the input are converted to 0.

    Notes
    -----
    The function divides the array by its q-th percentile, effectively
    normalizing it so that values at the q-th percentile become 1.0.
    """
    alpha = array.copy()
    nans = np.isnan(alpha)
    alpha /= np.nanpercentile(alpha, q)
    alpha[nans] = 0
    alpha = np.clip(alpha, 0, 1)
    return alpha


def scale_weights(weights, r2_scores):
    """Scale the regression coefficients according to the R2 scores.

    Parameters
    ----------
    weights : array of shape (n_features, n_voxels)
        Regression coefficients.
    r2_scores : array of shape (n_voxels, )
        Prediction accuracy scores.

    Returns
    -------
    weights : array of shape (n_features, n_voxels)
        Scaled regression weights.

    Notes
    -----
    The scaling is performed in two steps:
    1. Normalize the weights to have unit norm along the feature dimension:
       w_j = w_j / ||w_j||   (where w_j is the weight vector for voxel j)
    2. Scale by the square root of the R2 score:
       w_j = w_j * sqrt(max(0, R2_j))
    """
    norm = np.linalg.norm(weights, axis=0)
    weights[:, norm != 0] /= norm[norm != 0]
    weights *= np.sqrt(np.maximum(0, r2_scores))
    return weights
