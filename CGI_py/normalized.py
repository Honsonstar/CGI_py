"""
Normalized - Data normalization utilities
"""


def normalize(data):
    """
    Normalize data by z-score standardization.

    Args:
        data: numpy array of shape (n_samples, n_features)

    Returns:
        Normalized data array
    """
    import numpy as np
    normalized = np.zeros_like(data)
    for i in range(data.shape[1]):
        col = data[:, i]
        if np.std(col) > 0:
            normalized[:, i] = (col - np.mean(col)) / np.std(col)
        else:
            normalized[:, i] = col - np.mean(col)
    return normalized


def load_and_normalize(mat_file):
    """
    Load data from .mat file and normalize.

    Args:
        mat_file: path to .mat file

    Returns:
        Normalized data array
    """
    from scipy.io import loadmat
    data = loadmat(mat_file)
    # Try common variable names
    for key in ['d', 'data', 'D', 'normalized']:
        if key in data:
            return normalize(data[key])
    raise ValueError(f'Could not find data variable in {mat_file}')
