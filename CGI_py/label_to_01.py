"""
Label to 01 - Convert string labels to binary (0/1) format
"""


def label_to_01(labels, target_substring):
    """
    Convert string labels to binary (0/1) format.

    Args:
        labels: array of strings
        target_substring: substring to match for class 1

    Returns:
        Binary labels array (0s and 1s)
    """
    import numpy as np
    labels = np.array(labels).flatten()
    c = np.ones(len(labels), dtype=int)

    for i in range(len(labels)):
        if target_substring in str(labels[i]):
            c[i] = 0

    return c


def load_data_with_labels(mat_file, label_key='labels'):
    """
    Load data and labels from .mat file and create binary labels.

    Args:
        mat_file: path to .mat file
        label_key: key for labels in .mat file

    Returns:
        Tuple of (data, labels)
    """
    from scipy.io import loadmat
    data = loadmat(mat_file)
    labels = data.get(label_key, data.get('label', None))
    if labels is None:
        raise ValueError(f'Could not find labels in {mat_file}')
    return data['data'] if 'data' in data else data['d'], labels
