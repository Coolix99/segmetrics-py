import numpy as np

def is_array_of_integers(y):
    return isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.integer)

def label_are_sequential(y):
    """Returns true if y has only sequential labels from 1..."""
    labels = np.unique(y)
    return (set(labels) - {0}) == set(range(1, 1 + labels.max()))

def _check_label_array(y, name=None, check_sequential=False):
    """Validate that the array y is of non-negative integers and optionally sequential."""
    label_type = 'sequential ' if check_sequential else ''
    err_msg = f"{'Labels' if name is None else name} must be an array of {label_type}non-negative integers."
    
    if not is_array_of_integers(y):
        raise ValueError(err_msg)
    if len(y) == 0:
        return True
    if check_sequential and not label_are_sequential(y):
        raise ValueError(err_msg)
    elif y.min() < 0:
        raise ValueError(err_msg)
    return True
