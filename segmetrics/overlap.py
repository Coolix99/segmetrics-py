import numpy as np
from numba import jit
from segmetrics.utils import _check_label_array

def label_overlap(x, y, check=True):
    """
    Compute the overlap matrix between two label images `x` and `y`.

    Parameters
    ----------
    x : ndarray
        First label image, where each unique integer represents a different object.
    y : ndarray
        Second label image, where each unique integer represents a different object.
    check : bool, optional
        If True, checks that `x` and `y` are valid label images with sequential non-negative integers.
    
    Returns
    -------
    ndarray
        A 2D array where entry (i, j) is the number of pixels where label i in `x`
        overlaps with label j in `y`.
    """
    if check:
        _check_label_array(x, 'x', check_sequential=True)
        _check_label_array(y, 'y', check_sequential=True)
        
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
    
    return _label_overlap(x, y)

@jit(nopython=True)
def _label_overlap(x, y):
    """
    Internal function to calculate overlap between two flattened label images.
    
    Parameters
    ----------
    x : ndarray
        Flattened version of the first label image.
    y : ndarray
        Flattened version of the second label image.
    
    Returns
    -------
    ndarray
        Overlap matrix where each entry (i, j) is the count of pixels where label i in `x`
        overlaps with label j in `y`.
    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint32)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap
# overlap.py
import numpy as np
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor
from segmetrics.utils import _check_label_array

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None

@jit(nopython=True)
def _label_overlap_numba(x, y):
    """Numba JIT-compiled version of label overlap."""
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint32)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap

def label_overlap(x, y, check=True, method="numba"):
    """
    Compute the overlap matrix between two label images `x` and `y`.

    Parameters
    ----------
    x : ndarray
        First label image, where each unique integer represents a different object.
    y : ndarray
        Second label image, where each unique integer represents a different object.
    check : bool, optional
        If True, checks that `x` and `y` are valid label images with sequential non-negative integers.
    method : str, optional
        Method to use for calculating overlap ('numba', 'numpy', 'numpy_threaded', 'numba_threaded', 'cupy', 'cuda').
    
    Returns
    -------
    ndarray
        Overlap matrix where entry (i, j) is the number of pixels where label i in `x`
        overlaps with label j in `y`.
    """
    if check:
        _check_label_array(x, 'x', check_sequential=True)
        _check_label_array(y, 'y', check_sequential=True)
        
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
    
    if method == "numba":
        return _label_overlap_numba(x, y)
    elif method == "numpy":
        return _label_overlap_numpy(x, y)
    elif method == "numpy_threaded":
        return _label_overlap_numpy_threaded(x, y)
    elif method == "numba_threaded":
        return _label_overlap_numba_threaded(x, y)
    elif method == "cupy" and cp is not None:
        return _label_overlap_cupy(x, y)
    elif method == "cuda":
        return _label_overlap_cuda(x, y)
    elif method == "torch" and torch is not None:
        return _label_overlap_torch(x, y)
    else:
        raise ValueError(f"Unknown or unsupported method: {method}")

def _label_overlap_torch(x, y):
    """
    PyTorch-based implementation of label overlap.
    
    Parameters
    ----------
    x : ndarray
        First label image.
    y : ndarray
        Second label image.
    
    Returns
    -------
    torch.Tensor
        Overlap matrix where entry (i, j) is the count of pixels where label i in `x`
        overlaps with label j in `y`.
    """
    # Convert numpy arrays to torch tensors and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x.ravel(), device=device, dtype=torch.int64)
    y = torch.tensor(y.ravel(), device=device, dtype=torch.int64)

    # Determine the size of the overlap matrix
    max_x, max_y = x.max().item() + 1, y.max().item() + 1
    overlap = torch.zeros((max_x, max_y), device=device, dtype=torch.int32)

    # Use advanced indexing to count overlaps
    indices = torch.stack((x, y), dim=0)
    overlap.index_put_(tuple(indices), torch.ones_like(x, dtype=torch.int32), accumulate=True)

    # Move result back to CPU for compatibility with numpy if needed
    return overlap.cpu()

# Pure NumPy Implementation
def _label_overlap_numpy(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint32)
    for xi, yi in zip(x, y):
        overlap[xi, yi] += 1
    return overlap

# NumPy with Multi-threading
def _label_overlap_numpy_threaded(x, y):
    x = x.ravel()
    y = y.ravel()
    max_x = x.max() + 1
    max_y = y.max() + 1
    overlap = np.zeros((max_x, max_y), dtype=np.uint32)

    def process_range(start, end):
        local_overlap = np.zeros_like(overlap)
        for i in range(start, end):
            local_overlap[x[i], y[i]] += 1
        return local_overlap

    # Split work among threads
    num_threads = 8  # Adjust based on system
    chunk_size = len(x) // num_threads
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_threads)]
    ranges[-1] = (ranges[-1][0], len(x))  # Last chunk goes to the end

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(lambda r: process_range(*r), ranges)

    # Aggregate results
    for res in results:
        overlap += res
    return overlap

# Multi-threaded Numba Version
@jit(nopython=True, parallel=True)
def _label_overlap_numba_threaded(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint32)
    for i in prange(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap

# CuPy (GPU-accelerated)
def _label_overlap_cupy(x, y):
    if cp is None:
        raise ImportError("CuPy is not installed.")
    x = cp.asarray(x.ravel())
    y = cp.asarray(y.ravel())
    overlap = cp.zeros((1 + x.max().get(), 1 + y.max().get()), dtype=cp.uint32)
    for xi, yi in zip(x, y):
        overlap[xi, yi] += 1
    return overlap.get()

# GPU-accelerated Numba CUDA Version
@jit
def _label_overlap_cuda(x, y):
    from numba import cuda

    # Allocate overlap matrix on the device
    max_x = int(x.max()) + 1
    max_y = int(y.max()) + 1
    overlap = np.zeros((max_x, max_y), dtype=np.uint32)
    d_overlap = cuda.to_device(overlap)

    # Flatten inputs and move them to device
    x = x.ravel()
    y = y.ravel()
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)

    # CUDA kernel for calculating overlap
    @cuda.jit
    def cuda_overlap(d_x, d_y, d_overlap):
        idx = cuda.grid(1)
        if idx < d_x.size:
            xi = d_x[idx]
            yi = d_y[idx]
            cuda.atomic.add(d_overlap, (xi, yi), 1)

    # Define threads and blocks
    threads_per_block = 256
    blocks_per_grid = (x.size + (threads_per_block - 1)) // threads_per_block
    cuda_overlap[blocks_per_grid, threads_per_block](d_x, d_y, d_overlap)

    # Copy result back to host
    return d_overlap.copy_to_host()
