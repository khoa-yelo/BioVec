from typing import Any, Dict

import numpy as np

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None  # type: ignore[assignment]


def write_h5(path: str, arrays: Dict[str, Any]) -> None:
    """
    Write arrays to an HDF5 file.

    Args:
        arrays: mapping of dataset name -> array-like
    """
    if h5py is None:
        raise ImportError("h5py is required to write HDF5 files.")
    with h5py.File(path, "w") as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=np.asarray(arr))


def read_h5(path: str) -> Dict[str, np.ndarray]:
    """
    Read all datasets from an HDF5 file.

    Returns:
        Dict of dataset name -> numpy array
    """
    if h5py is None:
        raise ImportError("h5py is required to read HDF5 files.")
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for name in f.keys():
            out[name] = f[name][:]
    return out

