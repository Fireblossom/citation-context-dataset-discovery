"""
Custom JSON encoder for numpy types
"""

import json

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy and pandas types.

    Usage:
        json.dump(data, f, cls=NumpyEncoder)
    """

    def default(self, obj):
        # Handle pandas NA types
        if HAS_PANDAS:
            if pd.isna(obj):
                return None
        if HAS_NUMPY:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
        return super().default(obj)
