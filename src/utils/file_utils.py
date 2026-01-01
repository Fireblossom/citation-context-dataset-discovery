"""
File utility functions for saving and managing output files
"""

import os
import json
import time
import math
from datetime import datetime
from typing import Any

from .numpy_encoder import NumpyEncoder


def clean_data_for_json(data: Any) -> Any:
    """
    Clean data for JSON serialization.
    Handles pandas NaN values and other non-serializable objects.

    Args:
        data: Any data structure to clean

    Returns:
        Cleaned data suitable for JSON serialization
    """
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    elif hasattr(data, "tolist"):  # numpy array
        return data.tolist()
    elif hasattr(data, "item"):  # numpy scalar
        return data.item()
    else:
        return data


def create_output_filename(
    query: str, context_type: str, limit: int = None, include_timestamp: bool = False
) -> str:
    """
    Create a standardized output filename.

    Args:
        query: Search query string
        context_type: Type of context ('citing', 'cited', 'both')
        limit: Optional paper count limit
        include_timestamp: Whether to add timestamp to filename

    Returns:
        Generated filename with .json extension
    """
    # Clean query string for use as filename
    safe_query = query.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    safe_query = "".join(c for c in safe_query if c.isalnum() or c in "_").strip("_")

    # Generate base filename
    if context_type == "citing":
        base_name = f"{safe_query}_citing_contexts"
        if limit:
            base_name += f"_{limit}"
    elif context_type == "cited":
        base_name = f"{safe_query}_cited_contexts"
        if limit:
            base_name += f"_{limit}"
    elif context_type == "both":
        base_name = f"{safe_query}_combined_contexts"
        if limit:
            base_name += f"_{limit}"
    else:
        base_name = f"{safe_query}_contexts"

    # Add timestamp if requested
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name += f"_{timestamp}"

    return f"{base_name}.json"


def save_json_with_backup(
    data: Any,
    filepath: str,
    backup_existing: bool = True,
    use_numpy_encoder: bool = True,
) -> bool:
    """
    Save data to JSON file with optional backup of existing file.

    Args:
        data: Data to save
        filepath: Output file path
        backup_existing: Whether to backup existing file
        use_numpy_encoder: Whether to use NumpyEncoder for numpy types

    Returns:
        True if save was successful
    """
    try:
        # Backup existing file
        if backup_existing and os.path.exists(filepath):
            backup_suffix = 0
            while True:
                if backup_suffix == 0:
                    backup_path = f"{filepath}.backup"
                else:
                    backup_path = f"{filepath}.backup_{backup_suffix}"

                if not os.path.exists(backup_path):
                    os.rename(filepath, backup_path)
                    print(f"Existing file backed up to: {backup_path}")
                    break
                backup_suffix += 1

        # Save JSON
        encoder = NumpyEncoder if use_numpy_encoder else None
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, cls=encoder)

        print(f"Data saved to: {filepath}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to save file: {e}")
        return False
