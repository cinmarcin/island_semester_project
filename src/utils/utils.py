import os
import yaml
import numpy as np



def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def make_dir(path):
    os.makedirs(path, exist_ok=True)


def json_safe(obj):
    """Recursively convert numpy/scipy/statistical results into JSON-safe formats."""
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle scipy.stats result objects like SignificanceResult
    if hasattr(obj, "statistic") and hasattr(obj, "pvalue"):
        return {"statistic": float(obj.statistic), "pvalue": float(obj.pvalue)}

    # Handle dictionaries and lists recursively
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(x) for x in obj]

    # Handle numpy scalar types
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # Handle basic JSON-safe types
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj

    # Fallback: convert everything else to string
    return str(obj)